import itertools
import logging
import os
import re
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple, Union

import mat73
import numpy as np
from matplotlib import pyplot as plt
from numpy import ma as ma
from numpy.typing import ArrayLike, NDArray
from scipy.io import loadmat
from skimage.measure import block_reduce

import QDMpy
import QDMpy.utils
from QDMpy.exceptions import WrongFileNumber


class ODMR2:
    LOG = logging.getLogger("QDMpy.ODMR2")

    def __repr__(self):
        return f"ODMR2(data_shape={self._data_shape}, frequencies={self._frequencies.shape})"

    def __init__(
        self, data: NDArray, data_shape: NDArray, frequencies: NDArray, binning: float = 1, **kwargs: Any
    ) -> None:
        """Initialize the ODMR class.

        Args:
            data: np.ndarray: The data to be analyzed.
            data_shape: np.ndarray: The shape of the data.
            frequencies: np.ndarray: The frequencies of the data.
            **kwargs: Additional keyword arguments.
        """
        # raw data
        self.__data__ = data
        # accessable data
        self._data = None
        self.__mean_odmr_uncorrected = None
        self.__mean_baseline = None

        self._data_shape = tuple(data_shape)
        self._frequencies = frequencies
        self._binning = binning

        self.n_pol, self.n_frange, self.n_pixel, self.n_freq = data.shape
        self.LOG.info(f"ODMR initialized with data of shape {self._data_shape}")
        self.LOG.debug(
            f"    n_pol: {self.n_pol}, n_frange: {self.n_frange}, n_pixel: {self.n_pixel}, n_freq: {self.n_freq}"
        )

        self._mask = False
        self._outlier_mask = np.zeros(self.n_pixel, dtype=bool)

        self._norm = True
        self.norm_method = kwargs.get("norm_method", QDMpy.SETTINGS["odmr"]["norm_method"])
        self._norm_factors = self.get_norm_factors(method=self.norm_method)

        self._gf_correct = False
        self._gf_factor = kwargs.get("gf_factor", 0)
        self._gf_correction = np.zeros((self.n_pol, self.n_frange, self.n_freq))


    @property
    def norm(self):
        return self._norm
    @norm.setter
    def norm(self, value: bool) -> None:
        """Set the normalization."""
        self.LOG.debug(f"Setting norm to {value}")
        self._norm = value
        self._data = None
        self.__mean_odmr_uncorrected = None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value: bool) -> None:
        """Set the masking."""
        self.LOG.debug(f"Setting mask to {value}")
        self._mask = value
        self._data = None
        self.__mean_odmr_uncorrected = None

    @property
    def gf_correct(self):
        return self._gf_correct

    @gf_correct.setter
    def gf_correct(self, value: bool) -> None:
        """Set the global fluorescence correction."""
        self.LOG.debug(f"Setting gf_correct to {value}")
        self._gf_correct = value
        self._data = None

    # ------------------------- INDEXING ----------------------#

    def rc2idx(self, rc: ArrayLike) -> NDArray:
        """

        Args:
          rc:

        Returns:

        """
        return QDMpy.utils.rc2idx(rc, self.data_shape)  # type: ignore[arg-type]

    def idx2rc(self, idx: ArrayLike) -> Tuple[NDArray, NDArray]:
        """

        Args:
          idx:

        Returns:

        """
        return QDMpy.utils.idx2rc(idx, self.data_shape)  # type: ignore[arg-type]

    def get_most_divergent_from_mean(self) -> Tuple[int, int]:
        """Get the most divergent pixel from the mean in data coordinates."""
        delta = self.delta_mean.copy()
        delta[delta > 0.001] = np.nan
        return np.unravel_index(np.argmax(delta, axis=None), self.delta_mean.shape)  # type: ignore[return-value]

    def get_binned_pixel_indices(self, x: int, y: int) -> Tuple[Sequence[int], Sequence[int]]:
        """

        Args:
          x:
          y:

        Returns:
          :return: numpy.ndarray

        """
        idx = list(
            itertools.product(
                np.arange(y * self.bin_factor, (y + 1) * self.bin_factor),
                np.arange(x * self.bin_factor, (x + 1) * self.bin_factor),
            )
        )
        xid = [i[0] for i in idx]
        yid = [i[1] for i in idx]
        return xid, yid

    @property
    def data_shape(self) -> Tuple[int, int]:
        """Return the shape of the data."""
        return self._data_shape

    # --------------------- GLOBAL FLUORESCENCE --------------------- #
    @property
    def gf_factor(self):
        return self._gf_factor

    @gf_factor.setter
    def gf_factor(self, gf: float):
        self._gf_correct = True
        self._gf_factor = gf
        self._gf_correction = self.get_gf_correction(self._gf_factor)
        self._data = None
        self.LOG.debug(f"Setting global fluorescence factor to {gf} and gf_correct to True.")

    @property
    def _mean_odmr_uncorrected(self):
        if self.__mean_odmr_uncorrected is None:
            self.__mean_odmr_uncorrected = self.get_mean_odmr(norm=True, gf_correct=False, mask=True)
        return self.__mean_odmr_uncorrected
    @property
    def _mean_baseline(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Calculate the mean baseline of the data."""
        if self.__mean_baseline is None:
            mean_odmr = self._mean_odmr_uncorrected
            baseline_left_mean = np.mean(mean_odmr[:, :, :5], axis=-1)
            baseline_right_mean = np.mean(mean_odmr[:, :, -5:], axis=-1)
            baseline_mean = np.mean(np.stack([baseline_left_mean, baseline_right_mean], -1), axis=-1)
            self.__mean_baseline = baseline_left_mean, baseline_right_mean, baseline_mean
        return self.__mean_baseline

    def get_gf_correction(self, gf: float) -> NDArray:
        """Calculate the global fluorescence correction.

        Args:
          gf: The global fluorescence factor

        Returns: The global fluorescence correction
        """
        self.LOG.debug(f"Calculating global fluorescence correction with factor {gf}")
        baseline_left_mean, baseline_right_mean, baseline_mean = self._mean_baseline
        return gf * (self.mean_odmr - baseline_mean[:, :, np.newaxis])

    # -------------------------- OUTLIER ---------------------------- #
    @property
    def outlier_mask(self) -> NDArray:
        """Return the outlier mask.

        Returns: The outlier mask
        """
        return self._outlier_mask

    @outlier_mask.setter
    def outlier_mask(self, mask: NDArray) -> None:
        """Set the outlier mask.

        Args:
          mask: The mask to set
        """
        self.LOG.debug(f"Setting new outlier mask of shape {mask.shape}")
        if mask.ndim == 2:
            mask = mask.reshape(-1)
        if mask.size != self.n_pixel:
            raise ValueError(f"Mask size {mask.size} does not match number of pixels {self.n_pixel}.")
        self._mask = True
        self._outlier_mask = mask

    # -------------------------- NORMALIZATION ---------------------------- #
    def get_norm_factors(self, method: str = "max") -> np.ndarray:
        """Return the normalization factors for the raw data.

        Args:
          data: data
          method: return: (Default value = "max")

        Returns:

        Raises: NotImplementedError: if method is not implemented
        """
        factors = np.ones((self.n_pol, self.n_frange, self.n_pixel))

        if method is None:
            self.LOG.warning("Normalization method is None. Returning ones as normalization factors.")
        elif method == "max":
            factors = np.max(self.__data__, axis=-1)
        elif method == "allmax":
            factors *= np.max(self.__data__)
        else:
            raise NotImplementedError(f'Method "{method}" not implemented.')

        self.LOG.debug(f"Determining normalization factor with method {method}: " f"Shape of mx: {factors.shape}")
        assert factors.shape == (self.n_pol, self.n_frange, self.n_pixel)
        return factors

    @property
    def mean_odmr(self) -> NDArray:
        """Calculate the mean ODMR spectra of the final data.

        Note:
            This is the mean of the data after normalization and global fluorescence correction and outlier masking.
            Either of these can be disabled if the corresponding array is either all ones (norm) all 0 (gf) or all
            False (outlier).
        """
        return np.nanmean(self.data, axis=-2)

    @property
    def delta_mean(self) -> NDArray:
        """ """
        return np.sum(np.square(self.data - self.mean_odmr[:, :, np.newaxis, :]), axis=-1)

    # -------------------------- DATA ---------------------------- #

    @property
    def frequencies(self) -> NDArray:
        """

        Args:

        Returns:
          :return: numpy.ndarray

        """
        if self._frequencies_cropped is None:
            return self._frequencies
        else:
            return self._frequencies_cropped

    @property
    def f_hz(self) -> NDArray:
        """Returns the frequencies of the ODMR in Hz."""
        return self.frequencies

    @property
    def f_ghz(self) -> NDArray:
        """Returns the frequencies of the ODMR in GHz."""
        return self.frequencies / 1e9

    @property
    def data(self) -> NDArray:
        """Return the data."""
        if self._data is None:
            self._data = self.get_odmr(norm=self._norm, gf_correct=self._gf_correct, mask=self._mask)
        return self._data


    def get_mean_odmr(self, norm=True, gf_correct=True, mask=True):
        """Calculate the mean of the data."""
        return np.nanmean(self.get_odmr(norm, gf_correct, mask), axis=-2)

    @property
    def raw_contrast(self) -> NDArray:
        """Calculate the minimum of MW sweep for each pixel."""
        return np.min(self.data, -2)

    @property
    def mean_contrast(self) -> NDArray:
        """Calculate the mean of the minimum of MW sweep for each pixel."""
        return np.mean(self.raw_contrast)

    def get_odmr(self, norm=True, gf_correct=True, mask=True):
        """Calculate the mean of the data."""
        odmr = self.__data__.copy()
        if norm:
            odmr /= self._norm_factors[:, :, :, np.newaxis]
        if gf_correct:
            odmr -= self._gf_correction[:, :, np.newaxis, :]
        if mask:
            mask = (np.ones(self.__data__.shape) * self._outlier_mask[np.newaxis, np.newaxis, :, np.newaxis]).astype(bool)
            odmr = ma.masked_array(odmr, mask)
        return odmr


    # FROM / IMPORT
    @classmethod
    def from_qdmio(cls, data_folder: Union[str, os.PathLike], **kwargs: Any) -> "ODMR":
        """Loads QDM data from a Matlab file.

        Args:
          data_folder:

        Returns:

        """

        files = os.listdir(data_folder)
        run_files = [f for f in files if f.endswith(".mat") and "run_" in f and not f.startswith("#")]

        if not run_files:
            raise WrongFileNumber("No run files found in folder.")

        cls.LOG.info(f"Reading {len(run_files)} run_* files.")

        data = None

        for mfile in run_files:
            cls.LOG.debug(f"Reading file {mfile}")

            try:
                raw_data = loadmat(os.path.join(data_folder, mfile))
            except NotImplementedError:
                raw_data = mat73.loadmat(os.path.join(data_folder, mfile))

            d = cls._qdmio_stack_data(raw_data)
            data = d if data is None else np.stack((data, d), axis=0)

        # fix dimensions if only one run file was found -> only one polarity
        if len(run_files) == 1:
            data = data[np.newaxis, :, :, :]

        img_shape = np.array(
            [
                np.squeeze(raw_data["imgNumRows"]),
                np.squeeze(raw_data["imgNumCols"]),
            ],
            dtype=int,
        )

        n_freqs = int(np.squeeze(raw_data["numFreqs"]))
        frequencies = np.squeeze(raw_data["freqList"]).astype(np.float32)

        if n_freqs != len(frequencies):
            frequencies = np.array([frequencies[:n_freqs], frequencies[n_freqs:]])
        return cls(data=data, data_shape=img_shape, frequencies=frequencies, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def _qdmio_stack_data(cls, mat_dict: dict) -> NDArray:
        """Stack the data in the ODMR object.

        Args:
          mat_dict:

        Returns:

        """
        n_img_stacks = len([k for k in mat_dict.keys() if "imgStack" in k])
        img_stack1, img_stack2 = [], []

        if n_img_stacks == 2:
            # IF ONLY 2 IMG-STACKS, THEN WE ARE IN LOW freq. MODE (50 freq.)
            # imgStack1: [n_freqs, n_pixels] -> transpose to [n_pixels, n_freqs]
            cls.LOG.debug("Two ImgStacks found: Stacking data from imgStack1 and imgStack2.")
            img_stack1 = mat_dict["imgStack1"].T
            img_stack2 = mat_dict["imgStack2"].T
        elif n_img_stacks == 4:
            # 4 IMGSTACKS, THEN WE ARE IN HIGH freq. MODE (101 freqs)
            cls.LOG.debug("Four ImgStacks found: Stacking data from imgStack1, imgStack2 and imgStack3, imgStack4.")
            img_stack1 = np.concatenate([mat_dict["imgStack1"].T, mat_dict["imgStack2"].T])
            img_stack2 = np.concatenate([mat_dict["imgStack3"].T, mat_dict["imgStack4"].T])
        return np.stack((img_stack1, img_stack2), axis=0)

    # ------------------------------- BINNING -------------------------------- #
    def bin_data(self, bin_factor, norm=False, gf_correct=False, mask=True):
        """ creates a new odmr instance with different binning

        Returns: ODMR
        """
        self.LOG.info(f"Creating new ODMR object with binning {bin_factor}")

        if self._binning != 1:
            self.LOG.debug("data is already binned")

            if bin_factor == self._binning:
                self.LOG.warning("binning unchanged returning self")
                return self
            else:
                self.LOG.warning("binning changed, creating new object")
                bin_factor = bin_factor / self._binning

        data = self.get_odmr(norm=norm, gf_correct=gf_correct, mask=mask)
        data = data.reshape(self.data_shape)
        _odmr_binned = block_reduce(
            data,
            block_size=(1, 1, int(bin_factor), int(bin_factor), 1),
            func=np.nanmean,
            cval=np.median(data),
        )  # bins the data

        cls = ODMR2()

class ODMR:
    """ """

    def __getitem__(self, item: Union[Sequence[Union[str]], str]) -> NDArray:
        """
        Return the data of a given polarization, frequency range, pixel or frequency.

        Args:
            item: desired return value   (Default value = None)
                  currently available:
                      '+' - positive polarization
                      '-' - negative polarization
                      '<' - lower frequency range
                      '>' - higher frequency range
                      'r' - reshape to 2D image (data_shape)

        Returns: data of the desired return value

        Examples:
        >>> odmr['+'] -> pos. polarization
        >>> odmr['+', '<'] -> pos. polarization + low frequency range

        """

        if isinstance(item, int) or all(isinstance(i, int) for i in item):
            raise NotImplementedError("Indexing with only integers is not implemented yet.")

        idx, linear_idx = None, None

        self.LOG.debug(f"get item: {item}")

        items = ",".join(item)

        reshape = bool(re.findall("|".join(["r", "R"]), items))
        # get the data
        d = self.data
        if linear_idx is not None:
            d = d[:, :, linear_idx]
        elif reshape:
            self.LOG.debug("ODMR: reshaping data")
            d = d.reshape(self.n_pol, self.n_frange, self.data_shape[0], self.data_shape[1], self.n_freqs)

        # catch case where only indices are provided
        if len(item) == 0:
            return d

        # return the data
        if re.findall("|".join(["data", "d"]), items):
            return d

        # polarities
        if re.findall("|".join(["pos", re.escape("+")]), items):
            self.LOG.debug("ODMR: selected positive field polarity")
            pidx = [0]
        elif re.findall("|".join(["neg", "-"]), items):
            self.LOG.debug("ODMR: selected negative field polarity")
            pidx = [1]
        else:
            pidx = [0, 1]

        d = d[pidx]
        # franges
        if re.findall("|".join(["low", "l", "<"]), items):
            self.LOG.debug("ODMR: selected low frequency range")
            fidx = [0]
        elif re.findall("|".join(["high", "h", ">"]), items):
            self.LOG.debug("ODMR: selected high frequency range")
            fidx = [1]
        else:
            fidx = [0, 1]

        d = d[:, fidx]
        return np.squeeze(d)


    def bin_data(self, bin_factor: int, **kwargs: Any) -> None:
        """Bin the data.

        Args:
          bin_factor:
          **kwargs:

        Returns:

        """
        self._edit_stack.append(self._bin_data)
        self._apply_edit_stack(bin_factor=bin_factor)

    def _bin_data(self, bin_factor: Optional[Union[float, None]] = None, **kwargs: Any) -> None:
        """Bin the data from the raw data.

        Args:
          bin_factor:  (Default value = None)
          **kwargs:

        Returns:

        """
        if bin_factor is not None and self._pre_bin_factor:
            bin_factor /= self._pre_bin_factor
            self.LOG.debug(f"Pre binned data with factor {self._pre_bin_factor} reducing bin factor to {bin_factor}")

        if bin_factor is None:
            bin_factor = self._bin_factor

        self.LOG.debug(
            f"Binning data {self.data_shape} with factor {bin_factor} (pre bin factor: {self._pre_bin_factor})"
        )

        # reshape into image size
        reshape_data = self.data.reshape(
            self.n_pol,
            self.n_frange,
            *self.data_shape,
            self.n_freqs,
        )  # reshapes the data to the scan dimensions
        _odmr_binned = block_reduce(
            reshape_data,
            block_size=(1, 1, int(bin_factor), int(bin_factor), 1),
            func=np.nanmean,
            cval=np.median(reshape_data),
        )  # bins the data
        self._data_edited = _odmr_binned.reshape(
            self.n_pol, self.n_frange, -1, self.n_freqs
        )  # reshapes the data back to the ODMR data format

        self._bin_factor = int(bin_factor)  # sets the bin factor
        self.is_binned = True  # sets the binned flag

        self.LOG.info(
            f"Binned data from {reshape_data.shape[0]}x{reshape_data.shape[1]}x{reshape_data.shape[2]}x{reshape_data.shape[3]}x{reshape_data.shape[4]} "
            f"--> {_odmr_binned.shape[0]}x{_odmr_binned.shape[1]}x{_odmr_binned.shape[2]}x{_odmr_binned.shape[3]}x{_odmr_binned.shape[4]}"
        )

    def remove_overexposed(self, **kwargs: Any) -> None:
        """Remove overexposed pixels from the data.

        Args:
          **kwargs:

        Returns:

        """
        if self._data_edited is None:
            return self.LOG.warning("No data to remove overexposed pixels from.")

        self._overexposed = np.sum(self._data_edited, axis=-1) == self._data_edited.shape[-1]

        if np.sum(self._overexposed) > 0:
            self.LOG.warning(f"ODMR: {np.sum(self._overexposed)} pixels are overexposed")
            self._data_edited = ma.masked_where(self._data_edited == 1, self._data_edited)


def main() -> None:
    odmr = ODMR.from_qdmio(QDMpy.test_data_location())
    print(odmr[">"].shape)
    print(odmr["+"].shape)
    print(odmr["+>"].shape)


if __name__ == "__main__":
    main()
