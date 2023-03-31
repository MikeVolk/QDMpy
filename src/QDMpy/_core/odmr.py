import itertools
import logging
import os
import re
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from numpy import ma as ma
from numpy.typing import ArrayLike, NDArray
from skimage.measure import block_reduce

import QDMpy
import QDMpy.utils
from QDMpy.exceptions import WrongFileNumber

class ODMR:
    """Class for ODMR data object"""

    LOG = logging.getLogger(__name__)

    def __init__(
        self,
        data_array: ndarray,
        scan_dimensions: ndarray,
        frequencies_array: ndarray,
        **kwargs: Any,
    ) -> None:
        self.LOG.info("ODMR data object initialized")
        self.LOG.info("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.info(f"read parameter shape: data_array: {data_array.shape}")
        self.LOG.info(f"                      scan_dimensions: {scan_dimensions}")
        self.LOG.info(f"                      frequencies_array: {frequencies_array.shape}")
        self.LOG.info(f"                      n(freqs): {data_array.shape[-1]}")

        self.data_array = data_array

        self.n_pol = data_array.shape[0]
        self.n_frange = data_array.shape[1]

        self.frequencies_array = frequencies_array
        self.frequencies_cropped_array = None

        self.outlier_mask = None
        self.img_shape_array = np.array(scan_dimensions)

        self.norm_method = QDMpy.SETTINGS["odmr"]["norm_method"]

        self.edit_stack = [
            self.reset_data,
            self.normalize_data,
            None,  # placeholder for binning
            None,  # placeholder for outlier removal
            None,  # placeholder for global correction
        ]

        self.apply_edit_stack()

        self.imported_files_list: List[str] = kwargs.pop("imported_files", [])
        self.bin_factor = 1
        self.pre_bin_factor = 1  # in case pre binned data is loaded

        self.gf_factor = 0.0

        self.is_binned = False
        self.is_gf_corrected = False  # global fluorescence correction
        self.is_normalized = False
        self.is_cropped = False
        self.is_fcropped = False

    def __repr__(self) -> str:
        return (
            f"ODMR(data_array={self.data_array.shape}, "
            f"scan_dimensions={self.img_shape_array}, n_pol={self.n_pol}, "
            f"n_frange={self.n_frange}, n_pixel={self.n_pixel}, n_freqs={self.n_freqs}"
        )

    @property
    def n_freqs(self) -> int:
        return self.data_array.shape[-1]

    @property
    def n_pixel(self) -> int:
        return self.img_shape_array[0] * self.img_shape_array[1]

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
                      'f{freq}' - data at a specific frequency {freq}

        Returns: data of the desired return value

        Examples:
        ``` py
        >>> odmr['+'] #-> pos. polarization
        >>> odmr['+', '<'] #-> pos. polarization + low frequency range
        >>> odmr['f1.0'] #-> data at frequency 1.0 GHz
        ```

        """

        if isinstance(item, int) or all(isinstance(i, int) for i in item):
            raise NotImplementedError(
                "Indexing with only integers is not implemented yet."
            )

        idx, linear_idx = None, None

        self.LOG.debug(f"get item: {item}")

        items = ",".join(item)

        reshape = bool(re.findall("|".join(["r", "R"]), items))

        # get the data
        d = self.data_array

        if linear_idx is not None:
            d = d[:, :, linear_idx]
        elif reshape:
            self.LOG.debug("ODMR: reshaping data")
            d = d.reshape(
                self.n_pol,
                self.n_frange,
                self.img_shape_array[0],
                self.img_shape_array[1],
                self.n_freqs,
            )

        # catch case where only indices are provided
        if len(item) == 0:
            return d

        # return the data
        if "data" in items or "d" in items:
            return d

        # polarities
        pidx = []
        if "+" in items or "pos" in items:
            self.LOG.debug("ODMR: selected positive field polarity")
            pidx.append(0)
        if "-" in items or "neg" in items:
            self.LOG.debug("ODMR: selected negative field polarity")
            pidx.append(1)
        if not pidx:
            pidx = [0, 1]
        d = d[pidx]

        # franges
        fidx = []
        if "<" in items or "low" in items or "l" in items:
            self.LOG.debug("ODMR: selected low frequency range")
            fidx.append(0)
        if ">" in items or "high" in items or "h" in items:
            self.LOG.debug("ODMR: selected high frequency range")
            fidx.append(1)
        if not fidx:
            fidx = [0, 1]
        d = d[:, fidx]

        # frequency
        freq_idx = []
        for i in item:
            if isinstance(i, str) and i.startswith("f"):
                try:
                    freq = float(i[1:])
                except ValueError:
                    self.LOG.warning(f"Invalid frequency index {i}")
                    continue
                freq_idx.append(np.argmin(np.abs(self.frequencies_array - freq)))
        if freq_idx:
            d = d[:, :, :, :, freq_idx]

        return np.squeeze(d)

    # index related
    def get_binned_pixel_indices(
        self, x: int, y: int
    ) -> Tuple[Sequence[int], Sequence[int]]:
        """determines the indices of the pixels that are binned into the pixel at (x,y)

        Args:
          x: x index
          y: y index

        Returns:

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

    THRESHOLD = 0.001

    def most_divergent_from_mean(self) -> Tuple[int, int]:
        """
        Get the most divergent pixel from the mean in data coordinates.

        Returns:
            A tuple (i, j) representing the coordinates of the most divergent pixel.
        """
        delta = self.delta_mean.copy()
        delta[delta > self.THRESHOLD] = np.nan
        return np.unravel_index(np.nanargmax(delta), delta.shape)

    # from methods
    # FROM / IMPORT
    @classmethod
    def from_qdmio(cls, data_folder: Union[str, os.PathLike], **kwargs: Any) -> "ODMR":
        """Loads QDM data from a Matlab file.

        Args:
          data_folder:

        Returns:
            ODMR instance

        Raises:
            WrongFileTypeError: if no .mat files are found

        Notes:
            Filenames starting with '#' are ignored.
        """
        with os.scandir(data_folder) as entries:
            run_files = [
                entry.name
                for entry in entries
                if entry.name.endswith(".mat")
                   and "run_" in entry.name
                   and not entry.name.startswith("#")
            ]

        if not run_files:
            raise WrongFileNumber("No run files found in folder.")

        cls.LOG.info(f"Reading {len(run_files)} run_* files.")

        # initialize the data
        data = None

        # loop over all run files
        for mfile in run_files:
            cls.LOG.debug(f"Reading file {mfile}")

            raw_data = QDMpy.utils.loadmat(os.path.join(data_folder, mfile))

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

        return cls(data=data, scan_dimensions=img_shape, frequencies=frequencies, **kwargs)

    @classmethod
    def _qdmio_stack_data(cls, mat_dict: dict) -> NDArray:
        """Stack the data in the ODMR object.

        Args:
            mat_dict: dictionary containing the data from the QDMio matlab file

        Returns:
            A numpy array containing the data for both frequency ranges.

        Raises:
            ValueError: If the number of image stacks is not 2 or 4.

        Notes:
            The number of image stacks determines whether the data is in low-frequency mode (50 frequencies) or
            high-frequency mode (101 frequencies).

        """
        n_img_stacks = len([k for k in mat_dict if "imgStack" in k])
        img_stack1, img_stack2 = [], []

        if n_img_stacks == 2:
            # IF ONLY 2 IMG-STACKS, THEN WE ARE IN LOW freq. MODE (50 freq.)
            # imgStack1: [n_freqs, n_pixels] -> transpose to [n_pixels, n_freqs]
            cls.LOG.debug(
                "Two ImgStacks found: Stacking data from imgStack1 and imgStack2."
            )
            img_stack1 = mat_dict["imgStack1"].T
            img_stack2 = mat_dict["imgStack2"].T
        elif n_img_stacks == 4:
            # 4 IMGSTACKS, THEN WE ARE IN HIGH freq. MODE (101 freqs)
            cls.LOG.debug(
                "Four ImgStacks found: Stacking data from imgStack1, imgStack2, imgStack3, imgStack4."
            )
            img_stack1 = np.concatenate([mat_dict["imgStack1"], mat_dict["imgStack2"]]).T
            img_stack2 = np.concatenate([mat_dict["imgStack3"], mat_dict["imgStack4"]]).T
        else:
            raise ValueError(f"Expected 2 or 4 image stacks, got {n_img_stacks}.")

        return np.stack((img_stack1, img_stack2), axis=0)

    @classmethod
    def get_norm_factors(cls, data: ArrayLike, method: str = "max") -> np.ndarray:
        """
        Return the normalization factors for the data.

        Args:
            data: The data to normalize.
            method: The normalization method to use. Supported methods are "max" (default),
                "mean", "std", "mad", and "l2".

        Returns:
            A 1D array of normalization factors.

        Raises:
            NotImplementedError: if method is not implemented.
        """

        if method == "max":
            factors = np.max(data, axis=-1, keepdims=True)
            cls.LOG.debug(
                f"Determining normalization factor from maximum value of each pixel spectrum. "
                f"Shape of factors: {factors.shape}"
            )
        elif method == "mean":
            factors = np.mean(data, axis=-1, keepdims=True)
            cls.LOG.debug(
                f"Determining normalization factor from mean value of each pixel spectrum. "
                f"Shape of factors: {factors.shape}"
            )
        elif method == "std":
            factors = np.std(data, axis=-1, keepdims=True)
            cls.LOG.debug(
                f"Determining normalization factor from standard deviation of each pixel spectrum. "
                f"Shape of factors: {factors.shape}"
            )
        elif method == "mad":
            med = np.median(data, axis=-1, keepdims=True)
            factors = np.median(np.abs(data - med), axis=-1, keepdims=True)
            cls.LOG.debug(
                f"Determining normalization factor from median absolute deviation of each pixel spectrum. "
                f"Shape of factors: {factors.shape}"
            )
        elif method == "l2":
            factors = np.linalg.norm(data, axis=-1, keepdims=True)
            cls.LOG.debug(
                f"Determining normalization factor from L2-norm of each pixel spectrum. "
                f"Shape of factors: {factors.shape}"
            )
        else:
            raise NotImplementedError(f'Method "{method}" not implemented.')

        return factors


    # properties
    @property
    def data_shape(self) -> NDArray:
        """ """
        return (self.img_shape / self._bin_factor).astype(np.uint32)

    @property
    def img_shape(self) -> NDArray:
        """ """
        return self._img_shape

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
    def global_factor(self) -> float:
        """ """
        return self._gf_factor

    @property
    def data(self) -> NDArray:
        """ """
        if self._data_edited is None:
            return np.ascontiguousarray(self._raw_data)
        else:
            return np.ascontiguousarray(self._data_edited)

    @property
    def delta_mean(self) -> NDArray:
        """ """
        return np.sum(
            np.square(self.data - self.mean_odmr[:, :, np.newaxis, :]), axis=-1
        )

    @property
    def mean_odmr(self) -> NDArray:
        """Calculate the mean of the data."""
        return self.data.mean(axis=-2)

    @property
    def raw_contrast(self) -> NDArray:
        """Calculate the minimum of MW sweep for each pixel."""
        return np.min(self.data, -2)

    @property
    def mean_contrast(self) -> NDArray:
        """Calculate the mean of the minimum of MW sweep for each pixel."""
        return np.mean(self.raw_contrast)

    @property
    def _mean_baseline(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Calculate the mean baseline of the data."""
        baseline_left_mean = np.mean(self.mean_odmr[:, :, :5], axis=-1)
        baseline_right_mean = np.mean(self.mean_odmr[:, :, -5:], axis=-1)
        baseline_mean = np.mean(
            np.stack([baseline_left_mean, baseline_right_mean], -1), axis=-1
        )
        return baseline_left_mean, baseline_right_mean, baseline_mean

    @property
    def bin_factor(self) -> int:
        """ """
        return self._bin_factor * self._pre_bin_factor

    # edit methods

    def _apply_edit_stack(
        self, **kwargs: Any
    ) -> None:  # todo add index of calling method?
        """Apply the edit stack.

        Args:
          **kwargs:

        Returns:

        """
        self.LOG.debug("Applying edit stack")
        for edit_func in self._edit_stack:
            if edit_func is not None:
                edit_func(**kwargs)  # type: ignore[operator]

    def reset_data(self, **kwargs: Any) -> None:
        """Reset the data.

        Args:
          **kwargs:

        Returns:

        """
        self.LOG.debug("Resetting data to raw data.")
        self._data_edited = deepcopy(self._raw_data)
        self._norm_factors = None
        self.is_normalized = False
        self.is_binned = False
        self.is_gf_corrected = False
        self.is_cropped = False
        self.is_fcropped = False

    def normalize_data(self, method: Union[str, None] = None, **kwargs: Any) -> None:
        """Normalize the data.

        Args:
          method:  (Default value = None)
          **kwargs:

        Returns:

        """
        if method is None:
            method = self._norm_method
        self._edit_stack[1] = self._normalize_data
        self._apply_edit_stack(method=method)

    def _normalize_data(self, method: str = "max", **kwargs: Any) -> None:
        """Normalize the data.

        Args:
          method:  (Default value = "max")
          **kwargs:
        """
        self._norm_factors = self.get_norm_factors(self.data, method=method)  # type: ignore[assignment]
        self.LOG.debug(f"Normalizing data with method: {method}")
        self._norm_method = method
        self.is_normalized = True
        self._data_edited /= self._norm_factors  # type: ignore[arg-type]

    def apply_outlier_mask(
        self, outlier_mask: Union[NDArray, None] = None, **kwargs: Any
    ) -> None:
        """Apply the outlier mask to the data.

        Args:
          outlier_mask: np.ndarray:  (Default value = None)
          **kwargs:

        Returns:

        """

        self._edit_stack[3] = self._apply_outlier_mask

        if outlier_mask is None:
            outlier_mask = self.outlier_mask

        self.outlier_mask = outlier_mask
        self._apply_edit_stack()

    def _apply_outlier_mask(self, **kwargs: Any) -> None:
        """Apply the outlier mask.

        Args:
          **kwargs:

        Returns:

        """
        if self.outlier_mask is None:
            self.LOG.debug("No outlier mask applied.")
            return
        self.LOG.debug("Applying outlier mask")
        self._data_edited[:, :, self.outlier_mask.reshape(-1), :] = np.nan
        self.LOG.debug(
            f"Outlier mask applied, set {np.isnan(self._data_edited[0,0,:,0]).sum()} to NaN."
        )

    def bin_data(self, bin_factor: int, **kwargs: Any) -> None:
        """Bin the data.

        Args:
          bin_factor:
          **kwargs:

        Returns:

        """
        self._edit_stack[2] = self._bin_data
        self._apply_edit_stack(bin_factor=bin_factor)

    def _bin_data(
        self, bin_factor: Optional[Union[float, None]] = None, **kwargs: Any
    ) -> None:
        """Bin the data from the raw data.

        Args:
          bin_factor:  (Default value = None)
          **kwargs:

        Returns:

        """
        if bin_factor != 1 and self._pre_bin_factor != 1:
            bin_factor /= self._pre_bin_factor
            self.LOG.debug(
                f"Pre binned data with factor {self._pre_bin_factor} reducing bin factor to {bin_factor}"
            )

        if bin_factor is None:
            bin_factor = self._bin_factor

        self.LOG.debug(
            f"Binning data {self.img_shape} with factor {bin_factor} (pre bin factor: {self._pre_bin_factor})"
        )

        # reshape into image size
        reshape_data = self.data.reshape(
            self.n_pol,
            self.n_frange,
            int(self.img_shape[0] / self._pre_bin_factor),
            int(self.img_shape[1] / self._pre_bin_factor),
            self.n_freqs,
        )  # reshapes the data to the scan dimensions
        _odmr_binned = block_reduce(
            reshape_data,
            block_size=(1, 1, int(bin_factor), int(bin_factor), 1),
            func=np.nanmean,  # uses mean
            cval=np.median(reshape_data),  # replaces with median
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

        self._overexposed = (
            np.sum(self._data_edited, axis=-1) == self._data_edited.shape[-1]
        )

        if np.sum(self._overexposed) > 0:
            self.LOG.warning(
                f"ODMR: {np.sum(self._overexposed)} pixels are overexposed"
            )
            self._data_edited = ma.masked_where(
                self._data_edited == 1, self._data_edited
            )

    ### CORRECTION METHODS ###
    def calc_gf_correction(self, gf: float) -> NDArray:
        """Calculate the global fluorescence correction.

        Args:
          gf: The global fluorescence factor

        Returns: The global fluorescence correction
        """
        baseline_left_mean, baseline_right_mean, baseline_mean = self._mean_baseline
        return gf * (self.mean_odmr - baseline_mean[:, :, np.newaxis])

    def correct_glob_fluorescence(self, gf_factor: float, **kwargs: Any) -> None:
        """Correct the data for the gradient factor.

        Args:
          gf_factor:
          **kwargs:

        Returns:

        """
        self._edit_stack[3] = self._correct_glob_fluorescence
        self._gf_factor = gf_factor
        self._apply_edit_stack(glob_fluorescence=gf_factor, **kwargs)

    def _correct_glob_fluorescence(
        self, gf_factor: Union[float, None] = None, **kwargs: Any
    ) -> None:
        """Correct the data for the global fluorescence.

        Args:
          gf_factor: global fluorescence factor (Default value = None)
          **kwargs: pass though for additional _apply_edit_stack kwargs
        """
        if gf_factor is None:
            gf_factor = self._gf_factor

        self.LOG.info(f"Correcting for global fluorescence with value {gf_factor}")
        correction = self.calc_gf_correction(gf=gf_factor)

        self._data_edited -= correction[:, :, np.newaxis, :]
        self.is_gf_corrected = True  # sets the gf corrected flag
        self._gf_factor = gf_factor  # sets the gf factor

    def reset_gf_correction(self):
        """Reset the global fluorescence correction.

        Returns:

        """

        if self.global_factor == 0:
            self.LOG.debug(f"Removing old global fluorescence with value {self.global_factor}")
            old_correction = self.calc_gf_correction(gf=self.global_factor)
            self._data_edited += old_correction[:, :, np.newaxis, :]

    # noinspection PyTypeChecker
    def check_glob_fluorescence(
        self, gf_factor: Union[float, None] = None, idx: Union[int, None] = None
    ) -> None:
        """

        Args:
          gf_factor:  (Default value = None)
          idx:  (Default value = None)

        Returns:

        """
        if idx is None:
            idx = self.most_divergent_from_mean()[-1]

        if gf_factor is None:
            gf_factor = self._gf_factor

        new_correct = self.calc_gf_correction(gf=gf_factor)

        f, ax = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(9, 5))
        for p in np.arange(self.n_pol):
            for f in np.arange(self.n_frange):
                d = self.data[p, f, idx].copy()

                old_correct = self.calc_gf_correction(gf=self._gf_factor)
                if self._gf_factor != 0:
                    ax[p, f].plot(
                        self.f_ghz[f], d, "k:", label=f"current: GF={self._gf_factor}"
                    )

                (l,) = ax[p, f].plot(
                    self.f_ghz[f],
                    d + old_correct[p, f],
                    ".--",
                    mfc="w",
                    label="original",
                )
                ax[p, f].plot(
                    self.f_ghz[f],
                    d + old_correct[p, f] - new_correct[p, f],
                    ".-",
                    label=f"corrected (GF={gf_factor})",
                    color=l.get_color(),
                )
                ax[p, f].plot(
                    self.f_ghz[f], 1 + new_correct[p, f], "r--", label="correction"
                )
                ax[p, f].set_title(f"{['+', '-'][p]},{['<', '>'][f]}")
                ax[p, f].legend()
                # , ylim=(0, 1.5))
                ax[p, f].set(ylabel="ODMR contrast", xlabel="Frequency [GHz]")
            plt.tight_layout()



def main() -> None:
    QDMpy.LOG.setLevel(logging.DEBUG)
    odmr = ODMR.from_qdmio(
        "/media/paleomagnetism/Fort Staff/QDM/2022_Hawaii_1907/20221207_14G"
    )
    # for i in [4, 6, 8, 16]:
    #     odmr.bin_data(i)


if __name__ == "__main__":
    main()
