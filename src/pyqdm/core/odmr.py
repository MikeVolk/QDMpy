import itertools
import logging
import os
import re
from functools import cached_property

import mat73
import numpy as np
from matplotlib import pyplot as plt
from numpy import ma as ma
from scipy.io import loadmat
from skimage.transform import downscale_local_mean

from pyqdm.exceptions import WrongFileNumber
from pyqdm.utils import idx2rc, rc2idx


class ODMR:
    LOG = logging.getLogger("pyQDM.ODMR")

    def __repr__(self):
        return f"ODMR(data={self.data.shape}, scan_dimensions={self.scan_dimensions}, n_pol={self.n_pol}, n_frange={self.n_frange}, n_pixel={self.n_pixel}, n_freqs={self.n_freqs}, frequencies={self.frequencies.shape})"

    def __init__(self, data, scan_dimensions, frequencies):
        self.LOG.info("ODMR data object initialized")
        self.LOG.debug("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.debug(f"read parameter shape: data: {data.shape}")
        self.LOG.debug(f"                      scan_dimensions: {scan_dimensions}")
        self.LOG.debug(f"                      frequencies: {frequencies.shape}")
        self.LOG.debug(f"                      n_freqs: {data.shape[-1]}")

        self._raw_data = data

        self.n_pol = data.shape[0]
        self.n_frange = data.shape[1]
        self._frequencies = frequencies
        self._frequencies_cropped = None

        self._scan_dimensions = np.array(scan_dimensions)

        self._data_edited = None
        self._norm_method = "max"  # todo add to config
        self._edit_stack = [self.reset_data, self._normalize_data, self.remove_overexposed, None, None, None]

        self._apply_edit_stack()

        self._bin_factor = 1
        self._pre_bin_factor = 1  # in case pre binned data is loaded

        self._gf_factor = 0

        self.is_binned = False
        self.is_gf_corrected = False  # global fluorescence correction
        self.is_normalized = False
        self.is_cropped = False
        self.is_fcropped = False

    def remove_overexposed(self, **kwargs):
        """
        Remove overexposed pixels from the data.
        """
        self._overexposed = np.sum(self._data_edited, axis=-1) == self._data_edited.shape[-1]

        if np.sum(self._overexposed) > 0:
            self.LOG.warning(f"ODMR: {np.sum(self._overexposed)} pixels are overexposed")
            self._data_edited = ma.masked_where(self._data_edited == 1, self._data_edited)

    def __getitem__(self, item):
        """
        Return the data of a given polarization, frequency range, pixel or frequency.
        Usage:
            odmr['+'] -> pos. polarization
            odmr['+', '<'] -> pos. polarization + low frequency range
            NOTE: pixels need to be specified as a list of indices.
            odmr['+', '<', [0]] -> pos. polarization + low frequency range + pixel 0
            odmr['+', '<', [[1],[2]] -> pos. polarization + low frequency range + pixel with x/y coordinates (2,1)

        :param item: str, int, list
        :return: numpy.ndarray
        """
        # todo implement slicing
        # if isinstance(item, slice):
        #     return self.data[item]

        # if all items are indices make a list of indices
        if all(isinstance(i, int) for i in item):
            item = [item]

        item = np.atleast_1d(item)

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
            d = d.reshape(self.n_pol, self.n_frange, *self.scan_dimensions, self.n_freqs)

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

    def get_binned_pixel_indices(self, x, y):
        """
        Return the indices of the binned pixels. Reference is the data index.
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

    def rc2idx(self, rc):
        return rc2idx(rc, self.scan_dimensions)

    def idx2rc(self, idx):
        return idx2rc(idx, self.scan_dimensions)

    ### from methods ###

    @classmethod
    def _stack_data(cls, mfile):
        """
        Stack the data in the ODMR object.
        """
        n_img_stacks = len([k for k in mfile.keys() if "imgStack" in k])
        img_stack1, img_stack2 = [], []

        if n_img_stacks == 2:
            # IF ONLY 2 IMGSTACKS, THEN WE ARE IN LOW freq. MODE (50 freq.)
            # imgStack1: [n_freqs, n_pixels] -> transpose to [n_pixels, n_freqs]
            cls.LOG.debug("Two ImgStacks found: Stacking data from imgStack1 and imgStack2.")
            img_stack1 = mfile["imgStack1"].T
            img_stack2 = mfile["imgStack2"].T
        elif n_img_stacks == 4:
            # 4 IMGSTACKS, THEN WE ARE IN HIGH freq. MODE (101 freqs)
            cls.LOG.debug("Four ImgStacks found: Stacking data from imgStack1, imgStack2 and imgStack3, imgStack4.")
            img_stack1 = np.concatenate([mfile["imgStack1"], mfile["imgStack2"]]).T
            img_stack2 = np.concatenate([mfile["imgStack3"], mfile["imgStack4"]]).T
        return np.stack((img_stack1, img_stack2), axis=0)

    @classmethod
    def from_qdmio(cls, data_folder):
        """
        Loads QDM data from a Matlab file.
        """
        files = os.listdir(data_folder)
        run_files = [f for f in files if f.endswith(".mat") and "run_" in f and not f.startswith("#")]

        if not run_files:
            raise WrongFileNumber("No run files found in folder.")

        cls.LOG.info(f"Reading {len(run_files)} run_* files.")

        try:
            raw_data = [loadmat(os.path.join(data_folder, mfile)) for mfile in run_files]
        except NotImplementedError:
            raw_data = [mat73.loadmat(os.path.join(data_folder, mfile)) for mfile in run_files]

        data = None
        for mfile in raw_data:
            d = cls._stack_data(mfile)
            data = d if data is None else np.stack((data, d), axis=0)
        if data.ndim == 3:
            data = data[np.newaxis, :, :, :]
        scan_dimensions = np.array(
            [np.squeeze(raw_data[0]["imgNumRows"]), np.squeeze(raw_data[0]["imgNumCols"])], dtype=int
        )

        n_freqs = int(np.squeeze(raw_data[0]["numFreqs"]))
        frequencies = np.squeeze(raw_data[0]["freqList"]).astype(np.float32)
        if n_freqs != len(frequencies):
            frequencies = np.array([frequencies[:n_freqs], frequencies[n_freqs:]])
        return cls(data=data, scan_dimensions=scan_dimensions, frequencies=frequencies)

    @classmethod
    def get_norm_factors(cls, data, method="max"):
        """
        Return the normalization factors for the data.

        :param data: data
        :param method:
        :return:
        """

        match method:
            case "max":
                mx = np.max(data, axis=-1)
                cls.LOG.debug(
                    f"Determining normalization factor from maximum value of each pixel spectrum. "
                    f"Shape of mx: {mx.shape}"
                )
                factors = np.expand_dims(mx, axis=-1)
            case _:
                raise NotImplementedError('Method "{}" not implemented.'.format(method))

        return factors

    @property
    def scan_dimensions(self):
        return (self._scan_dimensions / self.bin_factor).astype(np.uint32)

    @property
    def n_pixel(self):
        """
        Return the number of pixels.
        :return: int
        """
        return int(self.scan_dimensions[0] * self.scan_dimensions[1])

    @property
    def n_freqs(self):
        """
        Return the number of frequencies to be used in the fits.
        :return: int
        """
        return self.frequencies.shape[1]

    @property
    def frequencies(self):
        """
        Return the cropped or uncropped frequencies.
        :return: numpy.ndarray
        """
        if self._frequencies_cropped is None:
            return self._frequencies
        else:
            return self._frequencies_cropped

    @property
    def f_hz(self):
        """
        Returns the frequencies of the ODMR in Hz.
        """
        return self.frequencies

    @property
    def f_ghz(self):
        """
        Returns the frequencies of the ODMR in GHz.
        """
        return self.frequencies / 1e9

    @property
    def data(self):
        if self._data_edited is None:
            return np.ascontiguousarray(self._raw_data)
        else:
            return np.ascontiguousarray(self._data_edited)

    @cached_property
    def mean_odmr(self):
        """
        Calculate the mean of the data.
        """
        return self.data.mean(axis=-2)

    @cached_property
    def raw_contrast(self):
        """
        Calculate the minimum of MW sweep for each pixel.
        """
        return np.min(self.data, -2)

    @cached_property
    def mean_contrast(self):
        """
        Calculate the mean contrast of the data.
        """
        return np.mean(self.raw_contrast)

    @property
    def bin_factor(self):
        return self._bin_factor * self._pre_bin_factor

    def _apply_edit_stack(self, **kwargs):
        """
        Apply the edit stack.
        """
        self.LOG.debug("Applying edit stack")
        for edit_func in self._edit_stack:
            if edit_func is not None:
                edit_func(**kwargs)

    def reset_data(self, **kwargs):
        """
        Reset the data.
        """
        self.LOG.debug("Resetting data to raw data.")
        self._data_edited = self._raw_data.copy()
        self._norm_factors = None
        self.is_normalized = False
        self.is_binned = False
        self.is_gf_corrected = False
        self.is_cropped = False
        self.is_fcropped = False

    def normalize_data(self, method=None, **kwargs):
        """
        Normalize the data.
        """
        if method is None:
            method = self._norm_method
        self._edit_stack[1] = self._normalize_data
        self._apply_edit_stack(method=method)

    def _normalize_data(self, method="max", **kwargs):
        """
        Normalize the data.
        """
        self._norm_factors = self.get_norm_factors(self.data, method=method)
        self.LOG.debug(f"Normalizing data with method: {method}")
        self._norm_method = method
        self.is_normalized = True
        self._data_edited /= self._norm_factors

    def bin_data(self, bin_factor, **kwargs):
        """
        Bin the data.
        """
        self._edit_stack[2] = self._bin_data
        self._apply_edit_stack(bin_factor=bin_factor)

    def _bin_data(self, bin_factor=None, **kwargs):
        """
        Bin the data from the raw data.
        """
        if bin_factor is not None and self._pre_bin_factor:
            bin_factor /= self._pre_bin_factor

        if bin_factor is None:
            bin_factor = self._bin_factor

        # reshape into image size
        reshape_data = self.data.reshape(
            self.n_pol,
            self.n_frange,
            *(self._scan_dimensions / self._pre_bin_factor).astype(int),
            self.n_freqs,
        )  # reshapes the data to the scan dimensions
        _odmr_binned = downscale_local_mean(
            reshape_data,
            factors=(1, 1, int(bin_factor), int(bin_factor), 1),
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

    @property
    def delta_mean(self):
        return np.sum(np.square(self.data - self.mean_odmr[:, :, np.newaxis, :]), axis=-1)

    def get_most_divergent_from_mean(self):
        """
        Get the most divergent pixel from the mean in data coordinates.
        """
        delta = self.delta_mean.copy()
        delta[delta > 0.001] = np.nan
        return np.unravel_index(np.argmax(delta, axis=None), self.delta_mean.shape)

    ### CORRECTION METHODS ###
    @property
    def global_factor(self):
        return self._gf_factor

    def correct_glob_fluorescence(self, gf_factor, **kwargs):
        """
        Correct the data for the gradient factor.
        """
        self._edit_stack[3] = self._correct_glob_fluorescence
        self._gf_factor = gf_factor
        self._apply_edit_stack(glob_fluorescence=gf_factor, **kwargs)

    def _correct_glob_fluorescence(self, gf_factor=None, **kwargs):
        if gf_factor is None:
            gf_factor = self._gf_factor

        self.LOG.debug(f"Correcting for global fluorescence with value {gf_factor}")
        correction = self._get_gf_correction(gf=gf_factor)

        self._data_edited -= correction[:, :, np.newaxis, :]
        self.is_gf_corrected = True  # sets the gf corrected flag
        self._gf_factor = gf_factor  # sets the gf factor

    def _get_gf_correction(self, gf):
        baseline_left_mean, baseline_right_mean, baseline_mean = self._mean_baseline
        return gf * (self.mean_odmr - baseline_mean[:, :, np.newaxis])

    # noinspection PyTypeChecker
    def check_glob_fluorescence(self, gf_factor=None, idx=None):
        if idx is None:
            idx = self.get_most_divergent_from_mean()[-1]

        if gf_factor is None:
            gf_factor = self._gf_factor

        new_correct = self._get_gf_correction(gf=gf_factor)

        f, ax = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(15, 10))
        for p in np.arange(self.n_pol):
            for f in np.arange(self.n_frange):
                d = self.data[p, f, idx].copy()

                old_correct = self._get_gf_correction(gf=self._gf_factor)
                if self._gf_factor != 0:
                    ax[p, f].plot(self.f_ghz[f], d, "k:", label=f"current: GF={self._gf_factor}")

                (l,) = ax[p, f].plot(self.f_ghz[f], d + old_correct[p, f], ".--", mfc="w", label="original")
                ax[p, f].plot(
                    self.f_ghz[f],
                    d + old_correct[p, f] - new_correct[p, f],
                    ".-",
                    label="corrected",
                    color=l.get_color(),
                )
                ax[p, f].plot(self.f_ghz[f], 1 + new_correct[p, f], "r--", label="correction")
                ax[p, f].set_title(f"{['+', '-'][p]},{['<', '>'][f]}")
                ax[p, f].legend()
                # , ylim=(0, 1.5))
                ax[p, f].set(ylabel="ODMR contrast", xlabel="Frequency [GHz]")

    @cached_property
    def _mean_baseline(self):
        baseline_left_mean = np.mean(self.mean_odmr[:, :, :5], axis=-1)
        baseline_right_mean = np.mean(self.mean_odmr[:, :, -5:], axis=-1)
        baseline_mean = np.mean(np.stack([baseline_left_mean, baseline_right_mean], -1), axis=-1)
        return baseline_left_mean, baseline_right_mean, baseline_mean
