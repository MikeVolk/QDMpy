import itertools
import logging
import os
import re
from copy import deepcopy
from typing import Any, Optional, Sequence, Tuple, Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import ma as ma
from numpy.typing import ArrayLike
from skimage.measure import block_reduce

import QDMpy
import QDMpy.utils as utils
from QDMpy.exceptions import WrongFileNumber


class ODMR:
    """
    Class for ODMR (Optically Detected Magnetic Resonance) data object. This
    class handles the data processing and organization for ODMR data, providing
    methods for normalization, binning, outlier removal, and global fluorescence
    correction. The class also manages the data's metadata and internal state.

    Attributes:
        LOG (logging.Logger): Logger for the class.

    Args:
        data_array (np.ndarray): Input data array for ODMR data. scan_dimensions
        (np.ndarray): Scan dimensions of the data. frequency_array (np.ndarray):
        Frequency array of the data. **kwargs (Any): Additional keyword
        arguments.

    Example:
        >>> import numpy as np
        >>> from ODMR import ODMR
        >>> data_array = np.random.rand(2, 2, 100, 100)
        >>> scan_dimensions = np.array([50, 50])
        >>> frequency_array = np.linspace(1, 5, 100)
        >>> odmr_data = ODMR(data_array, scan_dimensions, frequency_array)
        >>> print(odmr_data)
        ODMR(data_array=(2, 2, 100, 100), scan_dimensions=[50 50], n_pol=2, n_frange=2, n_pixel=100, n_freqs=100)

    """

    LOG = logging.getLogger(__name__)

    def __init__(
        self,
        data_array: np.ndarray,
        scan_dimensions: np.ndarray,
        frequency_array: np.ndarray,
        **kwargs: Any,
    ) -> None:
        self.LOG.info("ODMR data object initialized")
        self.LOG.info("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.info(f"read parameter shape: data_array: {data_array.shape}")
        self.LOG.info(f"                      scan_dimensions: {scan_dimensions}")
        self.LOG.info(f"                      frequencies_array: {frequency_array.shape}")
        self.LOG.info(f"                      n(freqs): {data_array.shape[-1]}")

        self._raw_data = data_array

        self.n_pol = data_array.shape[0]
        self.n_frange = data_array.shape[1]

        self._frequencies = frequency_array
        self._frequencies_cropped = None

        self.outlier_mask = None
        self._img_shape = np.array(scan_dimensions)
        self._data_xy_ratio = scan_dimensions[0] / scan_dimensions[1]

        self._norm_method = QDMpy.SETTINGS["odmr"]["norm_method"]

        self._edit_stack = [
            self.reset_data,
            self.normalize_data,
            None,  # placeholder for binning
            None,  # placeholder for outlier removal
            None,  # placeholder for global correction
        ]

        self._apply_edit_stack()

        self.imported_files_list: List[str] = kwargs.pop("imported_files", [])
        self._bin_factor = 1
        self._pre_bin_factor = 1  # in case pre binned data is loaded

        self._gf_factor = 0.0

        self.is_binned = False
        self.is_gf_corrected = False  # global fluorescence correction
        self.is_normalized = False
        self.is_cropped = False
        self.is_fcropped = False

    def __repr__(self) -> str:
        """
        Returns the string representation of the ODMR data object.

        Returns:
            str: String representation of the ODMR data object.
        """
        return (
            f"ODMR(data_array={self._raw_data.shape}, "
            f"scan_dimensions={self._img_shape}, n_pol={self.n_pol}, "
            f"n_frange={self.n_frange}, n_pixel={self.n_pixel}, n_freqs={self.n_freqs}"
        )

    @property
    def n_freqs(self) -> int:
        """
        Returns the number of frequencies in the ODMR data.

        Returns:
            int: Number of frequencies.
        """
        return self._raw_data.shape[-1]

    @property
    def n_pixel(self) -> int:
        """
        Returns the number of pixels in the ODMR data.

        Returns:
            int: Number of pixels.
        """
        return self.data.shape[2]

    def __getitem__(self, item: Union[Sequence[Union[str, Any]], str]) -> np.ndarray:
        """
        Return the data of a given polarization, frequency range, pixel, or
        frequency.

        Args:
            item: desired return value (Default value = None)
                currently available:
                    '+' - positive polarization
                    '-' - negative polarization
                    '<' - lower frequency range
                    '>' - higher frequency range
                    'r' - reshape to 2D image (data_shape)
                    'f{freq}' - data at a specific frequency {freq}
                additionally, supports NumPy slicing and integer indexing

        Returns:
            np.ndarray: Data of the desired return value.

        Examples:
            >>> odmr = ODMR(...)
            >>> odmr['+']         # positive polarization
            >>> odmr['+', '<']     # positive polarization + low frequency range
            >>> odmr['f1.0']       # data at frequency 1.0 GHz
            >>> odmr[:, :, 0:10]   # data with the first 10 frequencies
        """

        if isinstance(item, int) or all(isinstance(i, int) for i in item):
            raise NotImplementedError("Indexing with only integers is not implemented yet.")

        idx, linear_idx = None, None

        self.LOG.debug(f"get item: {item}")

        items = ",".join(item)

        reshape = bool(re.findall("|".join(["r", "R"]), items))

        # get the data
        d = self._raw_data

        if linear_idx is not None:
            d = d[:, :, linear_idx]
        elif reshape:
            self.LOG.debug("ODMR: reshaping data")
            d = d.reshape(
                self.n_pol,
                self.n_frange,
                self._img_shape[0],
                self._img_shape[1],
                self.n_freqs,
            )

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
            if isinstance(i, float):
                freq_idx.append(np.argmin(np.abs(self.frequencies - freq)))
        if freq_idx:
            d = d[..., freq_idx]

        return np.squeeze(d)

    def get_binned_pixel_indices(self, x: int, y: int) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Determines the indices of the pixels that are binned into the pixel at
        (x, y).

        Args:
            x (int): x index of the binned pixel. y (int): y index of the binned
            pixel.

        Returns:
            Tuple[Sequence[int], Sequence[int]]: A tuple of two lists containing
            the indices of
                                                the original pixels that are
                                                binned into the pixel at (x, y).

        Example:
            >>> odmr = ODMR(...)
            >>> x, y = 2, 3
            >>> xid, yid = odmr.get_binned_pixel_indices(x, y)
            >>> print(xid, yid)
            [6, 7, 8] [9, 10, 11]
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

    def rc2idx(self, rc: ArrayLike) -> np.ndarray:
        """
        Converts row-column indices to linear indices for the given data shape.

        Args:
            rc (ArrayLike): A sequence of row and column indices.

        Returns:
            np.ndarray: An array of linear indices corresponding to the
            row-column indices.

        Example:
            >>> odmr = ODMR(...)
            >>> rc = [(0, 1), (1, 2), (2, 3)]
            >>> idx = odmr.rc2idx(rc)
            >>> print(idx)
            [1, 6, 11]
        """
        return QDMpy.utils.rc2idx(rc, self.data_shape)  # type: ignore[arg-type]

    def idx2rc(self, idx: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts linear indices to row-column indices for the given data shape.

        Args:
            idx (ArrayLike): A sequence of linear indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing the row and
            column indices
                                        corresponding to the linear indices.

        Example:
            >>> odmr = ODMR(...)
            >>> idx = [1, 6, 11]
            >>> row, col = odmr.idx2rc(idx)
            >>> print(row, col)
            [0, 1, 2] [1, 2, 3]
        """
        return QDMpy.utils.idx2rc(idx, self.data_shape)  # type: ignore[arg-type]

    THRESHOLD = 0.001

    def most_divergent_from_mean(self) -> Tuple[int, int]:
        """
        Get the most divergent pixel from the mean in data coordinates,
        considering pixels with divergence less than the defined threshold.

        Returns:
            Tuple[int, int]: A tuple (i, j) representing the row and column
            indices of the
                            most divergent pixel within the threshold limit.

        Example:
            >>> odmr = ODMR(...)
            >>> most_divergent_pixel = odmr.most_divergent_from_mean()
            >>> print(most_divergent_pixel)
            (3, 7)
        """
        delta = self.delta_mean.copy()
        delta[delta > self.THRESHOLD] = np.nan
        return np.unravel_index(np.nanargmax(delta), delta.shape)

    # from methods FROM / IMPORT

    @classmethod
    def from_qdmio(cls, data_folder: Union[str, os.PathLike], **kwargs: Any) -> "ODMR":
        """
        Loads QDM data from a QDMio formatted Matlab file and creates an ODMR
        instance.

        Args:
            data_folder (Union[str, os.PathLike]): Path to the directory
            containing the QDM data files.

        Returns:
            ODMR: An instance of the ODMR class with the loaded QDM data.

        Raises:
            WrongFileTypeError: If no .mat files are found.

        Notes:
            Filenames starting with '#' are ignored.

        Example:
            >>> odmr_data = ODMR.from_qdmio('path/to/data_folder')
            >>> print(odmr_data)
            ODMR(data_array=(2, 1, 10000, 50), scan_dimensions=(100, 100), n_pol=2, n_frange=1, n_pixel=10000, n_freqs=50)
        """

        # Scan the directory for all run files
        with os.scandir(data_folder) as entries:
            run_files = [
                entry.name
                for entry in entries
                if entry.name.endswith(".mat")
                and "run_" in entry.name
                and not entry.name.startswith("#")
            ]

        # Raise an error if no run files are found
        if not run_files:
            raise WrongFileNumber("No run files found in folder.")

        cls.LOG.info(f"Reading {len(run_files)} run_* files.")

        # Initialize the data array
        data_array = None

        # Loop over all run files, load the data and stack it
        for mfile in run_files:
            cls.LOG.debug(f"Reading file {mfile}")

            # Load the raw data from the Matlab file
            raw_data = utils.loadmat(os.path.join(data_folder, mfile))

            # Stack the data using the _qdmio_stack_data method
            d = cls._qdmio_stack_data(raw_data)
            data_array = d if data_array is None else np.stack((data_array, d), axis=0)

        # Fix dimensions if only one run file was found -> only one polarity
        if len(run_files) == 1:
            data_array = data_array[np.newaxis, :, :, :]

        # Extract the image shape from the raw data
        img_shape = np.array(
            [
                np.squeeze(raw_data["imgNumRows"]),
                np.squeeze(raw_data["imgNumCols"]),
            ],
            dtype=int,
        )

        # Extract the number of frequencies and frequency array from the raw
        # data
        n_freqs = int(np.squeeze(raw_data["numFreqs"]))
        frequency_array = np.squeeze(raw_data["freqList"]).astype(np.float32)

        # Split the frequency array if needed
        if n_freqs != len(frequency_array):
            frequency_array = np.array([frequency_array[:n_freqs], frequency_array[n_freqs:]])

        # Create and return an instance of the ODMR class with the loaded data
        return cls(
            data_array=data_array,
            scan_dimensions=img_shape,
            frequency_array=frequency_array,
            **kwargs,
        )

    @classmethod
    def _qdmio_stack_data(cls, mat_dict: dict) -> np.ndarray:
        """
        Stack the data in the ODMR object from QDMio matlab file data.

        Args:
            mat_dict (dict): Dictionary containing the data from the QDMio
            matlab file.

        Returns:
            np.ndarray: A numpy array containing the data for both frequency
            ranges.

        Raises:
            ValueError: If the number of image stacks is not 2 or 4.

        Notes:
            The number of image stacks determines whether the data is in
            low-frequency mode (50 frequencies) or high-frequency mode (101
            frequencies).

        Example:
            >>> mat_data = loadmat('path_to_mat_file.mat')
            >>> stacked_data = ODMR._qdmio_stack_data(mat_data)
            >>> print(stacked_data.shape)
            (2, 10000, 50)  # For low-frequency mode
        """
        n_img_stacks = len([k for k in mat_dict if "imgStack" in k])
        img_stack1, img_stack2 = [], []

        if n_img_stacks == 2:
            # IF ONLY 2 IMG-STACKS, THEN WE ARE IN LOW freq. MODE (50 freq.)
            # imgStack1: [n_freqs, n_pixels] -> transpose to [n_pixels, n_freqs]
            cls.LOG.debug("Two ImgStacks found: Stacking data from imgStack1 and imgStack2.")
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
    def get_norm_factors(cls, data: ArrayLike, method: str = "max", **kwargs) -> np.ndarray:
        """
        Return the normalization factors for the data.

        Args:
            data (ArrayLike): The data to normalize. This should be a numpy
            array or
                other array-like object with a shape of (n_pol, n_frange,
                n_pixels, n_freqs).
            method (str): The normalization method to use. Supported methods
            are:
                - "max" (default): Normalize by the maximum value of each pixel
                  spectrum
                - "mean": Normalize by the mean value of each pixel spectrum
                - "std": Normalize by the standard deviation of each pixel
                  spectrum
                - "mad": Normalize by the median absolute deviation of each
                  pixel spectrum
                - "l2": Normalize by the L2-norm of each pixel spectrum

        Returns:
            np.ndarray: A 1D array of normalization factors with shape
            (n_pixels,).

        Raises:
            NotImplementedError: if the method is not implemented.

        Examples:
            >>> data = np.random.rand(2, 2, 100, 50)
            >>> norm_factors = ODMR.get_norm_factors(data, method="max")
        """

        # Determine the normalization factors based on the selected method
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

    @property
    def data_shape(self) -> np.ndarray:
        """
        Calculate and return the shape of the data array in terms of x and y
        dimensions.

        Returns:
            np.ndarray: A 1D numpy array with two elements, representing the x
            and y dimensions
                        of the data array.
        """
        y_dim = np.sqrt(self.n_pixel / self._data_xy_ratio)
        x_dim = self.n_pixel / y_dim
        return np.array([x_dim, y_dim]).astype(int)

    @property
    def img_shape(self) -> np.ndarray:
        """
        Get the shape of the image array.

        Returns:
            np.ndarray: A 1D numpy array with two elements, representing the
            number of rows
                        and columns of the image array.
        """
        return self._img_shape

    @property
    def frequencies(self) -> np.ndarray:
        """
        Get the frequencies for the ODMR data.

        Returns:
            np.ndarray: A 1D or 2D numpy array of frequency values. If the
            frequencies
                        are not cropped, it returns the original frequency
                        array. Otherwise, it returns the cropped frequency
                        array.
        """
        if self._frequencies_cropped is None:
            return self._frequencies
        else:
            return self._frequencies_cropped

    @property
    def f_hz(self) -> np.ndarray:
        """
        Get the frequencies of the ODMR in Hz.

        Returns:
            np.ndarray: A 1D or 2D numpy array of frequency values in Hz.
        """
        return self.frequencies

    @property
    def f_ghz(self) -> np.ndarray:
        """
        Get the frequencies of the ODMR in GHz.

        Returns:
            np.ndarray: A 1D or 2D numpy array of frequency values in GHz.
        """
        return self.frequencies / 1e9

    @property
    def global_factor(self) -> float:
        """
        Get the global factor value.

        Returns:
            float: The global factor value.
        """
        return self._gf_factor

    @property
    def data(self) -> np.ndarray:
        """
        Get the data array, either the edited data or the raw data.

        Returns:
            np.ndarray: The edited data array if available, otherwise the raw
            data array.
        """
        if self._data_edited is None:
            return np.ascontiguousarray(self._raw_data)
        else:
            return np.ascontiguousarray(self._data_edited)

    @property
    def delta_mean(self) -> np.ndarray:
        """
        Calculate the deviation of the data from its mean.

        Returns:
            np.ndarray: The deviation of the data from its mean.
        """
        return np.sum(np.square(self.data - self.mean_odmr[:, :, np.newaxis, :]), axis=-1)

    @property
    def mean_odmr(self) -> np.ndarray:
        """
        Calculate the mean of the data.

        Returns:
            np.ndarray: The mean of the data array.
        """
        return self.data.mean(axis=-2)

    @property
    def raw_contrast(self) -> np.ndarray:
        """
        Calculate the minimum of the microwave sweep for each pixel.

        Returns:
            np.ndarray: The minimum of the microwave sweep for each pixel.
        """
        return np.min(self.data, -2)

    @property
    def mean_contrast(self) -> np.ndarray:
        """
        Calculate the mean of the minimum of the microwave sweep for each pixel.

        Returns:
            np.ndarray: The mean of the minimum of the microwave sweep for each
            pixel.
        """
        return np.mean(self.raw_contrast)

    def _mean_baseline(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the mean baseline of the data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
            mean baselines of the left side, right side, and the overall mean
            baseline.
        """
        # Calculate the mean baseline for the left side of the data
        baseline_left_mean = np.mean(self.mean_odmr[:, :, :5], axis=-1)
        # Calculate the mean baseline for the right side of the data
        baseline_right_mean = np.mean(self.mean_odmr[:, :, -5:], axis=-1)

        # Calculate the overall mean baseline by averaging the left and right
        # baselines
        baseline_mean = np.concatenate(
            [baseline_left_mean[..., np.newaxis], baseline_right_mean[..., np.newaxis]], axis=-1
        ).mean(axis=-1)
        return baseline_left_mean, baseline_right_mean, baseline_mean

    @property
    def bin_factor(self) -> int:
        """
        Get the bin factor, which is the product of the pre-bin factor and the
        bin factor.

        Returns:
            int: The bin factor.
        """
        return self._bin_factor * self._pre_bin_factor

    def _apply_edit_stack(self, **kwargs: Any) -> None:
        """
        Apply the edit stack to the data, applying each edit function in the
        stack.

        Args:
            **kwargs: Keyword arguments to be passed to the edit functions.

        Returns:
            None
        """
        self.LOG.debug("Applying edit stack")
        for edit_func in self._edit_stack:
            if edit_func is not None:
                edit_func(**kwargs)  # type: ignore[operator]

    def reset_data(self, **kwargs: Any) -> None:
        """
        Reset the data to its original state (raw data), and reset all related
        flags.

        Args:
            **kwargs: Keyword arguments (unused).

        Returns:
            None
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
        """
        Normalize the data using the specified method or the instance's
        normalization method.

        Args:
            method (Union[str, None], optional): The normalization method to
            use. If None, the
                instance's normalization method will be used. Default is None.
            **kwargs: Keyword arguments to be passed to the normalization
            function.

        Returns:
            None
        """
        if method is None:
            method = self._norm_method
        self._edit_stack[1] = self._normalize_data
        self._apply_edit_stack(method=method)

    def _normalize_data(self, method: str = "max", **kwargs: Any) -> None:
        """
        Normalize the data using the specified method or provided factors.

        Args:
            method (str, optional): The normalization method to use. Supported
            methods are "max"
                (default), "mean", "std", "mad", and "l2".
            **kwargs: Additional arguments to pass to the normalization method.
            If a "factors"
                key is present, the method will use the specified factors
                instead of calculating new ones using the specified method.

        Returns:
            None
        """
        if "factors" in kwargs:
            factors = kwargs["factors"]
        else:
            factors = self.get_norm_factors(self.data, method=method, **kwargs)

        self._norm_factors = factors
        self.LOG.debug(f"Normalizing data with method: {method}")
        self._norm_method = method
        self.is_normalized = True
        self._data_edited /= self._norm_factors

    def apply_outlier_mask(
        self, outlier_mask: Union[np.ndarray, None] = None, **kwargs: Any
    ) -> None:
        """
        Apply the outlier mask to the data.

        Args:
            outlier_mask (Union[np.ndarray, None], optional): The outlier mask
            to apply.
                If None, the instance's outlier mask will be used. Default is
                None.
            **kwargs: Additional arguments (unused).

        Returns:
            None
        """
        self._edit_stack[3] = self._apply_outlier_mask

        if outlier_mask is None:
            outlier_mask = self.outlier_mask

        self.outlier_mask = outlier_mask
        self._apply_edit_stack()

    def _apply_outlier_mask(self, **kwargs: Any) -> None:
        """
        Apply the outlier mask to the data.

        Args:
            **kwargs: Additional arguments (unused).

        Returns:
            None
        """
        if self.outlier_mask is None:
            self.LOG.debug("No outlier mask applied.")
            return
        self.LOG.debug("Applying outlier mask")
        self._data_edited[:, :, self.outlier_mask.reshape(-1), :] = np.nan
        self.LOG.debug(
            f"Outlier mask applied, set {np.isnan(self._data_edited[0, 0, :, 0]).sum()} to NaN."
        )

    def bin_data(self, bin_factor: int, **kwargs: Any) -> None:
        """
        Bin the data using the specified bin factor.

        Args:
            bin_factor (int): The bin factor to use for binning the data.
            **kwargs: Additional arguments to pass to the binning method.

        Returns:
            None
        """
        self._edit_stack[2] = self._bin_data
        self._apply_edit_stack(bin_factor=bin_factor)

    def _bin_data(
        self,
        bin_factor: Optional[Union[float, None]] = None,
        func: Callable = np.nanmean,
        **kwargs: Any,
    ) -> None:
        """
        Bin the data from the raw data using the specified bin factor and
        function.

        Args:
            bin_factor (Optional[Union[float, None]], optional): The bin factor
            to use for binning the data.
                Default is None.
            func (Callable, optional): The function to use for block averaging.
            Default is np.nanmean. **kwargs: Additional arguments (unused).

        Returns:
            None
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
            self.data_shape[0],
            self.data_shape[1],
            self.n_freqs,
        )  # reshapes the data to the scan dimensions

        _odmr_binned = block_reduce(
            reshape_data,
            block_size=(1, 1, int(bin_factor), int(bin_factor), 1),
            func=func,  # uses the specified function
            cval=func(reshape_data),  # replaces with the specified function's default value
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

    def remove_overexposed(self, threshold: float = 1.0, **kwargs: Any) -> None:
        """Remove overexposed pixels from the data.

        Args:
          threshold: The threshold value to use. Pixels with a sum greater than
          or equal to the threshold
            are considered overexposed. Default is 1.0.
          **kwargs:

        Returns:

        """
        if self._data_edited is None:
            return self.LOG.warning("No data to remove overexposed pixels from.")

        overexposed_mask = (
            np.sum(self._data_edited, axis=-1) >= threshold * self._data_edited.shape[-1]
        )

        if np.sum(overexposed_mask) > 0:
            self.LOG.warning(f"ODMR: {np.sum(overexposed_mask)} pixels are overexposed")

            # Remove overexposed pixels by setting their value to NaN
            self._data_edited[overexposed_mask] = np.nan

    ### CORRECTION METHODS ###
    def calc_gf_correction(self, gf: float) -> np.ndarray:
        """Calculate the global fluorescence correction.

        Args:
          gf: The global fluorescence factor

        Returns: The global fluorescence correction
        """
        baseline_left_mean, baseline_right_mean, baseline_mean = self._mean_baseline()
        return gf * (self.mean_odmr - baseline_mean[:, :, np.newaxis])

    def correct_glob_fluorescence(self, gf_factor: float, **kwargs: Any) -> None:
        """Correct the data for the gradient factor.

        Args:
          gf_factor: **kwargs:

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
          gf_factor: global fluorescence factor (Default value = None) **kwargs:
          pass through for additional _apply_edit_stack kwargs

        Raises:
          TypeError: If `gf_factor` is not a float or None
        """
        if gf_factor is not None and not isinstance(gf_factor, float):
            raise TypeError(
                f"Expected `gf_factor` to be a float or None, but got {type(gf_factor)}"
            )

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
          gf_factor:  (Default value = None) idx:  (Default value = None)

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
                    ax[p, f].plot(self.f_ghz[f], d, "k:", label=f"current: GF={self._gf_factor}")

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
                ax[p, f].plot(self.f_ghz[f], 1 + new_correct[p, f], "r--", label="correction")
                ax[p, f].set_title(f"{['+', '-'][p]},{['<', '>'][f]}")
                ax[p, f].legend()
                # , ylim=(0, 1.5))
                ax[p, f].set(ylabel="ODMR contrast", xlabel="Frequency [GHz]")
            plt.tight_layout()


def main() -> None:
    QDMpy.LOG.setLevel(logging.DEBUG)
    odmr = ODMR.from_qdmio("/media/paleomagnetism/Fort Staff/QDM/2022_Hawaii_1907/20221207_14G")
    # for i in [4, 6, 8, 16]: odmr.bin_data(i)


if __name__ == "__main__":
    main()
