"""
This module, utils.py, provides a collection of utility functions for working
with magnetic resonance spectroscopy data, file handling, and data manipulation.
The functions in this module can be used in conjunction with other modules in
the project to perform data analysis, visualization, and fitting tasks.

The module includes functions for:

    Converting between different coordinate systems and representations (e.g.,
    Cartesian and spherical coordinates, linear and row-column indexing).
    Calculating root mean square, normalizing data, and generating dec/inc
    combinations for magnetic fields. Fitting two-dimensional polynomials,
    creating parameter arrays for QDMpy simulations, and running Monte Carlo
    simulations for different field configurations. Loading and saving data in
    various formats, such as CSV, JPG, and MATLAB files. Checking for the
    presence of specific file types in a list of files, and loading image data
    from a list of files in a specified folder.

Example usage:

``` python
  import numpy as np from utils import polyfit2d, loadmat, double_norm

  # Load data from a MATLAB file data = loadmat("path/to/matlab_file.mat")

  # Normalize the data along a specific axis normalized_data = double_norm(data,
  axis=1)

  # Perform a 2D polynomial fit x = np.linspace(0, 1, 100) y = np.linspace(0, 1,
  100) z = np.random.rand(100, 100) solution, res, rank, s = polyfit2d(x, y, z,
  kx=3, ky=3)
```

The functions in this module are designed to be easily integrated into larger
data processing pipelines, allowing for streamlined and efficient data analysis
workflows.
"""

import logging
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict
from os import PathLike
import mat73
import matplotlib.image as mpimg
import numpy as np
import scipy.io
from numba import float64, guvectorize
from numpy.typing import ArrayLike
from pypole.convert import dim2xyz, xyz2dim
from QDMpy._core.convert import b2shift, project
from QDMpy._core.models import esr15n

# from QDMpy import QDMpy.CONFIG_FILE, QDMpy.CONFIG_INI, QDMpy.CONFIG_PATH

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = logging.getLogger(__name__)

MILLNAMES = ["n", "μ", "m", "", " K", " M", " B", " T"]


ZFS = 2.87


def human_readable_number(n: float, sign: int = 1) -> str:
    """
    Convert a number to a human-readable string using metric prefixes.

    This function takes a number and returns a string representation with an
    appropriate metric prefix (e.g., " K" for thousands, " M" for millions,
    etc.). The number of digits after the decimal point can be specified using
    the `sign` parameter.

    Args:
        n (float): The number to convert. sign (int, optional): The number of
        digits after the decimal point. Default is 1.

    Returns:
        str: The human-readable string representation of the number with an
        appropriate metric prefix.

    Example:
        >>> millify(1500)
        '1.5 K'
        >>> millify(1500, sign=2)
        '1.50 K'
        >>> millify(0.0015)
        '1.5 m'
    """
    millidx = max(0, min(len(MILLNAMES) - 1, int(np.floor(0 if n == 0 else np.log10(abs(n)) / 3))))

    return f"{n / 10 ** (3 * millidx):.{sign}f}{MILLNAMES[millidx + 3]}"


def idx2rc(idx: ArrayLike, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D index to 2D row and column coordinates based on the given array
    shape.

    Args:
        idx (Union[int, List[int], np.ndarray]): An integer or an array-like of
        integers representing the 1D index/indices. shape (Tuple[int, int]): A
        tuple representing the shape of the 2D array (rows, columns).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing
        the row and column coordinates, respectively.

    Example:
        >>> idx = 6
        >>> shape = (3, 3)
        >>> idx2rc(idx, shape)
        (array([2]), array([0]))

        >>> idx = [6, 7, 8]
        >>> shape = (3, 3)
        >>> idx2rc(idx, shape)
        (array([2, 2, 2]), array([0, 1, 2]))
    """
    idx = np.atleast_1d(idx)
    idx = np.array(idx).astype(int)
    return np.unravel_index(idx, shape)  # type: ignore[return-value]


def rc2idx(row_col: List[Tuple[int, int]], shape: tuple) -> np.ndarray:
    """
    Converts row and column coordinates to a linear index based on the given
    array shape.

    Args:
        row_col (List[Tuple[int, int]]): A list of tuples representing row and
        column coordinates to be converted to linear indices. shape (Tuple[int,
        int]): A tuple representing the shape of the 2D array (rows, columns).

    Returns:
        np.ndarray: A numpy array containing the linear indices corresponding to
        the row and column coordinates.

    Example:
        >>> row_col = [(2, 0)]
        >>> shape = (3, 3)
        >>> rc2idx(row_col, shape)
        array([6])

        >>> row_col = [(2, 0), (2, 1), (2, 2)]
        >>> shape = (3, 3)
        >>> rc2idx(row_col, shape)
        array([6, 7, 8])
    """
    row_col_arr = np.array(row_col).astype(int)
    return np.ravel_multi_index(row_col_arr.T, shape)


def polyfit2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    kx: Optional[int] = 3,
    ky: Optional[int] = 3,
    order: Optional[Union[None, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-dimensional polynomial fitting by least squares.

    Fits the functional form f(x, y) = z.

    Args:
        x (np.ndarray): X values for the fit. y (np.ndarray): Y values for the
        fit. z (np.ndarray): Z values for the fit. kx (Optional[int]):
        Polynomial order in x. Default is 3. ky (Optional[int]): Polynomial
        order in y. Default is 3. order (Optional[Union[None, int]]): If int,
        coefficients up to a maximum of kx + ky <= order are considered. Default
        is None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Solution,
        residuals, rank, singular values.

    See:
    https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python

    See also: np.linalg.leastsq

    Notes:
        Resulting fit can be plotted with: >>>
        np.polynomial.polynomial.polygrid2d(x, y, solution.reshape((kx + 1, ky +
        1)))

    Example:
        >>> x = np.array([0, 1, 2])
        >>> y = np.array([0, 1, 2])
        >>> z = np.array([[0, 1, 4], [1, 2, 5], [4, 5, 8]])
        >>> solution, res, rank, s = polyfit2d(x, y, z)
    """
    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones(shape=(kx + 1, ky + 1))  # type: ignore[operator]

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    solution, res, rank, s = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
    return solution, res, rank, s


def generate_parameter(
    projected_shifts: np.ndarray, width: float, contrast: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the parameters (low/high frequency range) for the QDMpy simulation.

    Args:
        projected_shifts (np.ndarray): Shifts of the peaks in GHz. width
        (float): Width of the peaks in GHz. contrast (float): Contrast of the
        peaks.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left resonance and right resonance
        parameters for the QDMpy simulation.

    Example:
        >>> projected_shifts = np.array([0.1, 0.2, 0.3])
        >>> width = 0.05
        >>> contrast = 0.8
        >>> left_resonance, right_resonance = generate_parameter(projected_shifts, width, contrast)
    """
    left_resonance = np.stack(
        [
            ZFS - projected_shifts,
            np.ones(projected_shifts.shape[0]) * width,
            np.ones(projected_shifts.shape[0]) * contrast,
            np.ones(projected_shifts.shape[0]) * contrast,
            np.zeros(projected_shifts.shape[0]),
        ],
        axis=1,
    )
    right_resonance = np.stack(
        [
            ZFS + projected_shifts,
            np.ones(projected_shifts.shape[0]) * width,
            np.ones(projected_shifts.shape[0]) * contrast,
            np.ones(projected_shifts.shape[0]) * contrast,
            np.zeros(projected_shifts.shape[0]),
        ],
        axis=1,
    )
    return left_resonance, right_resonance


def monte_carlo_models(freqs, bias_xyz, b_source, nv_direction, width, contrast):
    """
    Calculates the models for all dec/inc combinations of the current applied
    field.

    This function is meant to calculate the "safe" field margins for the current
    applied field.

    Args:
        freqs (np.ndarray): Frequency values for the models. bias_xyz
        (np.ndarray): Bias field in XYZ coordinates. b_source (float): Source
        field magnitude. nv_direction (np.ndarray): NV direction as unit
        vectors. width (float): Width of the peaks in GHz. contrast (float):
        Contrast of the peaks.

    Returns:
        np.ndarray: The calculated models for all dec/inc combinations for NV
        axes > 0 of the current applied field.

    Example:
        >>> freqs = np.linspace(2.8, 3.3, 1000)
        >>> bias_xyz = np.array([0, 0, 0])
        >>> b_source = 10.0
        >>> nv_direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> width = 0.05
        >>> contrast = 0.8
        >>> models = monte_carlo_models(freqs, bias_xyz, b_source, nv_direction, width, contrast)
    """
    # generate the dec/inc combinations at current source field
    source_dim = generate_possible_dim(b_source=b_source, n=10)
    source_xyz = dim2xyz(source_dim)

    total_field_xyz = source_xyz + bias_xyz

    # calculate all projected shifts for all specified NVs
    projected_fields = project(total_field_xyz, nv_direction)
    projected_shifts = b2shift(projected_fields, in_unit="microT", out_unit="GHz")

    parameters = generate_parameter(
        projected_shifts=projected_shifts, width=width, contrast=contrast
    )
    all_nv_models = [esr15n(freqs, p) for p in parameters]  # calculate left/right spectra
    return np.min(all_nv_models, axis=0)  # returns only the minimum


def generate_possible_dim(b_source: float, n: int = 10) -> np.ndarray:
    """
    Generates a set of dec/inc combinations for a given source field.

    Args:
        b_source (float): The source field magnitude in microTesla. n (int,
        optional): The number of dec/inc combinations to generate. Defaults to
        10.

    Returns:
        np.ndarray: The generated dec/inc combinations.

    Example:
        >>> b_source = 10.0
        >>> n = 5
        >>> possible_dim = generate_possible_dim(b_source, n)
    """

    dec = np.linspace(0, 360, n)
    inc = np.linspace(-90, 90, n)
    di = np.array([[d, i] for d in dec for i in inc])
    source_dim = np.stack([di[:, 0], di[:, 1], np.ones(di.shape[0]) * b_source])

    return source_dim.T


def rms(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate the root mean square (RMS) of a data set.

    Args:
        data (np.ndarray): The data set as a NumPy array. axis (Optional[int]):
        The axis along which to compute the RMS. If None (default),
            the RMS is computed over the entire array.

    Returns:
        Union[float, np.ndarray]: The calculated root mean square value(s).

    Example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> rms_value = rms(data, axis=0)
    """

    return np.sqrt(np.mean(np.square(data), axis=axis))


def has_csv(lst: Sequence[Union[str, bytes, PathLike]]) -> bool:
    """Checks if a list of files contains a csv file.

    Args:
      lst: list of file paths

    Returns:
      bool: True if a CSV file is found, False otherwise

    """
    return any(Path(path).suffix == ".csv" for path in lst)


def get_image_file(lst: Sequence[Union[str, bytes, PathLike]]) -> str:
    """Returns the first image file in the list.

    Args:
      lst: list of file paths

    Returns:
      str: name of the image file

    """
    if has_csv(lst):
        image_files = [path for path in lst if Path(path).suffix == ".csv"]
    else:
        image_files = [path for path in lst if Path(path).suffix == ".jpg"]

    return str(image_files[0])


def get_image(
    folder: Union[str, bytes, PathLike],
    lst: Sequence[Union[str, bytes, PathLike]],
) -> np.ndarray:
    """Loads an image from a list of files.

    Args:
        folder: folder of the image lst: list of files to load the image from

    Returns:
        np.ndarray: loaded image from either csv or jpg file if csv is not
        available

    """
    folder = Path(folder)
    image_file = folder / get_image_file(lst)

    if image_file.suffix == ".csv":
        img = np.loadtxt(image_file)
    else:
        img = mpimg.imread(image_file)
    return np.array(img)


# GLOBAL FLUORESCENCE FUNCTIONS
def _mean_baseline(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mean baseline of the data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        mean baselines of the left side, right side, and the overall mean
        baseline.
    """
    mean_odmr = data.mean(axis=-2)

    # Calculate the mean baseline for the left side of the data
    baseline_left_mean = np.mean(mean_odmr[..., :5], axis=-1)
    # Calculate the mean baseline for the right side of the data
    baseline_right_mean = np.mean(mean_odmr[..., -5:], axis=-1)

    # Calculate the overall mean baseline by averaging the left and right
    # baselines
    baseline_mean = np.mean([baseline_left_mean, baseline_right_mean], axis=-1)
    return baseline_left_mean, baseline_right_mean, baseline_mean


def calc_gf_correction(data, gf_factor: float) -> np.ndarray:
    """Calculate the global fluorescence correction.

    Args:
      gf: The global fluorescence factor

    Returns: The global fluorescence correction
    """
    mean_odmr = data.mean(axis=-2)

    baseline_left_mean, baseline_right_mean, baseline_mean = _mean_baseline(data)
    return gf_factor * (
        mean_odmr[:, :, np.newaxis, :] - baseline_mean[:, :, np.newaxis, np.newaxis]
    )


# NORMALIZATION FUNCTIONS


def get_norm_factors(data: ArrayLike, method: str = "max", **kwargs) -> np.ndarray:
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
        LOG.debug(
            f"Determining normalization factor from maximum value of each pixel spectrum. "
            f"Shape of factors: {factors.shape}"
        )
    elif method == "mean":
        factors = np.mean(data, axis=-1, keepdims=True)
        LOG.debug(
            f"Determining normalization factor from mean value of each pixel spectrum. "
            f"Shape of factors: {factors.shape}"
        )
    elif method == "std":
        factors = np.std(data, axis=-1, keepdims=True)
        LOG.debug(
            f"Determining normalization factor from standard deviation of each pixel spectrum. "
            f"Shape of factors: {factors.shape}"
        )
    elif method == "mad":
        med = np.median(data, axis=-1, keepdims=True)
        factors = np.median(np.abs(data - med), axis=-1, keepdims=True) / 0.6745
        LOG.debug(
            f"Determining normalization factor from median absolute deviation of each pixel spectrum. "
            f"Shape of factors: {factors.shape}"
        )
    elif method == "l2":
        factors = np.linalg.norm(data, axis=-1, keepdims=True)
        LOG.debug(
            f"Determining normalization factor from L2-norm of each pixel spectrum. "
            f"Shape of factors: {factors.shape}"
        )
    else:
        raise NotImplementedError(f'Method "{method}" not implemented.')

    return factors


def double_norm(data: np.ndarray, axis: Optional[Union[int, None]] = None) -> np.ndarray:
    """Normalizes data from 0 to 1.

    Args:
        data: np.ndarray, data to normalize axis: Optional[Union[int, None]],
        axis to normalize (Default value = None)

    Returns:
        np.ndarray: normalized data

    """
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    return (data - data_min) / (data_max - data_min)


def loadmat(path: Union[str, bytes, os.PathLike]) -> Dict[str, Any]:
    """Loads a MATLAB file using the appropriate function (i.e.,
    scipy.io.loadmat or mat73.loadmat) and returns the raw data.

    Args:
        path: Union[str, bytes, os.PathLike], path to the MATLAB file

    Returns:
        Dict[str, Any]: raw data from the MATLAB file as a dictionary with
        variable names as keys
    """
    try:
        raw_data = scipy.io.loadmat(path)
    except NotImplementedError:
        raw_data = mat73.loadmat(path)
    return raw_data
