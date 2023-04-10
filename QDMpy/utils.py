import logging
import os
import sys
from typing import Any, Optional, Sequence, Tuple, Union, List

import mat73
import matplotlib.image as mpimg
import numpy as np
import scipy.io
from numpy.typing import ArrayLike
from numba import guvectorize, float64

from pypole.convert import dim2xyz, xyz2dim

from QDMpy._core.convert import b2shift, project
from QDMpy._core.models import esr15n

# from QDMpy import QDMpy.CONFIG_FILE, QDMpy.CONFIG_INI, QDMpy.CONFIG_PATH

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = logging.getLogger(__name__)

MILLNAMES = ["n", "μ", "m", "", " K", " M", " B", " T"]


def human_readable_number(n: float, sign: int = 1) -> str:
    """
    Convert a number to a human-readable string using metric prefixes.

    This function takes a number and returns a string representation with an appropriate metric prefix
    (e.g., " K" for thousands, " M" for millions, etc.). The number of digits after the decimal point
    can be specified using the `sign` parameter.

    Args:
        n (float): The number to convert.
        sign (int, optional): The number of digits after the decimal point. Default is 1.

    Returns:
        str: The human-readable string representation of the number with an appropriate metric prefix.

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
    Convert a 1D index to 2D row and column coordinates based on the given array shape.

    Args:
        idx (Union[int, List[int], np.ndarray]): An integer or an array-like of integers representing the 1D index/indices.
        shape (Tuple[int, int]): A tuple representing the shape of the 2D array (rows, columns).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays containing the row and column coordinates, respectively.

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
    Converts row and column coordinates to a linear index based on the given array shape.

    Args:
        row_col (List[Tuple[int, int]]): A list of tuples representing row and column coordinates to be converted to linear indices.
        shape (Tuple[int, int]): A tuple representing the shape of the 2D array (rows, columns).

    Returns:
        np.ndarray: A numpy array containing the linear indices corresponding to the row and column coordinates.

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
        x (np.ndarray): X values for the fit.
        y (np.ndarray): Y values for the fit.
        z (np.ndarray): Z values for the fit.
        kx (Optional[int]): Polynomial order in x. Default is 3.
        ky (Optional[int]): Polynomial order in y. Default is 3.
        order (Optional[Union[None, int]]): If int, coefficients up to a maximum of kx + ky <= order are considered. Default is None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Solution, residuals, rank, singular values.

    See: https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python

    See also: np.linalg.leastsq

    Notes:
        Resulting fit can be plotted with:
        >>> np.polynomial.polynomial.polygrid2d(x, y, solution.reshape((kx + 1, ky + 1)))

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


from itertools import product

ZFS = 2.87


def generate_parameter(
    projected_shifts: np.ndarray, width: float, contrast: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the parameters (low/high frequency range) for the QDMpy simulation.

    Args:
        projected_shifts (np.ndarray): Shifts of the peaks in GHz.
        width (float): Width of the peaks in GHz.
        contrast (float): Contrast of the peaks.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left resonance and right resonance parameters for the QDMpy simulation.

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
    Calculates the models for all dec/inc combinations of the current applied field.

    This function is meant to calculate the "safe" field margins for the current applied field.

    Args:
        freqs (np.ndarray): Frequency values for the models.
        bias_xyz (np.ndarray): Bias field in XYZ coordinates.
        b_source (float): Source field magnitude.
        nv_direction (np.ndarray): NV direction as unit vectors.
        width (float): Width of the peaks in GHz.
        contrast (float): Contrast of the peaks.

    Returns:
        np.ndarray: The calculated models for all dec/inc combinations for NV axes > 0 of the current applied field.

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
    """Generates a set of dec/inc combinations for a given source field.

    Args:
        b_source (float): The source field magnitude in microTesla.
        n (int, optional): The number of dec/inc combinations to generate. Defaults to 10.

    Returns:
        np.ndarray: The generated dec/inc combinations.
    """

    dec = np.linspace(0, 360, n)
    inc = np.linspace(-90, 90, n)
    di = np.array(list(product(dec, inc)))
    source_dim = np.stack([di[:, 0], di[:, 1], np.ones(di.shape[0]) * b_source])

    return source_dim.T


def rms(data):
    """Calculate the root mean square of a data set.

    Args:
      data: data set

    Returns:
      root mean square

    """
    return np.sqrt(np.mean(np.square(data)))


def has_csv(lst: Sequence[Union[str, bytes, os.PathLike[Any]]]) -> bool:
    """Checks if a list of files contains a csv file.

    Args:
      lst: list of str
      lst: list:

    Returns:
      bool

    """
    return any(".csv" in str(s) for s in lst)


def get_image_file(lst: Sequence[Union[str, bytes, os.PathLike[Any]]]) -> str:
    """Returns the first image file in the list.

    Args:
      lst: list of str
    list of files to load the image from
      lst: list:

    Returns:
      name of the image file

    """
    if has_csv(lst):
        lst = [s for s in lst if ".csv" in str(s)]
    else:
        lst = [s for s in lst if ".jpg" in str(s)]
    return str(lst[0])


def get_image(
    folder: Union[str, bytes, os.PathLike],
    lst: Sequence[Union[str, bytes, os.PathLike]],
) -> np.ndarray:
    """Loads an image from a list of files.

    Args:
        folder: folder of the image
        lst: list of files to load the image from
    Returns: loaded image from either csv or jpg file if csv is not available

    """
    folder = str(folder)

    if has_csv(lst):
        img = np.loadtxt(os.path.join(folder, get_image_file(lst)))
    else:
        img = mpimg.imread(os.path.join(folder, get_image_file(lst)))
    return np.array(img)


def double_norm(data: np.ndarray, axis: Optional[Union[int, None]] = None) -> np.ndarray:
    """Normalizes data from 0 to 1.

    Args:
      data: np.array
    data to normalize
      axis: int
    axis to normalize
      data: np.ndarray:
      axis: Optional[Union[int:
      None]]:  (Default value = None)

    Returns:
      np.array
      normalized data

    """
    mn = np.expand_dims(np.min(data, axis=axis), data.ndim - 1)
    data -= mn
    mx = np.expand_dims(np.max(data, axis=axis), data.ndim - 1)
    data /= mx
    return data


def loadmat(path):
    """Loads a Matlab file using the correct function (i.e. scipy.io.loadmat or mat73.loadmat)
    and returns the raw data.

        Args:
            data_folder:
            mfile:

        Returns:
            raw data
    """
    try:
        raw_data = scipy.io.loadmat(path)
    except NotImplementedError:
        raw_data = mat73.loadmat(path)
    return raw_data


def main() -> None:
    """Main function."""
    print(human_readable_number(0.001, 10))


if __name__ == "__main__":
    main()
