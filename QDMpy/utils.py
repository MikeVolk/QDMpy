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


def millify(n: float, sign: int = 1) -> str:
    """Convert a number to a human readable string.

    Args:
      n: float
    number to convert
      sign: int
    number of digits after the decimal point
      n: float:
      sign: int:  (Default value = 1)

    Returns:
      str
      human readable string

    """
    millidx = max(0, min(len(MILLNAMES) - 1, int(np.floor(0 if n == 0 else np.log10(abs(n)) / 3))))

    return f"{n / 10 ** (3 * millidx):.{sign}f}{MILLNAMES[millidx + 3]}"


def idx2rc(idx: ArrayLike, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an index to a yx coordinate of the reference.

    Args:
      idx: int or numpy.np.ndarray [idx]
      shape: shape of the array
      idx: Union[int:
      List[int]:
      np.ndarray]:
      shape: Tuple[int:
      int]:

    Returns:
      numpy.np.ndarray [[y], [x]] ([row][column])

    """
    idx = np.atleast_1d(idx)
    idx = np.array(idx).astype(int)
    return np.unravel_index(idx, shape)  # type: ignore[return-value]


def rc2idx(row_col: List[Tuple[int, int]], shape: tuple) -> np.ndarray:
    """
    Converts row-column indexing into linear indexing.

    Args:
      row_col: List of List of ints, [[row, col], [row, col], ...] rows and columns to be converted to linear index
      shape: tuple, (int, int), shape of array being indexed

    Returns:
      np.ndarray, returns array of linear index
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
    """Two dimensional polynomial fitting by least squares.

    Fits the functional form f(x,y) = z.

    Args:
      x: np.ndarray: x values for the fit
      y: np.ndarray:  y values for the fit
      z: np.ndarray:  z values for the fit
      kx: Optional[int]: Polynomial order in x. (Default value = 3)
      ky: Optional[int]: Polynomial order in y. (Default value = 3)
      order: If int, coefficients up to a maximum of kx+ky <= order are considered. (Default value = None)

    Returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] solution, residuals, rank, singular values

    See: https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python

    See also: np.linalg.leastsq

    Notes: Resulting fit can be plotted with:
    >>> np.polynomial.polynomial.polygrid2d(x, y, solution.reshape((kx+1, ky+1)))
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
    Generate the parameter (low/high frange) for the QDMpy simulation.
    Args:
        projected_shifts: shift of the peaks in GHz
        width: width of the peaks in GHz
        contrast: contrast of the peaks

    Returns:
        parameter: parameter for the QDMpy simulation
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
    """Calculates the models for all dec/inc combinations of the current applied field.

    This function is meant to calculate the "safe" field margins for the current applied field.

    Returns:
        np.ndarray: The calculated models for all dec/inc combinations for NV axes > 0 of the current applied field.
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


@guvectorize(["float64[:], float64[:]"], "(n) -> ()", target="parallel")
def rms(data, ret):
    """Calculate the root mean square of a data set.

    Args:
      data: data set

    Returns:
      root mean square

    """
    ret[0] = np.sqrt(np.mean(np.square(data)))


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
    print(millify(0.001, 10))


if __name__ == "__main__":
    main()