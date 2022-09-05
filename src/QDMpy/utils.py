import logging
import os
import shutil
import sys
from typing import Union, Tuple, List, Optional

import matplotlib.image as mpimg
import numpy as np
import tomli

from QDMpy import CONFIG_FILE, CONFIG_INI, CONFIG_PATH

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

    return f"{n / 10**(3 * millidx):.{sign}f}{MILLNAMES[millidx + 3]}"


def idx2rc(idx: Union[int, List[int], np.ndarray], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an index to a yx coordinate of the reference.

    Args:
      idx: int or numpy.ndarray [idx]
      shape: shape of the array
      idx: Union[int:
      List[int]:
      np.ndarray]:
      shape: Tuple[int:
      int]:

    Returns:
      numpy.ndarray [[y], [x]] ([row][column])

    """
    idx = np.atleast_1d(idx)
    idx = np.array(idx).astype(int)
    return np.unravel_index(idx, shape)  # type: ignore[return-value]


def rc2idx(rc: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert the xy coordinates to the index of the data.

    Args:
      rc: numpy.ndarray [[row], [col]]
      shape: shape of the array
      rc: np.ndarray:
      shape: Tuple[int:
      int]:

    Returns:
      numpy.ndarray [idx]

    """
    rc = np.array(rc).astype(int)
    return np.ravel_multi_index(rc, shape)  # type: ignore[no-any-return, call-overload]


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


def load_config() -> dict:
    """Loads the config file.

    :return: dict
        config file

    Args:

    Returns:

    """
    with open(CONFIG_FILE, "rb") as fileObj:
        return tomli.load(fileObj)


def make_configfile(reset: bool = False) -> None:
    """Creates the config file if it does not exist.

    Args:
      reset: bool:  (Default value = False)

    Returns:

    """
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists() or reset:
        LOG.info(f"Copying default QDMpy 'config.ini' file to {CONFIG_FILE}")
        shutil.copy2(CONFIG_INI, CONFIG_FILE)


def has_csv(lst: list) -> bool:
    """Checks if a list of files contains a csv file.

    Args:
      lst: list of str
      lst: list:

    Returns:
      bool

    """
    return any(".csv" in s for s in lst)


def get_image_file(lst: list) -> str:
    """Returns the first image file in the list.

    Args:
      lst: list of str
    list of files to load the image from
      lst: list:

    Returns:
      name of the image file

    """
    if has_csv(lst):
        lst = [s for s in lst if ".csv" in s]
    else:
        lst = [s for s in lst if ".jpg" in s]

    return str(lst[0])


def get_image(folder: str, lst: List[str]) -> np.ndarray:
    """Loads an image from a list of files.

    Args:
        folder: folder of the image
        lst: list of files to load the image from
    Returns: loaded image from either csv or jpg file if csv is not available

    """
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


def main() -> None:
    """Main function."""
    print(millify(0.001, 10))


if __name__ == "__main__":
    main()
