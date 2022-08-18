import logging
import os
import tomli
import matplotlib.image as mpimg
import numpy as np
import sys
import pyqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = logging.getLogger(f'pyqdm.{__name__}')


def idx2rc(idx, shape):
    """
    Convert an index to a yx coordinate of the reference.

    :param idx: int or numpy.ndarray [idx]
    :param shape: shape of the array

    :return: numpy.ndarray [[y], [x]] ([row][column])
    """
    idx = np.array(idx).astype(int)
    return np.unravel_index(idx, shape)


def rc2idx(rc, shape):
    """
    Convert the xy coordinates to the index of the data.
    :param rc: numpy.ndarray [[row], [col]]
    :param shape: shape of the array
    :return: numpy.ndarray [idx]
    """
    rc = np.array(rc).astype(int)
    return np.ravel_multi_index(rc, shape)


def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    found at :https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx + 1, ky + 1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x ** i * y ** j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)


def load_config(config_file='config.ini'):
    """
    Loads the config file.

    :param config_file: str, default is 'config.ini'
        name to the config file
    :return: dict
        config file
    """
    with open(os.path.join(pyqdm.projectdir, config_file)) as fileObj:
        content = fileObj.read()
        pyqdm_config = tomli.loads(content)
        return pyqdm_config


def set_path(path, config, default):
    """
    Sets the path to the data.

    :param path:
    :param config:
    :param default:
    :return:
    """
    if path in config['default_paths']:
        p = config['default_paths'][path]
    else:
        p = default
    logging.debug(f'Setting {path} to {p}')
    return p


def has_csv(lst):
    """
    Checks if a list of files contains a csv file.

    :param lst: list of str
    :return: bool
    """
    return any(('.csv' in s for s in lst))


def get_image_file(lst):
    """
    Returns the first image file in the list.

    :param lst: list of str
        list of files to load the image from
    :return: name of the image file
    """
    if has_csv(lst):
        return [s for s in lst if '.csv' in s][0]
    else:
        return [s for s in lst if '.jpg' in s][0]


def get_image(folder, lst):
    """
    Loads an image from a list of files.

    :param folder: str
        folder to load the image from
    :param lst: list of str
        list of files to load the image from
    :return: np.array
        image
    """
    return np.loadtxt(os.path.join(folder, get_image_file(lst))) if has_csv(lst) \
        else mpimg.imread(os.path.join(folder, get_image_file(lst)))


def double_norm(data, axis):
    """
    Normalizes data from 0 to 1.
    :param data: np.array
        data to normalize
    :param axis: int
        axis to normalize

    :return: np.array
        normalized data
    """
    mn = np.expand_dims(np.min(data, axis=axis), data.ndim - 1)
    data -= mn
    mx = np.expand_dims(np.max(data, axis=axis), data.ndim - 1)
    return data / mx
