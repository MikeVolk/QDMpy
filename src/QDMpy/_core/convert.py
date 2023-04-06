"""
This module contains various functions to convert magnetic field vectors to
resonance frequencies and vice versa. It also performs conversions of maps and
signals. The following functions are implemented:
    
    b111_to_bxyz(bmap: numpy.ndarray, pixel_size: float = 1.175e-06,
    rotation_angle: float = 0, direction_vector: Union[Tuple[float, float,
    float], None] = None) -> numpy.ndarray:
        This function converts a map measured along direction "u" to a Bz map of
        the sample. It takes a numpy NDArray for the map, as well as pixel size,
        rotation angle, and direction vector defaulting to (180°,35.3°).
    \n    get_unit_vector(rotation_angle: float, direction_vector:
    Union[Tuple[float, float, float], None] = None) -> numpy.ndarray:
        This function takes the rotation angle and direction vector to output a
        normalized unit vector.
        
    b2shift(b: float, in_unit: str = "T", out_unit: str = "GHz") -> float:
        This function converts a magnetic field along as single NV-axis into a
        resonance freq. shift for that axis. It takes the field, and inputs for
        the field unit (in_unit) and output resonance frequency unit (out_unit).
        
    convert_to_unit(value, in_unit, out_unit):
        This function converts a value from in_unit to out_unit. It takes the
        value and input/output units in string format.
        
    freq2b(freq: float, in_unit: str = "Hz", out_unit: str = "T") -> float:
        This function converts a resonance frequency into a magnetic field along
        as single NV-axis. It takes the frequency and inputs for the frequency
        unit (in_unit) and output field unit (out_unit).
    
    shift2b(shift: float, in_unit: str = "GHz", out_unit: str = "T") -> float:
        This function converts a resonance frequency shift into a magnetic field
        along as single NV-axis. It takes the frequency shift and inputs for the
        frequency shift unit (in_unit) and output field unit (out_unit).
    
    project(a: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray):
        This function projects a vector a onto b and returns the result in c. It
        takes numpy NDArrays for vectors a and b, and an empty numpy NDArray, c,
        to store the result.
"""

from typing import Tuple, Union


import numpy as np
import pint
from numba import float64, guvectorize, vectorize
from numpy.typing import NDArray

import QDMpy

UREG = pint.UnitRegistry()

_VECTOR_OR_NONE = Union[Tuple[float, float, float], None]

""" convert.py contains a number of functions used to convert, signals, maps and
frequencies. 
"""


def b111_to_bxyz(
    bmap: NDArray,
    pixel_size: float = 1.175e-06,
    rotation_angle: float = 0,
    direction_vector: _VECTOR_OR_NONE = None,
) -> NDArray:
    """
    Convert a map measured along the direction u to a Bz map of the sample.

    Args:
        bmap: 2D array
            The map to be converted
        pixel_size: float
            The size of the pixel in the map in m
        rotation_angle: float
            The rotation of the diamond lattice axes around z-axis
        direction_vector:
            The direction of the 111 axis with respect to the QDM measurement
            frame. Default (180°, 35.3°)


    Returns:
        2D array
            The converted map
    """

    unit_vector = get_unit_vector(rotation_angle, direction_vector)

    ypix, xpix = bmap.shape
    step_size = 1 / pixel_size

    # these freq. coordinates match the fft algorithm
    fx = np.concatenate([np.arange(0, xpix / 2, 1), np.arange(-xpix / 2, 0, 1)]) * step_size / xpix
    fy = np.concatenate([np.arange(0, ypix / 2, 1), np.arange(-ypix / 2, 0, 1)]) * step_size / ypix

    fgrid_x, fgrid_y = np.meshgrid(fx + 1e-30, fy + 1e-30)

    kx = 2 * np.pi * fgrid_x
    ky = 2 * np.pi * fgrid_y
    k = np.sqrt(kx**2 + ky**2)

    e = np.fft.fft2(bmap)

    x_filter = -1j * kx / k
    y_filter = -1j * ky / k
    z_filter = k / (
        unit_vector[2] * k - unit_vector[1] * 1j * ky - unit_vector[0] * 1j * kx
    )  # calculate the filter frequency response associated with the x component

    map_x = np.fft.ifft2(e * x_filter)
    map_y = np.fft.ifft2(e * y_filter)
    map_z = np.fft.ifft2(e * z_filter)
    return np.stack([map_x.real, map_y.real, map_z.real])


def get_unit_vector(rotation_angle: float, direction_vector: _VECTOR_OR_NONE = None) -> NDArray:
    """
    Get the unit vector of the sample in the lab frame.

    Args:
        rotation_angle: float
            The rotation of the diamond lattice axes around z-axis
        direction_vector:
            The direction of the 111 axis with respect to the QDM measurement frame. Default (180°, 35.3°)

    Returns:
        A normalized unit vector in the instrument frame
    """

    if direction_vector is None:
        direction_vector = np.array([0, np.sqrt(2 / 3), np.sqrt(1 / 3)])

    QDMpy.LOG.info(
        f"Getting unit vector from rotation angle {rotation_angle} along direction vector {direction_vector}"
    )

    alpha = np.rad2deg(rotation_angle)
    rotation_matrix = [
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1],
    ]

    unit_vector = np.matmul(rotation_matrix, direction_vector)
    unit_vector /= np.linalg.norm(unit_vector)
    return unit_vector


@vectorize([float64(float64)])
def _b2shift(b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """converts a magnetic field in Tesla along an NV-axis into a resonance freq.

    Args:
        b: magnetic field in T to be converted

    Returns:
        resonance frequency in GHz

    Note:
        This function is vectorized and can be used with numpy arrays.

    >>> _b2shift([1, 0.1])
    >>> np.array([28.024e9 ,  2.8024e9])
    """
    return b * 28.024e9  # Hz/T


@vectorize([float64(float64)])
def _shift2b(shift: Union[float, list, NDArray]):
    """converts a resonance freq. shift into a magnetic field in Tesla along as single NV-axis.

    Args:
        shift: resonance frequency shift in GHz to be converted

    Returns:
        magnetic field in T

    Note:
        This function is vectorized and can be used with numpy arrays.

    >>> _shift2b([1, 0.1])
    >>> np.array([0.0356837 , 0.00356837])
    """
    return shift / 28.024e9  # T


def b2shift(b: float, in_unit: str = "T", out_unit: str = "GHz") -> float:
    """converts a magnetic field along as single NV-axis into a resonance freq.
    shift for that axis.

    Args:
        b: field to be converted
        in_unit: unit of b defaults to 'T'
        out_unit: unit of the freq defaults to 'GHz'

    Returns:
        freq. shift in the unit specified by out_unit

    Note:
        This function is vectorized and can be used with numpy arrays.

    >>> b2shift(1, in_unit='nT', out_unit='Hz')
    >>> 28.024000000000004
    """
    b = convert_to_unit(b, in_unit, "T")
    shift = _b2shift(b)
    return convert_to_unit(shift, "Hz", out_unit)


def convert_to_unit(value, in_unit, out_unit):
    """converts a value from in_unit to out_unit

    Args:
        value: value to be converted
        in_unit: unit of value
        out_unit: unit of the converted value

    Returns:
        converted value

    >>> convert_to_unit(1, 'nT', 'T')
    >>> 1e-9
    """
    value = UREG.Quantity(value, in_unit)
    value = value.to(out_unit, "Gau").magnitude
    return value


def freq2b(freq: float, in_unit: str = "Hz", out_unit: str = "T") -> float:
    """converts a resonance frequency into a magnetic field along as single NV-axis.

    Args:
        freq: frequency to be converted
        in_unit: unit of freq defaults to 'GHz'
        out_unit: unit of the field defaults to 'T'

    Returns:
        field in the unit specified by out_unit

    See Also:
        b2shift, shift2b
    >>> # a +/- 10MHz shift in the resonance corresponds to a +/- 356 nT field along the NV-axis
    >>> freq2b(2.86, in_unit='MHz', out_unit='nT')
    >>> -356.8369968598426 # negative because of the sign convention
    >>> freq2b(2.88, in_unit='MHz', out_unit='nT')
    >>> 356.8369968598426
    """
    zfs = 2.87e9  # zero-field splitting in Hz
    # calculate the shift by subtracting the zfs
    freq = convert_to_unit(freq, in_unit, "Hz")
    shift = freq - zfs

    return shift2b(shift, "Hz", out_unit)


def shift2b(shift: float, in_unit: str = "GHz", out_unit: str = "T") -> float:
    """converts a resonance frequency shift into a magnetic field along as single NV-axis.

    Args:
        shift: shift to be converted
        in_unit: unit of shift defaults to 'GHz'
        out_unit: unit of the field defaults to 'T'

    Returns:
        field in the unit specified by out_unit

    >>> shift2b(1, in_unit='MHz', out_unit='mT')
    >>> 0.03571428571428571
    """
    shift = convert_to_unit(shift, in_unit, "Hz")
    b = _shift2b(shift)  # in T
    return convert_to_unit(b, "T", out_unit)


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->()", forceobj=True)
def project(a: NDArray, b: NDArray, c: NDArray):
    """projects vector a onto b and returns the result in c

    Args:
        a: vector(s) to be projected
        b: vector(s) to project onto
        c: vector to store the result

    Notes:
        1. a and b must have the same shape
        2. you can either pass a single vector or a list of vectors but only for either a or b
           This means that you can either project a single vector onto a list of vectors or
           project a list of vectors onto a single vector
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([1, 1, 1])
    >>> project(v1, v2)
    >>> np.array([1.5, 1.5, 1.5])
    """
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    c[0] = np.dot(a, b) / np.linalg.norm(b)
