from typing import Tuple, Union

import numpy as np
import pint
from numba import vectorize, float64
from numpy.typing import NDArray

import QDMpy

UREG = pint.UnitRegistry()

_EPSILON = 1e-30
_VECTOR_OR_NONE = Union[Tuple[float, float, float], None]


def b111_to_xyz(
    map: NDArray,
    pixel_size: float = 1.175e-06,
    rotation_angle: float = 0,
    direction_vector: _VECTOR_OR_NONE = None,
) -> NDArray:
    """
    Convert a map measured along the direction u to a Bz map of the sample.

    Args:
        map: 2D array
            The map to be converted
        pixel_size: float
            The size of the pixel in the map in m
        rotation_angle: float
            The rotation of the diamond lattice axes around z-axis
        direction_vector: 3-tuple of floats
            The direction of the u vector in the lab frame
            (i.e. the orientation of the 111 axis with respect to the
            magnetic bias field)

    Returns:
        2D array
            The converted map
    """

    unit_vector = get_unit_vector(rotation_angle, direction_vector)

    ypix, xpix = map.shape
    step_size = 1 / pixel_size

    # these freq. coordinates match the fft algorithm
    x_steps = np.concatenate([np.arange(0, xpix / 2, 1), np.arange(-xpix / 2, 0, 1)])
    fx = x_steps * step_size / xpix
    y_steps = np.concatenate([np.arange(0, ypix / 2, 1), np.arange(-ypix / 2, 0, 1)])
    fy = y_steps * step_size / ypix

    fgrid_x, fgrid_y = np.meshgrid(fx + _EPSILON, fy + _EPSILON)

    kx = 2 * np.pi * fgrid_x
    ky = 2 * np.pi * fgrid_y
    k = np.sqrt(kx ** 2 + ky ** 2)

    e = np.fft.fft2(map)

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
    """ converts a magnetic field along as single NV-axis into a resonance freq.

    Args:
        b: magnetic field in T to be converted

    Returns:
        resonance frequency in GHz

    >>> _b2shift([1, 0.1])
    >>> array([28.024e9 ,  2.8024e9])
    """
    return b * 28.024e9  # Hz/T


@vectorize([float64(float64)])
def _shift2b(shift):
    """ converts a resonance freq. shift into a magnetic field along as single NV-axis.

    Args:
        shift: resonance frequency shift in GHz to be converted

    Returns:
        magnetic field in T

    >>> _shift2b([1, 0.1])
    >>> array([0.0356837 , 0.00356837])
    """
    return shift / 28.024e9  # T


def b2shift(b: float, in_unit: str = 'T', out_unit: str = 'GHz') -> float:
    """ converts a magnetic field along as single NV-axis into a resonance freq.
    shift for that axis.

    Args:
        b: field to be converted
        in_unit: unit of b defaults to 'T'
        out_unit: unit of the freq defaults to 'GHz'

    Returns:
        freq. shift in the unit specified by out_unit

    >>> b2shift(1, in_unit='nT', out_unit='Hz')
    >>> 28.024000000000004
    """
    b = convert_to(b, in_unit, 'T')
    shift = _b2shift(b)
    return convert_to(shift, 'Hz', out_unit)
def convert_to(value, in_unit, out_unit):
    """ converts a value from in_unit to out_unit

    Args:
        value: value to be converted
        in_unit: unit of value
        out_unit: unit of the converted value

    Returns:
        converted value

    >>> convert_to(1, 'nT', 'T')
    >>> 1e-9
    """
    value = UREG.Quantity(value, in_unit)
    value = value.to(out_unit, 'Gau').magnitude
    return value

def freq2b(freq: float, in_unit: str = 'Hz', out_unit: str = 'T') -> float:
    """ converts a resonance frequency into a magnetic field along as single NV-axis.

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
    ZFS = 2.87e9  # zero-field splitting in Hz
    # calculate the shift by subtracting the ZFS
    freq = convert_to(freq, in_unit, 'Hz')
    shift = freq - ZFS

    return shift2b(shift, 'Hz', out_unit)


def shift2b(shift: float, in_unit: str = 'GHz', out_unit: str = 'T') -> float:
    """ converts a resonance frequency shift into a magnetic field along as single NV-axis.

    Args:
        shift: shift to be converted
        in_unit: unit of shift defaults to 'GHz'
        out_unit: unit of the field defaults to 'T'

    Returns:
        field in the unit specified by out_unit

    >>> shift2b(1, in_unit='MHz', out_unit='mT')
    >>> 0.03571428571428571
    """
    shift = convert_to(shift, in_unit, 'Hz')
    b = _shift2b(shift) # in T
    return convert_to(b, 'T', out_unit)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    map = loadmat(
        "/Users/mike/Dropbox/data/QDMlab_example_data/data/viscosity/ALH84001-FOV1_VISC_20-20/4x4Binned/B111dataToPlot.mat"
    )["B111ferro"]
    out = toBz(map, pixel_size=4.70e-6)
    print(out)
    plt.imshow(out[2])
