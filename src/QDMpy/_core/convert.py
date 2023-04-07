from typing import Tuple, Union

import numpy as np
import pint
from numba import vectorize, float64
from numpy.typing import NDArray

import QDMpy

UREG = pint.UnitRegistry()

_VECTOR_OR_NONE = Union[Tuple[float, float, float], None]


def toBxyz(
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

    Returns:
        2D array
            The converted map
    """

    unit_vector = get_unit_vector(rotation_angle, direction_vector)

    ypix, xpix = map.shape
    step_size = 1 / pixel_size

    # these freq. coordinates match the fft algorithm
    fx = np.concatenate([np.arange(0, xpix / 2, 1), np.arange(-xpix / 2, 0, 1)]) * step_size / xpix
    fy = np.concatenate([np.arange(0, ypix / 2, 1), np.arange(-ypix / 2, 0, 1)]) * step_size / ypix

    fgrid_x, fgrid_y = np.meshgrid(fx + 1e-30, fy + 1e-30)

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
        "Getting unit vector from rotation angle {} along direction vector {}".format(
            rotation_angle, direction_vector
        )
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

#         function [Bz] = QDMBzFromBu(Bu, fs, u)
# %[Bz] = QDMBzFromBu(Bu, fs, u)
# %   Retrieves the z component of the magnetic field from a QDM map of the u component. It performs
# %   this operation in the frequency domain.
# %
# %   Code by Eduardo A. Lima - (c) 2017
#
# %   ----------------------------------------------------------------------------------------------
# %   Bu     -> u component of the magnetic field measured in a regular planar grid
# %   fs     -> Sampling frequency in 1/m
# %   u      -> unit vector representing the field component measured
# %   ----------------------------------------------------------------------------------------------
#
#
# SHOWGRAPHS = 0; % Set this value to 1 to show frequency response plots, or to 0 otherwise.
#
# [SIZEx, SIZEy] = size(Bu);
# N1 = SIZEx;
# N2 = SIZEy;
#
# f1 = [0:N1 / 2 - 1, -(N1 / 2):-1] * fs / N1; %these freq. coordinates match the fft algorithm
# f2 = [0:N2 / 2 - 1, -(N2 / 2):-1] * fs / N2;
#
# [F2, F1] = meshgrid(f2+1e-30, f1+1e-30);
#
# ky = 2 * pi * F1;
# kx = 2 * pi * F2;
#
# k = sqrt(kx.^2+ky.^2);
#
#
# etz = k ./ (u(3) * k - u(2) * 1i * ky - u(1) * 1i * kx); % calculate the filter frequency response associated with the x component
#
#
# e = fft2(Bu, N1, N2);
#
# Bz = ifft2(e.*etz, N1, N2, 'symmetric');
# %Bz=real(ifft2(e.*etz));
