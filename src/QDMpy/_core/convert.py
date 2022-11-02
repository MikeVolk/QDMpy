import numpy as np
from typing import Tuple, List, Union, Optional, Dict, Any, Callable, TypeVar, Generic, Type, cast
from numpy.typing import NDArray
import QDMpy

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
    k = np.sqrt(kx**2 + ky**2)

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


def get_unit_vector(rotation_angle:float, direction_vector:_VECTOR_OR_NONE=None) -> NDArray:
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
