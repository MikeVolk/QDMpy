import numpy as np
import logging
from typing import (
    Tuple,
    List,
    Union,
    Optional,
    Dict,
    Any,
    Callable,
    TypeVar,
    Generic,
    Type,
    cast,
)
from numpy.typing import NDArray
import QDMpy
LOG = logging.getLogger(__name__)

_EPSILON = 1e-30


def upward_continue(map, distance, pixel_size: float = 1.175e-06, oversample: int = 2):
    """
    Upward continues a map.

    Args:
        map: 2D array
            The map to be continued
        distance: float
            The distance to upward continue the map in m
        pixel_size: float
            The size of the pixel in the map in m

    Returns:
        2D array
            The continued map
    """
    ypix, xpix = map.shape
    step_size = 1 / pixel_size
    new_x, new_y = xpix * oversample, ypix * oversample

    # these freq. coordinates match the fft algorithm
    x_steps = np.concatenate([np.arange(0, new_x / 2, 1), np.arange(-new_x / 2, 0, 1)])
    fx = x_steps * step_size / new_x
    y_steps = np.concatenate([np.arange(0, new_y / 2, 1), np.arange(-new_y / 2, 0, 1)])
    fy = y_steps * step_size / new_y

    fgrid_x, fgrid_y = np.meshgrid(fx + _EPSILON, fy + _EPSILON)

    kx = 2 * np.pi * fgrid_x
    ky = 2 * np.pi * fgrid_y
    k = np.sqrt(kx**2 + ky**2)

    # Calculate the filter frequency response associated with the x component
    x_filter = np.exp(-distance * k)

    # Compute FFT of the field map
    fft_map = np.fft.fft2(map, s=(new_y, new_x))

    # Calculate x component
    B_out = np.fft.ifft2(fft_map * x_filter)
    LOG.debug("Upward continued map by %s m", distance)
    # Crop matrices to get rid of zero padding
    return B_out[:ypix, :xpix].real
