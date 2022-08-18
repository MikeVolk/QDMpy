import numpy as np
from pyqdm.utils import double_norm
FIT_PARAMETER = {'GAUSS_1D': ['contrast', 'center', 'width', 'offset'],
                 'ESR14N': ['center', 'width', 'contrast', 'contrast', 'contrast', 'offset'],
                 'ESR15N': ['center', 'width', 'contrast', 'contrast', 'offset'],
                 'ESRSINGLE': ['center', 'width', 'contrast', 'offset']}


def guess_contrast(data):
    """
    Guess the contrast of a ODMR data.

    :param data: np.array
        data to guess the contrast from
    :return: np.array
        contrast of the data
    """
    mx = np.nanmax(data, axis=-1)
    mn = np.nanmin(data, axis=-1)
    amp = np.abs(((mx - mn) / mx))
    return amp * 0.9

def guess_center(data, freq):
    """
    Guess the center frequency of ODMR data.

    :param data: np.array
        data to guess the center frequency from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        center frequency of the data
    """
    # center frequency
    center_lf = guess_center_freq_single(data[:, 0], freq[0])
    center_rf = guess_center_freq_single(data[:, 1], freq[1])
    center = np.stack([center_lf, center_rf], axis=0)
    center = np.swapaxes(center, 0, 1)
    assert np.all(center[:, 0] == center_lf)
    assert np.all(center[:, 1] == center_rf)

    return center


def guess_width(data, freq):
    """
    Guess the width of a ODMR resonance peaks.

    :param data: np.array
        data to guess the width from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        width of the data
    """
    # center frequency
    width_lf = guess_width_single(data[:, 0], freq[0])
    width_rf = guess_width_single(data[:, 1], freq[1])
    width = np.stack([width_lf, width_rf], axis=1)

    assert np.all(width[:, 0] == width_lf)
    assert np.all(width[:, 1] == width_rf)

    return width / 6


def guess_width_single(data, freq):
    """
    Guess the width of a single frequency range.

    :param data: np.array
        data to guess the width from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        width of the data
    """
    data = np.cumsum(data - 1, axis=-1)
    data -= np.expand_dims(np.min(data, axis=-1), axis=2)
    data /= np.expand_dims(np.max(data, axis=-1), axis=2)
    lidx = np.argmin(np.abs(data - 0.25), axis=-1)
    ridx = np.argmin(np.abs(data - 0.75), axis=-1)
    return (freq[lidx] - freq[ridx])


def guess_center_freq_single(data, freq):
    """
    Guess the center frequency of a single frequency range.

    :param data: np.array
        data to guess the center frequency from
    :param freq: np.array
        frequency range of the data
    :return: np.array
        center frequency of the data
    """
    data = np.cumsum(data - 1, axis=-1)
    data -= np.expand_dims(np.min(data, axis=-1), axis=2)
    data /= np.expand_dims(np.max(data, axis=-1), axis=2)
    idx = np.argmin(np.abs(data - 0.5), axis=-1)
    return freq[idx]

