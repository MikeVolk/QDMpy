import numpy as np
from numba import njit


FIT_PARAMETER = {'GAUSS_1D': ['contrast', 'center', 'width', 'offset'],
                 'ESR14N': ['center', 'width', 'contrast', 'contrast', 'contrast', 'offset'],
                 'ESR15N': ['center', 'width', 'contrast', 'contrast', 'offset'],
                 'ESRSINGLE': ['center', 'width', 'contrast', 'offset']}



def guess_contrast(data):
    mx = np.nanmax(data, axis=-1)
    mn = np.nanmin(data, axis=-1)
    amp = np.abs(((mx - mn) / mx))
    return amp * 0.9

def guess_center(data, freq):
    # center frequency
    center_lf = guess_center_freq_single(data[:,0], freq[0])
    center_rf = guess_center_freq_single(data[:,1], freq[1])
    center = np.stack([center_lf, center_rf], axis=0)
    center = np.swapaxes(center, 0, 1)
    assert np.all(center[:, 0] == center_lf)
    assert np.all(center[:, 1] == center_rf)

    return center

def guess_width(data, freq):
    # center frequency
    width_lf = guess_width_single(data[:, 0], freq[0])
    width_rf = guess_width_single(data[:, 1], freq[1])
    width = np.stack([width_lf, width_rf], axis=1)

    assert np.all(width[:, 0] == width_lf)
    assert np.all(width[:, 1] == width_rf)

    return width/6

def guess_width_single(data, freq):
    data = np.cumsum(data-1, axis = -1)
    data -= np.expand_dims(np.min(data, axis=-1), axis=2)
    data /=  np.expand_dims(np.max(data, axis=-1), axis= 2)
    lidx = np.argmin(np.abs(data-0.25), axis=-1)
    ridx = np.argmin(np.abs(data-0.75), axis=-1)
    width = (freq[lidx] - freq[ridx])
    return width

def guess_center_freq_single(data, freq):
    data = np.cumsum(data-1, axis = -1)
    data -= np.expand_dims(np.min(data, axis=-1), axis=2)
    data /=  np.expand_dims(np.max(data, axis=-1), axis= 2)
    idx = np.argmin(np.abs(data-0.5), axis=-1)
    center = freq[idx]
    return center

def guess_center_freq_single_v0(data, freq):
    # sort all pixels for contrast
    idx = np.argsort(data, axis=-1)

    # swap axis needed to easily only take first l elements
    # without knowing the dimension of the data array
    idx = np.swapaxes(idx, 0, -1)

    # get the first l elements
    l = 10;  # lowest n values
    idx = idx[:l]

    # swap the axes back
    idx = np.swapaxes(idx, 0, -1)

    # get min and max of index
    mxidx = np.max(idx, axis=-1);
    mnidx = np.min(idx, axis=-1);
    # calculate the center freq
    # center = np.mean([mxidx, mnidx], axis=0)
    center = np.mean([freq[mxidx], freq[mnidx]], axis=0)
    return center


