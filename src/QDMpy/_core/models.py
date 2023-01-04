from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import float64, guvectorize
from numpy.typing import NDArray

import QDMpy


@guvectorize(
    [(float64[:], float64, float64, float64, float64[:])],
    "(n),(),(),()->(n)",
    forceobj=True,
)
def lorentzian_peak(f, center, width, contrast, model):
    """Lorentzian peak

    Calculates a single lorentzian peak.

    Args:
        f : array_like
            Frequency axis
        center : float
            Center frequency of the peak
        width : float
            Width of the peak
        contrast : float
            Contrast of the peak
        model : array_like
            Model array to be filled with the model

    Notes:
        The model is calculated as numpy ufunc (using numba), so it can be used in a vectorized way.
        Meaning that the parameters can be arrays but need to have the same shape.

    """
    sq_width = np.square(width)
    delta = f - center
    model[:] = contrast * sq_width / (delta**2 + sq_width)


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr14n(x, parameter, model):
    """ESR14N model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = contrast
            parameter[5] = offset

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.002158

    center, width, c0, c1, c2, offset = parameter
    center0 = center + AHYP
    center1 = center
    center2 = center - AHYP

    model[:] = (
        1
        + offset
        - np.sum(
            lorentzian_peak(x, [center0, center1, center2], width, [c0, c1, c2]), axis=0
        )
    )


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr14n_folded(x, parameter, model):
    """ESR14N model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = contrast
            parameter[5] = offset

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.002158

    center, width, c0, c1, c2, offset = parameter
    center0 = center + AHYP
    center1 = center
    center2 = center - AHYP

    model[:] = -np.sum(
        lorentzian_peak(x, [center0, center1, center2], width, [c0, c1, c2]), axis=0
    )
    model[:] -= np.sum(
        lorentzian_peak(x + 2 * AHYP, [center0, center1, center2], width, [c0, c1, c2]),
        axis=0,
    )
    model[:] += 1 + offset


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr15n(x, parameter, model):
    """ESR14N model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = offset

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.0015

    center, width, c0, c1, offset = parameter
    center0 = center + AHYP
    center1 = center - AHYP

    model[:] = (
        1
        + offset
        - np.sum(lorentzian_peak(x, [center0, center1], width, [c0, c1]), axis=0)
    )


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr15n_folded(x, parameter, model):
    """ESR14N model, with folded spectrum

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = offset

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.0015

    center, width, c0, c1, offset = parameter
    center0 = center - AHYP
    center1 = center + AHYP

    model[:] = -np.sum(lorentzian_peak(x, [center0, center1], width, [c0, c1]), axis=0)
    model[:] -= np.sum(
        lorentzian_peak(x + 2 * AHYP, [center0, center1], width, [c0, c1]), axis=0
    )
    model[:] += 1 + offset


@guvectorize(
    [(float64[:], float64[:], float64[:])],
    "(n),(m)->(n)",
    forceobj=True,
    target="parallel",
)
def esrsingle(x, param, model):
    """Single Lorentzian model

    Args:
        x (np.ndarray): frequency values
        param (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = offset

    Returns:
        np.ndarray: y values
    """
    model[:] = (
        1 + param[3] - np.sum(lorentzian_peak(x, param[0], param[1], param[2]), axis=0)
    )


def esrsingle_(x, parameter):
    """ESRSINGLE model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = offset

    Returns:
        np.ndarray: y values
    """
    out = []
    parameter = np.atleast_2d(parameter)
    for i in range(parameter.shape[0]):
        p = parameter[i]
        width_squared = p[1] * p[1]

        aux1 = x - p[0]
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        out.append(1 + p[3] - dip1)
    return np.array(out)


IMPLEMENTED = {
    "GAUSS1D": {
        "func_name": "GAUSS1D",
        "n_peaks": 1,
        "func": None,
        "params": ["contrast", "center", "width", "offset"],
        "model_id": 0,
        "name": "GAUSS_MISC",
    },
    "ESR14N": {
        "func_name": "ESR14N",
        "n_peaks": 3,
        "func": esr14n,
        "params": ["center", "width", "contrast", "contrast", "contrast", "offset"],
        "model_id": 13,
        "name": "N14",
    },
    "ESR15N": {
        "func_name": "ESR15N",
        "n_peaks": 2,
        "func": esr15n,
        "params": ["center", "width", "contrast", "contrast", "offset"],
        "model_id": 14,
        "name": "N15",
    },
    "ESRSINGLE": {
        "func_name": "ESRSINGLE",
        "n_peaks": 1,
        "func": esrsingle,
        "params": ["center", "width", "contrast", "offset"],
        "model_id": 15,
        "name": "SINGLE_MISC.",
    },
}
PEAK_TO_TYPE = {1: "ESRSINGLE", 2: "ESR15N", 3: "ESR14N"}


def full_model(model, freqs, parameters):
    """Full model

    Args:
        model (str): Model name
        freqs (np.ndarray): Frequencies
        parameters (np.ndarray): Parameters

    Returns:
        np.ndarray: Model
    """
    if model == "ESR14N":
        model = esr14n
    elif model == "ESR15N":
        model = esr15n
    elif model == "ESRSINGLE":
        model = esrsingle
    else:
        raise ValueError("Model not implemented")

    low_f_data = model(freqs[: len(freqs) // 2], parameters)
    high_f_data = model(freqs[len(freqs) // 2 :], parameters)

    return np.concatenate((low_f_data, high_f_data), axis=-1)


def guess_model(data: NDArray, check: bool = False) -> Tuple[int, bool, Any]:
    """Guess the diamond type based on the number of peaks.

    :return: diamond_type (int)

    Args:
        data (np.ndarray): data


    Returns:

    """
    from scipy.signal import find_peaks

    indices = []

    # Find the indices of the peaks
    for p, f in np.ndindex(*data.shape[:2]):
        peaks = find_peaks(
            -data[p, f], prominence=QDMpy.SETTINGS["model"]["find_peaks"]["prominence"]
        )
        indices.append(peaks[0])
        if check:
            (l,) = plt.plot(data[p, f])
            plt.plot(
                peaks[0],
                data[p, f][peaks[0]],
                "x",
                color=l.get_color(),
                label=f"({p},{f}): {len(indices[n])}",
            )

    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))

    doubt = np.std([len(idx) for idx in indices]) != 0

    if check:
        plt.show()
        plt.legend()

    return n_peaks, doubt, peaks
