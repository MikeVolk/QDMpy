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
def lorentzian_peak(f, center, width, contrast, out):
    """
    Calculates a single Lorentzian peak.

    Args:
        f : ndarray
            Frequency axis.
        center : float
            Center frequency of the peak.
        width : float
            Width of the peak.
        contrast : float
            Contrast of the peak.
        out : ndarray
            Model array to be filled with the model.

    Notes:
        The model is calculated as a numpy ufunc (using numba), so it can be used in a vectorized way.
        Meaning that the parameters can be arrays but need to have the same shape.

    Example:
        # Generate a frequency axis
        f = np.linspace(0, 10, 1000)

        # Define parameters for a single Lorentzian peak
        center = 5.0
        width = 1.0
        contrast = 0.5

        # Compute the model
        model = np.zeros_like(f)
        lorentzian_peak(f, center, width, contrast, model)

        # Plot the result
        import matplotlib.pyplot as plt
        plt.plot(f, model)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Signal (a.u.)")
        plt.show()

    """
    sq_width = np.square(width)
    delta = f - center
    out[:] = contrast * sq_width / (delta**2 + sq_width)


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr14n(x: NDArray, parameter: NDArray, model: NDArray) -> NDArray:
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
        - np.sum(lorentzian_peak(x, [center0, center1, center2], width, [c0, c1, c2]), axis=0)
    )


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr14n_folded(x: NDArray, parameter: NDArray, model: NDArray) -> NDArray:
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
            parameter[6] = secondary frequency

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.002158

    center, width, c0, c1, c2, offset, fsec = parameter
    center0 = center + AHYP
    center1 = center
    center2 = center - AHYP

    model[:] = -np.sum(lorentzian_peak(x, [center0, center1, center2], width, [c0, c1, c2]), axis=0)
    model[:] -= np.sum(
        lorentzian_peak(x + 2 * AHYP, [center0, center1, center2], width, [c0, c1, c2]),
        axis=0,
    )
    model[:] += 1 + offset


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr15n(x: NDArray, parameter: NDArray, model: NDArray) -> NDArray:
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

    model[:] = 1 + offset - np.sum(lorentzian_peak(x, [center0, center1], width, [c0, c1]), axis=0)


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr15n_folded(x: NDArray, parameter: NDArray, model: NDArray) -> NDArray:
    """ESR14N model, with folded spectrum

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = offset
            parameter[5] = secondary frequency

    Returns:
        np.ndarray: y values
    """
    AHYP = 0.0015

    center, width, c0, c1, offset, fsec = parameter
    center0 = center - AHYP
    center1 = center + AHYP

    model[:] = -np.sum(lorentzian_peak(x, [center0, center1], width, [c0, c1]), axis=0)
    model[:] -= np.sum(lorentzian_peak(x + 2 * AHYP, [center0, center1], width, [c0, c1]), axis=0)
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
    model[:] = 1 + param[3] - np.sum(lorentzian_peak(x, param[0], param[1], param[2]), axis=0)


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


def guess_model(data: np.ndarray, check: bool = False) -> Tuple[int, bool, Any]:
    """
    Guess the diamond type based on the number of peaks.

    Args:
        data (np.ndarray): A 2D numpy array containing the diamond data.
        check (bool, optional): If True, displays a plot of the diamond data with the peaks marked. Default is False.

    Returns:
        Tuple[int, bool, Any]: A tuple containing the number of peaks, a boolean indicating whether there is doubt in the model, and the indices of the peaks.

    Example:
        >>> data = np.random.rand(10, 10)
        >>> guess_model(data)
        (1, False, array([array([], dtype=int64), array([], dtype=int64), ..., array([], dtype=int64)]))

    This example generates a 10x10 numpy array containing random data and passes it to the guess_model() function. The function returns a tuple containing the number of peaks (which in this case is 1), a boolean indicating that there is no doubt in the model (since the standard deviation is 0), and an empty numpy array (since there are no peaks in the random data).
    """
    from scipy.signal import find_peaks

    # Compute the indices of the peaks for each row of the data array
    peak_indices = np.apply_along_axis(
        lambda row: find_peaks(
            -row, prominence=QDMpy.SETTINGS["model"]["find_peaks"]["prominence"]
        )[0],
        axis=1,
        arr=data,
    )

    if check:
        # Display a plot of the data with the peaks marked
        for i, row in enumerate(data):
            (l,) = plt.plot(row)
            plt.plot(
                peak_indices[i],
                row[peak_indices[i]],
                "x",
                color=l.get_color(),
                label=f"Row {i}: {len(peak_indices[i])} peaks",
            )
        plt.legend()

    # Compute the number of peaks and whether there is doubt in the model
    n_peaks = int(np.round(np.mean([len(indices) for indices in peak_indices])))
    doubt = np.std([len(indices) for indices in peak_indices]) != 0

    return n_peaks, doubt, peak_indices
