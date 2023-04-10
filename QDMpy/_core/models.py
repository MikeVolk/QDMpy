"""
models.py

This module provides a collection of functions for fitting various electron spin resonance (ESR) models. 
It includes models for ESR14N, ESR15N, ESR14N_folded, ESR15N_shifted, and single Lorentzian peaks. 
It also provides a function to guess the diamond type based on the number of peaks in the data.

Functions:
    - lorentzian_peak: Calculates a single Lorentzian peak
    - esr14n: ESR14N model with three Lorentzian peaks
    - esr14n_folded: Folded ESR14N model
    - esr15n: ESR15N model with two Lorentzian peaks
    - esr15n_shifted: Shifted ESR15N model
    - esrsingle: Single Lorentzian peak model
    - guess_model: Guesses the diamond type based on the number of peaks
    - full_model: Computes the full model for a given set of frequencies and parameters

Dictionaries:
    - IMPLEMENTED: Contains information about implemented models, including function names, number of peaks, 
                   parameter names, model IDs, and display names
    - PEAK_TO_TYPE: Maps the number of peaks to the corresponding model type

Example:
    >>> import numpy as np
    >>> from models import esr14n

    # Generate a frequency axis
    >>> f = np.linspace(0, 10, 1000)

    # Define parameters for the ESR14N model
    >>> parameters = np.array([5.0, 1.0, 0.5, 0.5, 0.5, 0.0])

    # Compute the model
    >>> model = np.zeros_like(f)
    >>> esr14n(f, parameters, model)

    # Plot the result
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, model)
    >>> plt.xlabel("Frequency (GHz)")
    >>> plt.ylabel("Signal (a.u.)")
    >>> plt.show()
"""

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import float64, guvectorize


import QDMpy


"""
Dictionary containing the implemented models and their properties.

Each key represents a model, and its value is a dictionary containing:

- "func_name": The string name of the model.
- "n_peaks": The number of peaks in the model.
- "func": The function implementing the model.
- "params": A list of parameter names for the model.
- "model_id": An integer identifier for the model.
- "name": A human-readable name for the model.
"""


@guvectorize(
    [(float64[:], float64, float64, float64, float64[:])],
    "(n),(),(),()->(n)",
    forceobj=True,
)
def lorentzian_peak(f, center, width, contrast, out):
    """
    Calculates a single Lorentzian peak.

    Args:
        f : np.ndarray
            Frequency axis.
        center : float
            Center frequency of the peak.
        width : float
            Width of the peak.
        contrast : float
            Contrast of the peak.
        out : np.ndarray
            Model array to be filled with the model.

    Returns:
        np.ndarray of the y values. The function directly updates the 'out'
        parameter with the calculated model.

    Notes:
        The model is calculated as a numpy ufunc (using numba), so it can be
        used in a vectorized way. Meaning that the parameters can be arrays but
        need to have the same shape.

    Example:
        # Generate a frequency axis f = np.linspace(0, 10, 1000)

        # Define parameters for a single Lorentzian peak center = 5.0 width =
        1.0 contrast = 0.5

        # Compute the model model = np.zeros_like(f) lorentzian_peak(f, center,
        width, contrast, model)

        # Plot the result import matplotlib.pyplot as plt plt.plot(f, model)
        plt.xlabel("Frequency (GHz)") plt.ylabel("Signal (a.u.)") plt.show()

    """

    sq_width = np.square(width)
    delta = f - center
    out[:] = contrast * sq_width / (delta**2 + sq_width)


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr14n(x: np.ndarray, parameter: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Calculates the ESR14N model, which consists of three Lorentzian peaks.

    Args:
        x (np.ndarray): Frequency axis values. parameter (np.ndarray): Model
        parameters as follows:
            parameter[0] = center: Center frequency of the middle peak.
            parameter[1] = width: Width of the peaks. parameter[2] = c0:
            Contrast of the left peak. parameter[3] = c1: Contrast of the middle
            peak. parameter[4] = c2: Contrast of the right peak. parameter[5] =
            offset: Offset of the baseline.
        model (np.ndarray): Model array to be filled with the calculated model.

    Returns:
        np.ndarray: Directly updates the 'model' parameter with the calculated y
        values.

    Notes:
        The ESR14N model is calculated as a numpy ufunc (using numba), so it can
        be used in a vectorized way. This means that the parameters can be
        arrays but need to have the same shape.

    Example:
        # Generate a frequency axis x = np.linspace(0, 10, 1000)

        # Define parameters for the ESR14N model parameters = np.array([5.0,
        1.0, 0.5, 0.6, 0.7, 0.1])

        # Compute the model model = np.zeros_like(x) esr14n(x, parameters,
        model)

        # Plot the result import matplotlib.pyplot as plt plt.plot(x, model)
        plt.xlabel("Frequency (GHz)") plt.ylabel("Signal (a.u.)") plt.show()

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
def esr14n_folded(x: np.ndarray, parameter: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Calculates the ESR14N folded model, which consists of six Lorentzian peaks.

    Args:
        x (np.ndarray): Frequency axis values. parameter (np.ndarray): Model
        parameters as follows:
            parameter[0] = center: Center frequency of the middle peak.
            parameter[1] = width: Width of the peaks. parameter[2] = c0:
            Contrast of the left peak. parameter[3] = c1: Contrast of the middle
            peak. parameter[4] = c2: Contrast of the right peak. parameter[5] =
            offset: Offset of the baseline. parameter[6] = fsec: Secondary
            frequency for folding.
        model (np.ndarray): Model array to be filled with the calculated model.

    Returns:
        np.ndarray: Directly updates the 'model' parameter with the calculated y
        values.

    Notes:
        The ESR14N folded model is calculated as a numpy ufunc (using numba), so
        it can be used in a vectorized way. This means that the parameters can
        be arrays but need to have the same shape.

    Example:
        # Generate a frequency axis x = np.linspace(0, 10, 1000)

        # Define parameters for the ESR14N folded model parameters =
        np.array([5.0, 1.0, 0.5, 0.6, 0.7, 0.1, 0.002158])

        # Compute the model model = np.zeros_like(x) esr14n_folded(x,
        parameters, model)

        # Plot the result import matplotlib.pyplot as plt plt.plot(x, model)
        plt.xlabel("Frequency (GHz)") plt.ylabel("Signal (a.u.)") plt.show()

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
def esr15n(x: np.ndarray, parameter: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Calculates the ESR15N model, which consists of two Lorentzian peaks.

    Args:
        x (np.ndarray): Frequency axis values. parameter (np.ndarray): Model
        parameters as follows:
            parameter[0] = center: Center frequency of the peaks. parameter[1] =
            width: Width of the peaks. parameter[2] = c0: Contrast of the left
            peak. parameter[3] = c1: Contrast of the right peak. parameter[4] =
            offset: Offset of the baseline.
        model (np.ndarray): Model array to be filled with the calculated model.

    Returns:
        np.ndarray: Directly updates the 'model' parameter with the calculated y
        values.

    Notes:
        The ESR15N model is calculated as a numpy ufunc (using numba), so it can
        be used in a vectorized way. This means that the parameters can be
        arrays but need to have the same shape.

    Example:
        # Generate a frequency axis x = np.linspace(0, 10, 1000)

        # Define parameters for the ESR15N model parameters = np.array([5.0,
        1.0, 0.5, 0.6, 0.1])

        # Compute the model model = np.zeros_like(x) esr15n(x, parameters,
        model)

        # Plot the result import matplotlib.pyplot as plt plt.plot(x, model)
        plt.xlabel("Frequency (GHz)") plt.ylabel("Signal (a.u.)") plt.show()

    """
    AHYP = 0.0015

    center, width, c0, c1, offset = parameter
    center0 = center + AHYP
    center1 = center - AHYP

    model[:] = 1 + offset - np.sum(lorentzian_peak(x, [center0, center1], width, [c0, c1]), axis=0)


@guvectorize(
    [(float64[:], float64[:], float64[:])],
    "(n),(m)->(n)",
    forceobj=True,
    target="parallel",
)
def esrsingle(x: np.ndarray, param: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Calculates a single Lorentzian model.

    Args:
        x (np.ndarray): Frequency axis values. param (np.ndarray): Model
        parameters as follows:
            param[0] = center: Center frequency of the peak. param[1] = width:
            Width of the peak. param[2] = contrast: Contrast of the peak.
            param[3] = offset: Offset of the baseline.
        model (np.ndarray): Model array to be filled with the calculated model.

    Returns:
        np.ndarray: Directly updates the 'model' parameter with the calculated y
        values.

    Notes:
        The single Lorentzian model is calculated as a numpy ufunc (using numba)
        with parallel target, so it can be used in a vectorized way. This means
        that the parameters can be arrays but need to have the same shape.

    Example:
        # Generate a frequency axis x = np.linspace(0, 10, 1000)

        # Define parameters for the single Lorentzian model parameters =
        np.array([5.0, 1.0, 0.5, 0.1])

        # Compute the model model = np.zeros_like(x) esrsingle(x, parameters,
        model)

        # Plot the result import matplotlib.pyplot as plt plt.plot(x, model)
        plt.xlabel("Frequency (GHz)") plt.ylabel("Signal (a.u.)") plt.show()

    """
    model[:] = 1 + param[3] - np.sum(lorentzian_peak(x, param[0], param[1], param[2]), axis=0)


def full_model(model: str, freqs: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """
    Full model that applies a specified model to the given frequencies and
    parameters.

    Args:
        model (str): Model name. One of {"ESR14N", "ESR15N", "ESRSINGLE"}. freqs
        (np.ndarray): Frequencies as a 1D numpy array. parameters (np.ndarray):
        Model parameters as a 1D numpy array.

    Returns:
        np.ndarray: Model output as a 1D numpy array.

    Raises:
        ValueError: If the specified model name is not implemented.

    Example:
        >>> import numpy as np
        >>> freqs = np.linspace(2.85, 3.05, 1000)
        >>> params = np.array([2.950, 0.001, 0.25, 0.25, 0.25, 0.01])
        >>> output = full_model("ESR14N", freqs, params)
        >>> output.shape
        (1000,)

    In this example, we first generate a frequency axis using `numpy.linspace`
    and define a set of parameters `params`. We then call the `full_model`
    function with the model name "ESR14N", the frequency axis, and the
    parameters. The function returns the model output as a 1D numpy array with
    the same length as the input frequency axis.
    """
    model = model.upper()

    if model not in IMPLEMENTED:
        raise ValueError("Model not implemented")

    model_func = IMPLEMENTED[model]["func"]
    low_f_data = model_func(freqs[: len(freqs) // 2], parameters)
    high_f_data = model_func(freqs[len(freqs) // 2 :], parameters)

    return np.concatenate((low_f_data, high_f_data), axis=-1)


def guess_model(data: np.ndarray, check: bool = False) -> Tuple[int, bool, Any]:
    """
    Guess the diamond type based on the number of peaks.

    Args:
        data (np.ndarray): A 2D numpy array containing the diamond data. check
        (bool, optional): If True, displays a plot of the diamond data with the
        peaks marked. Default is False.

    Returns:
        Tuple[int, bool, Any]: A tuple containing the number of peaks, a boolean
        indicating whether there is doubt in the model, and the indices of the
        peaks.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.85, 3.05, 1000)
        >>> params = np.array([2.950, 0.001, 0.25, 0.25, 0.25, 0.01])
        >>> model = np.zeros_like(x)
        >>> esr14n(x, params, model)
        >>> data = np.stack([model] * 10)
        >>> guess_model(data)
        (3, False, array([array([348, 500, 652], dtype=int64), array([348, 500, 652], dtype=int64), ..., array([348, 500, 652], dtype=int64)]))

    This example generates a frequency axis using numpy.linspace and test data
    using the esr14n model. The generated data contains 3 peaks. The
    guess_model() function then returns a tuple containing the number of peaks
    (which in this case is 3), a boolean indicating that there is no doubt in
    the model (since the standard deviation is 0), and an array of arrays
    containing the peak indices for each row (the same peak indices repeated for
    each row, as the data is the same for all rows).
    """

    from scipy.signal import find_peaks

    # Compute the indices of the peaks for each row of the data array
    peak_indices = []
    for p, f in np.ndindex(*data.shape[:2]):
        peaks = find_peaks(
            -data[p, f], prominence=QDMpy.SETTINGS["model"]["find_peaks"]["prominence"]
        )
        peak_indices.append(peaks[0])
        if check:
            (l,) = plt.plot(data[p, f])
            plt.plot(
                peaks[0],
                data[p, f][peaks[0]],
                "x",
                color=l.get_color(),
                label=f"({p},{f}): {len(peak_indices[p])}",
            )
            plt.legend()

    # Compute the number of peaks and whether there is doubt in the model
    n_peaks = int(np.round(np.mean([len(indices) for indices in peak_indices])))
    doubt = np.std([len(indices) for indices in peak_indices]) != 0

    return n_peaks, doubt, peak_indices


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(n)", forceobj=True)
def esr15n_superposition(x: np.ndarray, parameter: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    ESR15N superposition model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast of the left doublet
            parameter[3] = contrast of the right doublet
            parameter[4] = offset
            parameter[5] = shift frequency

    Returns:
        np.ndarray: y values

    Example:
        # Generate a frequency axis
        f = np.linspace(0, 10, 1000)

        # Define parameters for the ESR15N superposition model
        center = 5.0
        width = 1.0
        contrast0 = 0.5
        contrast1 = 0.7
        offset = 0.1
        shift_freq = 0.2

        # Compute the model
        parameters = [center, width, contrast0, contrast1, offset, shift_freq]
        model = np.zeros_like(f)
        esr15n_superposition(f, parameters, model)

        # Plot the result
        import matplotlib.pyplot as plt
        plt.plot(f, model)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Signal (a.u.)")
        plt.title("ESR15N Superposition Model")
        plt.show()
    """
    AHYP = 0.0015

    center, width, c0, c1, offset, shift_freq = parameter
    center0 = center + AHYP
    center1 = center - AHYP

    model1 = np.zeros_like(x)
    model2 = np.zeros_like(x)

    esr15n(x, [center, width, c0, c0, offset], model1)
    esr15n(x + shift_freq, [center, width, c1, c1, offset], model2)

    model[:] = model1 + model2


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


"""
Dictionary mapping the number of peaks to the corresponding model type.

The keys are integers representing the number of peaks, and the values are strings representing the corresponding model types. This dictionary is dynamically generated from the IMPLEMENTED dictionary.
"""

PEAK_TO_TYPE = {v["n_peaks"]: k for k, v in IMPLEMENTED.items()}
