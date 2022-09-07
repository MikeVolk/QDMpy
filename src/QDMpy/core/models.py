from abc import ABC, abstractmethod
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import QDMpy


class Model(ABC):
    """Abstract class for models.

    Args:
        ABC (ABC): Abstract class
    """

    def __init__(self, name: str, n_peaks: int, parameters_unique: List[str]):
        self.name = name
        self.parameters_unique = parameters_unique
        self.n_peaks = n_peaks

    @property
    def parameter(self):
        return [i.split('_')[0] for i in self.parameters_unique]

    @abstractmethod
    def func(self):
        """Abstract method for model function.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def calculate(self, x: NDArray, parameter: NDArray) -> NDArray:
        """Model function.

        Args:
            x (np.ndarray): x values
            parameter (np.ndarray): parameters

        Returns:
            np.ndarray: y values
        """
        self.func(x, parameter)

    @property
    def n_parameters(self) -> int:
        """Number of parameters.

        Returns:
            int: Number of parameters
        """
        return len(self.parameters_unique)

    def __repr__(self):
        return f"model({self.name}, n_parameters: {self.n_parameters}, n_peaks: {self.n_peaks})"


class ESR14N(Model):
    """ESR14N model
    """
    AHYP = 0.002158

    def __init__(self):
        super().__init__("ESR14N", 3, ["contrast", "center", "width_0", "width_1", "width_2", "offset"])

    def func(self):
        """Model function.

        Returns:
            function: Model function
        """
        return esr14n


class ESR15N(Model):

    def __init__(self):
        super().__init__("ESR15N", 2, ["contrast", "center", "width_0", "width_1", "offset"])

    def func(self):
        """Model function.

        Returns:
            function: Model function
        """
        return esr15n


class ESRSINGLE(Model):

    def __init__(self):
        super().__init__("ESRSINGLE", 1, ["contrast", "center", "width_0", "offset"])

    def func(self):
        """Model function.

        Returns:
            function: Model function
        """
        return esrsingle


def esr14n(x, parameter):
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
    out = []
    AHYP = 0.002158
    parameter = np.atleast_2d(parameter)
    for i in range(parameter.shape[0]):
        p = parameter[i]
        aux1 = x - p[0] + AHYP
        width_squared = p[1] * p[1]

        dip1 = p[2] * width_squared / (aux1 * aux1 + p[1] * p[1])

        aux2 = x - p[0]
        dip2 = p[3] * width_squared / (aux2 * aux2 + p[1] * p[1])

        aux3 = x - p[0] - AHYP
        dip3 = p[4] * width_squared / (aux3 * aux3 + p[1] * p[1])

        out.append(1 + p[5] - dip1 - dip2 - dip3)
    return np.array(out)


def esr15n(x, parameter):
    """ESR15N model

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
    out = []
    AHYP = 0.0015
    parameter = np.atleast_2d(parameter)

    for i in range(parameter.shape[0]):
        p = parameter[i]
        width_squared = p[1] * p[1]

        aux1 = x - p[0] + AHYP
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = x - p[0] - AHYP
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        out.append(1 + p[4] - dip1 - dip2)
    return np.array(out)


def esrsingle(x, parameter):
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
        peaks = find_peaks(-data[p, f], prominence=QDMpy.SETTINGS['model']['find_peaks']['prominence'])
        indices.append(peaks[0])
        if check:
            l, = plt.plot(data[p, f])
            plt.plot(peaks[0], data[p, f][peaks[0]], "x", color=l.get_color(),
                     label=f"({p},{f}): {len(indices[n])}")

    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))

    doubt = np.std([len(idx) for idx in indices]) != 0

    if check:
        plt.show()
        plt.legend()

    return n_peaks, doubt, peaks
