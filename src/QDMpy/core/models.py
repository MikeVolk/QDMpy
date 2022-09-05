import numpy as np


def esr14n(x, parameter):
    """ ESR14N model

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
    """ ESR15N model

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
    """ ESRSINGLE model

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
        "n_peaks": 1,
        "model_func": None,
        "param": ["contrast", "center", "width", "offset"],
        "model_id": 0,
    },
    "ESR14N": {
        "n_peaks": 3,
        "model_func": esr14n,
        "param": ["center", "width", "contrast", "contrast", "contrast", "offset"],
        "model_id": 13,
    },
    "ESR15N": {
        "n_peaks": 2,
        "model_func": esr15n,
        "param": ["center", "width", "contrast", "contrast", "offset"],
        "model_id": 14,
    },
    "ESRSINGLE": {
        "n_peaks": 1,
        "model_func": esrsingle,
        "param": ["center", "width", "contrast", "offset"],
        "model_id": 15,
    },
}