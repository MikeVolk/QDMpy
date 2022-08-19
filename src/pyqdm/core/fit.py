import logging

import numpy as np
import pygpufit.gpufit as gf

from pyqdm import settings
from pyqdm.core import models

FIT_PARAMETER = {
    "GAUSS_1D": ["contrast", "center", "width", "offset"],
    "ESR14N": ["center", "width", "contrast", "contrast", "contrast", "offset"],
    "ESR15N": ["center", "width", "contrast", "contrast", "offset"],
    "ESRSINGLE": ["center", "width", "contrast", "offset"],
}
UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}
BOUND_TYPES = {
    "FREE": 0,
    "LOWER": 1,
    "UPPER": 2,
    "LOWER_UPPER": 3,
}
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}

MODELS = [models.esr_14n, models.esr_15n, models.esr_single]


class Fit:
    def __init__(self, data, frequencies, model, constraints=None):

        self.LOG = logging.getLogger(f"pyQDM.{self.__class__.__name__}")
        self._data = data
        self.f_ghz = frequencies
        self.LOG.debug(
            f"Initializing Fit instance with data: {self.data.shape} at {frequencies.shape} frequencies with {model}"
        )
        self._model = model
        self._initial_parameter = None

        # fit results
        self._reset_fit()
        self._constraints: dict = {}
        self._constraint_types: np.array = None

        self.estimator_id = ESTIMATOR_ID[settings["fitting"]["estimator"]]  # 0 for LSE, 1 for MLE
        self._set_initial_constraints()
        if constraints is not None:
            for k in constraints:
                self.set_constraints(k, constraints[k])

    def _reset_fit(self):
        self._fitted = False
        self._parameter: np.array = None
        self._states: np.array = None
        self._chi_squares: np.array = None
        self._number_iterations: np.array = None
        self._execution_time: np.array = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if np.all(self._data == data):
            return
        self._data = data
        self._initial_parameter = None
        self._reset_fit()

    @property
    def initial_parameter(self):
        """
        Return the initial parameter.
        """
        if self._initial_parameter is None:
            self._initial_parameter = self.get_initial_parameter()
        return self._initial_parameter

    @property
    def model_id(self):
        return getattr(gf.ModelID, self._model)

    @property
    def fitting_parameter(self):
        return FIT_PARAMETER[self._model]

    @property
    def fitting_parameter_unique(self):
        """
        Return a list of unique fitting parameters.
        :return: list
        """
        lst = []
        for v in self.fitting_parameter:
            if self.fitting_parameter.count(v) > 1:
                for n in range(10):
                    if f"{v}_{n}" not in lst:
                        lst.append(f"{v}_{n}")
                        break
            else:
                lst.append(v)
        return lst

    @property
    def n_parameter(self):
        return len(self.fitting_parameter)

    def set_constraints(self, param, vmin=None, vmax=None, bound_type=None):
        """
        Set the constraints for the fit.

        Parameters
        ----------
        param : str
            The parameter to set the constraints for.
        values : list, optional
            The values to set the constraints to. The default is None.
        bound_type : str, int optional
            The bound type to set the constraints to. The default is None.
        """
        if isinstance(bound_type, int):
            bound_type = BOUND_TYPES[bound_type]

        if bound_type is not None and bound_type not in BOUND_TYPES:
            raise ValueError(f"Unknown constraint type: {bound_type} choose from {BOUND_TYPES}")

        if param == "contrast":
            for contrast in [v for v in self.fitting_parameter_unique if "contrast" in v]:
                self.set_constraints(contrast, vmin=vmin, vmax=vmax, bound_type=bound_type)
        else:
            self.LOG.debug(f"Setting constraints for {param}: ({vmin}, {vmax}) with {bound_type}")
            self._constraints[param] = [
                vmin,
                vmax,
                bound_type,
                UNITS[param.split("_")[0]],
            ]
            self._reset_fit()

    def set_free_constraints(self):
        """
        Set the constraints to be free.
        """
        for param in set(self.fitting_parameter_unique):
            self.set_constraints(param, bound_type="FREE")

    def _set_initial_constraints(self):
        """
        Set the initial constraints for the fit.
        """
        default_constraints = settings["default_fitconstraints"]
        self.set_constraints(
            "center",
            default_constraints["center_min"],
            default_constraints["center_max"],
            bound_type=default_constraints["center_type"],
        )
        self.set_constraints(
            "width",
            default_constraints["width_min"],
            default_constraints["width_max"],
            bound_type=default_constraints["width_type"],
        )
        self.set_constraints(
            "contrast",
            default_constraints["contrast_min"],
            default_constraints["contrast_max"],
            bound_type=default_constraints["contrast_type"],
        )
        self.set_constraints(
            "offset",
            default_constraints["offset_min"],
            default_constraints["offset_max"],
            bound_type=default_constraints["offset_type"],
        )

    def constraints_changed(self, constraints, constraint_types):
        """
        Check if the constraints have changed.
        """
        return self._constraints != constraints or self._constraint_types != constraint_types

    def get_constraints_array(self, n_pixel):
        """
        Return the constraints as an array (pixel, 2*fitting_parameters).

        :return: np.array
        """
        constraints = np.zeros((n_pixel, 2 * len(self.fitting_parameter)), dtype=np.float32)

        constraints_list = []
        for k in self.fitting_parameter_unique:
            constraints_list.extend((self._constraints[k][0], self._constraints[k][1]))
        constraints[:, :] = constraints_list
        return constraints

    def get_constraint_types(self):
        """
        Return the constraint types.
        :return: np.array
        """
        fit_bounds = [BOUND_TYPES[self._constraints[k][2]] for k in self.fitting_parameter_unique]
        return np.array(fit_bounds).astype(np.int32)

    # initial guess
    def _guess_center(self):
        """
        Guess the center of the ODMR spectra.
        """
        center = guess_center(self.data, self.f_ghz)
        self.LOG.debug(f"Guessing center frequency [GHz] of ODMR spectra {center.shape}.")
        return center

    def _guess_contrast(self):
        """
        Guess the contrast of the ODMR spectra.
        """
        contrast = guess_contrast(self.data)
        self.LOG.debug(f"Guessing contrast of ODMR spectra {contrast.shape}.")
        # np.ones((self.n_pol, self.n_frange, self.n_pixel)) * 0.03
        return contrast

    def _guess_width(self):
        """
        Guess the width of the ODMR spectra.
        """
        width = guess_width(self.data, self.f_ghz)
        self.LOG.debug(f"Guessing width of ODMR spectra {width.shape}.")
        return width

    def _guess_offset(self):
        """
        Guess the offset from 0 of the ODMR spectra. Usually this is 1
        """
        npol, nfrange, npixel, _ = self.data.shape
        offset = np.zeros((npol, nfrange, npixel))
        self.LOG.debug(f"Guessing offset {offset.shape}")
        return offset

    def get_initial_parameter(self):
        """
        Constructs an initial guess for the fit.
        """
        fit_parameter = []

        for p in self.fitting_parameter:
            param = getattr(self, f"_guess_{p}")()
            fit_parameter.append(param)

        fit_parameter = np.stack(fit_parameter, axis=fit_parameter[-1].ndim)
        return np.ascontiguousarray(fit_parameter, dtype=np.float32)

    def fit_odmr(self):
        if self._fitted:
            self.LOG.debug("Already fitted")
            return

        for irange in np.arange(0, self.data.shape[1]):
            self.LOG.info(
                f"Fitting frange {irange} from {self.f_ghz[irange].min():5.3f}-{self.f_ghz[irange].max():5.3f} GHz"
            )

            results = self.fit_frange(self.data[:, irange], self.f_ghz[irange], self.initial_parameter[:, irange])
            results = self.reshape_results(results)

            if self._parameter is None:
                self._parameter = results[0]
                self._states = results[1]
                self._chi_squares = results[2]
                self._number_iterations = results[3]
                self._execution_time = results[4]
            else:
                self._parameter = np.stack((self._parameter, results[0]))
                self._states = np.stack((self._states, results[1]))
                self._chi_squares = np.stack((self._chi_squares, results[2]))
                self._number_iterations = np.stack((self._number_iterations, results[3]))
                self._execution_time = np.stack((self._execution_time, results[4]))

            self.LOG.info(f"fit finished in {results[4]:.2f} seconds")
        self._parameter = np.swapaxes(self._parameter, 0, 1)
        self._fitted = True

    def fit_frange(self, data, freq, initial_parameters):
        """
        Wrapper for the fit_constrained function.

        :param data: np.array
            data for one frequency range, to be fitted
        :param freq: np.array
            frequency range
        :param initial_parameters: np.array
            initial guess for the fit
        :return: np.array of results
            results consist of: parameters, states, chi_squares, number_iterations, execution_time
        """
        # reshape the data into a single array with (n_pix*n_pol, n_freq)
        npol, npix, nfreq = data.shape
        data = data.reshape((-1, nfreq))
        initial_parameters = initial_parameters.reshape((-1, self.n_parameter))
        n_pixel = data.shape[0]
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        results = gf.fit_constrained(
            data=np.ascontiguousarray(data, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(initial_parameters, dtype=np.float32),
            weights=None,
            model_id=self.model_id,  # estimator_id = estimator_id,
            max_number_iterations=settings["fitting"]["max_number_iterations"],
            tolerance=settings["fitting"]["tolerance"],
        )

        return list(results)

    def reshape_results(self, results):
        for i in range(len(results)):
            if isinstance(results[i], float):
                continue
            results[i] = self.reshape_result(results[i])
        return results

    def reshape_result(self, result):
        """
        Reshape the results to the original shape of (npol, npix, -1)
        """
        npol, npix, _ = self.data[0].shape
        result = result.reshape((npol, npix, -1))
        return np.squeeze(result)

    #
    #     # # calculate the fit for each f_range -> 1st dimension is f_range not n_pol
    #     # # needs to be swapped later
    #     #
    #     # for i in np.arange(0, self.ODMRobj.n_frange):
    #     #     self.LOG.info(f"Fitting {self.POLARITIES[i]} polarization")
    #     #     data = self.ODMRobj.data[:, i].reshape(-1, self.ODMRobj.n_freqs).astype(np.float32)
    #     #     initial_guess = self.initial_guess[:, i].reshape(-1, len(self._fitting_params)).astype(np.float32)
    #     #
    #     #     # fit the data

    #     #     parameters.append(p)
    #     #     states.append(s)
    #     #     chi_squares.append(c)
    #     #     number_iterations.append(n)
    #     #     execution_time.append(t)
    #     #
    #     # if np.ndim(parameters) == 3:
    #     #     parameters = np.array(parameters)[:, np.newaxis, :, :]
    #     #     states = np.array(states)[:, np.newaxis, :]
    #     #     chi_squares = np.array(chi_squares)[:, np.newaxis, :]
    #     #     number_iterations = np.array(number_iterations)[:, np.newaxis, :]
    #     #
    #     # # swapping the f_range and n_pol axes
    #     # parameters = self._reshape_parameter(parameters, self.ODMRobj.n_pol, self.ODMRobj.n_frange)
    #     #
    #     # # check if the reshaping is correct
    #     # assert np.all(parameters[0, 1] == p[: int(p.shape[0] / self.ODMRobj.n_pol)])
    #     #
    #     # # reshape the states and chi_squares
    #     # shape = (self.ODMRobj.n_pol, self.ODMRobj.n_frange, self.ODMRobj.n_pixel)
    #     # states = np.reshape(states, shape)
    #     # chi_squares = np.reshape(chi_squares, shape)
    #     # number_iterations = np.reshape(number_iterations, shape)
    #     #
    #     # # flip pol / frange
    #     # states = np.swapaxes(states, 0, 1)
    #     # chi_squares = np.swapaxes(chi_squares, 0, 1)
    #     # number_iterations = np.swapaxes(number_iterations, 0, 1)
    #     #
    #     # self._fitted_parameter = np.rollaxis(parameters, -1)
    #     # self._states = states
    #     # self._chi_squares = chi_squares
    #     # self._number_iterations = number_iterations
    #     # self.execution_time = execution_time
    #     #
    #     # t = "; ".join([f"{v:.2f}" for v in execution_time])
    #     # self.LOG.debug(f"Fitting took {np.sum(execution_time):.3f} ({t}) seconds.")
    #     # if np.any(np.flatnonzero(states)):
    #     #     n = len(np.flatnonzero(states))
    #     #     self.LOG.warning(f"Fit did not converge in {n} pixels.")
    #     #
    #     # self._fitted = True
    #     # self._fitting_constraints = self.constraints


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
    amp = np.abs((mx - mn) / mx)
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
    return freq[lidx] - freq[ridx]


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


def make_dummy_data(model="esr_14n", n_freq=100):
    from pyqdm.core import models

    if model == "esr_14n":
        model = models.esr_14n
    elif model == "esr_15n":
        model = models.esr_15n
    else:
        raise ValueError(f"Unknown model {model}")

    f0 = np.linspace(2.84, 2.85, n_freq)  # in GHz
    f1 = np.linspace(2.89, 2.9, n_freq)

    # model for the left frange
    # +
    p00 = np.tile([np.mean(f0) - 0.0001, 0.0001, 0.01, 0.01, 0.01, 0], (100, 1))
    m00 = model(f0, p00)
    # -
    p10 = np.tile([np.mean(f0) + 0.0001, 0.0001, 0.01, 0.01, 0.01, 0], (100, 1))
    m10 = model(f0, p10)

    # model for the right frange
    # +
    p01 = np.tile([np.mean(f1) - 0.0001, 0.0001, 0.01, 0.01, 0.01, 0], (100, 1))
    m01 = model(f1, p01)
    # -
    p11 = np.tile([np.mean(f1) + 0.0001, 0.0001, 0.01, 0.01, 0.01, 0], (100, 1))
    m11 = model(f1, p11)

    for p in [p00, p10, p01, p11]:
        print(p[0])
    # combine the models
    mall = np.stack([np.stack([m00, m01]), np.stack([m10, m11])])
    pall = np.stack([np.stack([p00, p01]), np.stack([p10, p11])])
    f_ghz = np.stack([f0, f1])
    return mall, f_ghz, pall
