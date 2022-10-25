import logging
import os.path
from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple, Union, Any, Dict, Optional

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import QDMpy
from QDMpy._core.models import guess_model

if QDMpy.PYGPUFIT_PRESENT:  # type: ignore[has-type]
    import pygpufit.gpufit as gf

from scipy.io import savemat

from QDMpy._core import models

UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}
CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}


def main():
    from QDMpy._core.qdm import QDM

    q = QDM.from_qdmio(QDMpy.test_data_location())
    q.fit_odmr()


if __name__ == "__main__":
    main()


class Fit:
    LOG = logging.getLogger(__name__)

    def __init__(
        self,
        data: NDArray,
        frequencies: NDArray,
        model_name: str = "auto",
        constraints: Optional[Dict[str, Any]] = None,
    ):
        """
        Fit the data to a model.
        Args:
            data: 3D array of the data to fit.
            frequencies: 1D array of the frequencies.
            model_name: Name of the model to fit. (Default value = 'auto')
                if 'auto' the model is guessed from the data.
                See Also: `models.guess_model_name`
            constraints: Constraints for the fit. (Default value = None)
                If None, the default constraints from the config.ini file in QDMpy.CONFIG_PATH are used.
        """

        self._data = data
        self.f_ghz = frequencies
        self.LOG.debug(
            f"Initializing Fit instance with data: {self.data.shape} at {frequencies.shape} frequencies."
        )

        if model_name == "auto":
            model_name = self.guess_model_name()

        # setting model name resets the fit parameters
        self._model = models.IMPLEMENTED[model_name.upper()]
        self._initial_parameter = None

        # fit results
        self._reset_fit()
        self._constraints = (
            self._set_initial_constraints()
        )  # structure is: type: [float(min), float(vmax), str(constraint_type), str(unit)]

        self.estimator_id = ESTIMATOR_ID[
            QDMpy.SETTINGS["fit"]["estimator"]
        ]  # 0 for LSE, 1 for MLE

    def __repr__(self) -> str:
        return f"Fit(data: {self.data.shape},f: {self.f_ghz.shape}, model:{self.model_name})"

    @property
    def data(self) -> NDArray:
        return self._data

    @data.setter
    def data(self, data: NDArray) -> None:
        self.LOG.info("Data changed, fits need to be recalculated!")
        if np.all(self._data == data):
            return
        self._data = data
        self._initial_parameter = None
        self._reset_fit()

    ### MODEL RELATED METHODS ###
    def guess_model_name(self) -> str:
        """Guess the model name from the data."""
        data = np.median(self.data, axis=2)
        n_peaks, doubt, peaks = guess_model(data)

        if doubt:
            self.LOG.warning(
                "Doubt on the diamond type. Check using `guess_diamond_type('debug')` and set manually if incorrect."
            )

        model = [
            mdict
            for m, mdict in models.IMPLEMENTED.items()
            if mdict["n_peaks"] == n_peaks
        ][0]
        
        self.LOG.info(
            f"Guessed diamond type: {n_peaks} peaks -> {model['func_name']} ({model['name']})"
        )
        return model["func_name"]

    @property
    def model_func(self) -> Callable:
        return self._model["func"]

    @property
    def model_params(self) -> dict:
        """
        Return the model parameters.
        """
        return self._model["params"]

    @property
    def model(self) -> dict:
        """
        Return the model dictionary.
        """
        return self._model

    @property
    def model_name(self) -> Callable:
        return self._model["func_name"]

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        if model_name.upper() not in models.IMPLEMENTED:
            raise ValueError(
                f"Unknown model: {model_name} choose from {list(models.IMPLEMENTED.keys())}"
            )

        self._model = models.IMPLEMENTED[model_name]
        self._constraints = self._set_initial_constraints()

        self.LOG.debug(
            f"Setting model to {model_name}, resetting all fit results and initial parameters."
        )
        self._reset_fit()
        self._initial_parameter = self.get_initial_parameter()

    @property
    def model_id(self) -> int:
        return self._model["model_id"]

    @property
    def model_params_unique(self) -> List[str]:
        """
        Return a list of unique fitting parameters.
        :return: list
        """
        lst = []
        for v in self.model_params:
            if self.model_params.count(v) > 1:
                for n in range(10):
                    if f"{v}_{n}" not in lst:
                        lst.append(f"{v}_{n}")
                        break
            else:
                lst.append(v)
        return lst

    @property
    def n_parameter(self) -> int:
        return len(self.model_params)

    ### INITIAL PARAMETER RELATED METHODS ###
    @property
    def initial_parameter(self) -> NDArray:
        """
        Return the initial parameter.
        """
        if self._initial_parameter is None:
            self._initial_parameter = self.get_initial_parameter()
        return self._initial_parameter

    ### COSTRAINTS RELATED METHODS ###
    def _set_initial_constraints(self) -> Dict[str, List[Any]]:
        """
        Get the default constraints dictionary for the fit.
        """
        constraints = QDMpy.SETTINGS["fit"]["constraints"]

        defaults = {}
        for value in self.model_params_unique:
            v = value.split("_")[0]
            defaults[value] = [
                constraints[f"{v}_min"],
                constraints[f"{v}_max"],
                constraints[f"{v}_type"],
                UNITS[v],
            ]
        return defaults

    def set_constraints(
        self,
        param: str,
        vmin: Union[float, None] = None,
        vmax: Union[float, None] = None,
        constraint_type: Union[str, None] = None,
        reset_fit: bool = True,
    ):
        """
        Set the constraints for the fit.

        :param param: str
            The parameter to set the constraints for.
        :param vmin: float, optional
            The minimum value to set the constraints to. The default is None.
        :param vmax: float, optional
            The maximum value to set the constraints to. The default is None.
        :param constraint_type: str optional
            The bound type to set the constraints to. The default is None.
        :param reset_fit: bool, optional
            Whether to reset the fit results. The default is True.
        """
        if isinstance(constraint_type, int):
            constraint_type = CONSTRAINT_TYPES[constraint_type]

        if constraint_type is not None and constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(
                f"Unknown constraint type: {constraint_type} choose from {CONSTRAINT_TYPES}"
            )

        if param == "contrast" and self.model_params_unique != self.model_params:
            for contrast in [v for v in self.model_params_unique if "contrast" in v]:
                self.set_constraints(
                    contrast, vmin=vmin, vmax=vmax, constraint_type=constraint_type
                )
        else:
            self.LOG.debug(
                f"Setting constraints for {param}: ({vmin}, {vmax}) with {constraint_type}"
            )
            self._constraints[param] = [
                vmin,
                vmax,
                constraint_type,
                UNITS[param.split("_")[0]],
            ]

        if reset_fit:
            self._reset_fit()

    def set_free_constraints(self):
        """
        Set the constraints to be free.
        """
        for param in set(self.model_params_unique):
            self.set_constraints(param, constraint_type="FREE")

    @property
    def constraints(self) -> Dict[str, List[Union[float, str]]]:
        return self._constraints

    # todo not used
    def constraints_changed(
        self, constraints: List[float], constraint_types: List[str]
    ) -> bool:
        """
        Check if the constraints have changed.
        """
        return (
            list(self._constraints.keys()) != constraints
            or self._constraint_types != constraint_types
        )

    def get_constraints_array(self, n_pixel: int) -> NDArray:
        """
        Return the constraints as an array (pixel, 2*fitting_parameters).
        :return: np.array
        """
        constraints_list: List[float] = []
        for k in self.model_params_unique:
            constraints_list.extend((self._constraints[k][0], self._constraints[k][1]))
        constraints = np.tile(constraints_list, (n_pixel, 1))
        return constraints

    def get_constraint_types(self) -> NDArray:
        """
        Return the constraint types.
        :return: np.array
        """
        fit_bounds = [
            CONSTRAINT_TYPES.index(self._constraints[k][2])
            for k in self.model_params_unique
        ]
        return np.array(fit_bounds).astype(np.int32)

    # parameters
    @property
    def parameter(self) -> NDArray:
        return self._fit_results

    def get_param(self, param: str) -> Union[NDArray, None]:
        """
        Get the value of a parameter reshaped to the image dimesions.
        """
        if not self.fitted:
            raise NotImplementedError("No fit has been performed yet. Run fit_odmr().")
        if param in ("chi2", "chi_squares", "chi_squared"):
            return self._chi_squares
        idx = self._param_idx(param)
        if param == "mean_contrast":
            return np.mean(self._fit_results[:, :, :, idx], axis=-1)  # type: ignore[index]
        return self._fit_results[:, :, :, idx]  # type: ignore[index]

    def _param_idx(self, parameter: str) -> List[int]:
        """
        Get the index of the fitted parameter.
        :param parameter:
        :return:
        """
        if parameter == "resonance":
            parameter = "center"
        if parameter == "mean_contrast":
            parameter = "contrast"
        idx = [i for i, v in enumerate(self.model_params) if v == parameter]
        if not idx:
            idx = [i for i, v in enumerate(self.model_params_unique) if v == parameter]
        if not idx:
            raise ValueError(f"Unknown parameter: {parameter}")
        return idx

    # initial guess
    def _guess_center(self) -> NDArray:
        """
        Guess the center of the ODMR spectra.
        """
        center = guess_center(self.data, self.f_ghz)
        self.LOG.debug(
            f"Guessing center frequency [GHz] of ODMR spectra {center.shape}."
        )
        return center

    def _guess_contrast(self) -> NDArray:
        """
        Guess the contrast of the ODMR spectra.
        """
        contrast = guess_contrast(self.data)
        self.LOG.debug(f"Guessing contrast of ODMR spectra {contrast.shape}.")
        # np.ones((self.n_pol, self.n_frange, self.n_pixel)) * 0.03
        return contrast

    def _guess_width(self) -> NDArray:
        """
        Guess the width of the ODMR spectra.
        """
        correct = 0

        # detection thresholds
        if self._model["n_peaks"] == 1:
            vmin, vmax = 0.3, 0.7
        elif self._model["n_peaks"] == 2:
            vmin, vmax = 0.4, 0.6
            correct = -0.001
        elif self._model["n_peaks"] == 3:
            vmin, vmax = 0.35, 0.65
        else:
            raise ValueError("n_peaks must be 1, 2 or 3")

        width = guess_width(self.data, self.f_ghz, vmin, vmax)
        self.LOG.debug(f"Guessing width of ODMR spectra {width.shape}.")
        return width

    def _guess_offset(self) -> NDArray:
        """
        Guess the offset from 0 of the ODMR spectra. Usually this is 1
        """
        n_pol, nfrange, n_pixel, _ = self.data.shape
        offset = np.zeros((n_pol, nfrange, n_pixel))
        self.LOG.debug(f"Guessing offset {offset.shape}")
        return offset

    def get_initial_parameter(self) -> NDArray:
        """
        Constructs an initial guess for the fit.
        """
        fit_parameter = []

        for p in self.model_params:
            param = getattr(self, f"_guess_{p}")()
            fit_parameter.append(param)

        fit_parameter = np.stack(fit_parameter, axis=fit_parameter[-1].ndim)
        return np.ascontiguousarray(fit_parameter, dtype=np.float32)

    ### fitting related methods ###
    def _reset_fit(self) -> None:
        """Reset the fit results."""
        self._fitted = False
        self._fit_results = None
        self._states = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit_odmr(self, refit=False) -> None:
        if self._fitted and not refit:
            self.LOG.debug("Already fitted")
            return
        if self.fitted and refit:
            self._reset_fit()
            self.LOG.debug("Refitting the ODMR data")

        for irange in np.arange(0, self.data.shape[1]):
            self.LOG.info(
                f"Fitting frange {irange} from {self.f_ghz[irange].min():5.3f}-{self.f_ghz[irange].max():5.3f} GHz"
            )

            results = self.fit_frange(
                self.data[:, irange],
                self.f_ghz[irange],
                self.initial_parameter[:, irange],
            )
            results = self.reshape_results(results)

            if self._fit_results is None:
                self._fit_results = results[0]
                self._states = results[1]
                self._chi_squares = results[2]
                self._number_iterations = results[3]
                self._execution_time = results[4]
            else:
                self._fit_results = np.stack((self._fit_results, results[0]))
                self._states = np.stack((self._states, results[1]))
                self._chi_squares = np.stack((self._chi_squares, results[2]))
                self._number_iterations = np.stack(
                    (self._number_iterations, results[3])
                )
                self._execution_time = np.stack((self._execution_time, results[4]))

            self.LOG.info(f"fit finished in {results[4]:.2f} seconds")
        self._fit_results = np.swapaxes(self._fit_results, 0, 1)  # type: ignore[call-overload]
        self._fitted = True

    def fit_frange(
        self, data: NDArray, freq: NDArray, initial_parameters: NDArray
    ) -> List[NDArray]:
        """
        Wrapper for the fit_constrained function.

        Args:
            data: data for one frequency range, to be fitted. array of size (n_pol, n_pixel, n_freqs) of the ODMR data
            freq: array of size (n_freqs) of the frequencies
            initial_parameters: initial guess for the fit, an array of size (n_pol * n_pixel, 2 * n_param) of
                the initial parameters

        Returns:
            fit_results: results consist of: parameters, states, chi_squares, number_iterations, execution_time
                results: array of size (n_pol*n_pixel, n_param) of the fitted parameters
                states: array of size (n_pol*n_pixel) of the fit states (i.e. did the fit work)
                chi_squares: array of size (n_pol*n_pixel) of the chi squares
                number_iterations: array of size (n_pol*n_pixel) of the number of iterations
                execution_time: execution time
        """
        # reshape the data into a single array with (n_pix*n_pol, n_freqs)
        n_pol, n_pix, n_freqs = data.shape
        data = data.reshape((-1, n_freqs))
        initial_parameters = initial_parameters.reshape((-1, self.n_parameter))
        n_pixel = data.shape[0]
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        results = gf.fit_constrained(
            data=np.ascontiguousarray(data, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(
                initial_parameters, dtype=np.float32
            ),
            weights=None,
            model_id=self.model_id,
            max_number_iterations=QDMpy.SETTINGS["fit"]["max_number_iterations"],
            tolerance=QDMpy.SETTINGS["fit"]["tolerance"],
        )

        return list(results)

    def reshape_results(self, results: List[NDArray]) -> NDArray:
        """Reshape the results from the fit_constrained function into the correct shape.

        Args:
            results: results consist of: parameters, states, chi_squares, number_iterations, execution_time

        Returns:
            results: results consist of: parameters, states, chi_squares, number_iterations, execution_time
                results: array of size (n_pol, n_pixel, n_param) of the fitted parameters
                states: array of size (n_pol, n_pixel) of the fit states (i.e. did the fit work)
                chi_squares: array of size (n_pol, n_pixel) of the chi squares
                number_iterations: array of size (n_pol, n_pixel) of the number of iterations
                execution_time: execution time
        """
        for i in range(len(results)):
            if isinstance(results[i], float):
                continue
            results[i] = self.reshape_result(results[i])
        return results

    def reshape_result(self, result: NDArray) -> NDArray:
        """
        Reshape the results to the original shape of (n_pol, npix, -1)

        Args:
            result: array of size (n_pol * n_pixel, -1) of the fitted parameters

        Returns:
            result: array of size (n_pol, n_pixel, -1) of the fitted parameters
        """
        n_pol, npix, _ = self.data[0].shape
        result = result.reshape((n_pol, npix, -1))
        return np.squeeze(result)


@numba.njit(parallel=True)
def guess_contrast(data):
    """
    Guess the contrast of a ODMR data.

    :param data: np.array
        data to guess the contrast from
    :return: np.array
        contrast of the data
    """
    amp = np.zeros(data.shape[:-1])

    for i, j in np.ndindex(data.shape[0], data.shape[1]):
        for p in numba.prange(data.shape[2]):
            amp[i, j, p] = guess_contrast_pixel(data[i, j, p])
    return amp


@numba.njit()
def guess_contrast_pixel(data):
    mx = np.nanmax(data)
    mn = np.nanmin(data)
    return np.abs((mx - mn) / mx)


@numba.njit(parallel=True, fastmath=True)
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
    center = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        for px in numba.prange(data.shape[2]):
            center[p, f, px] = guess_center_pixel(data[p, f, px], freq[f])
    return center


@numba.njit(fastmath=True)
def guess_center_pixel(pixel, freq):
    """
    Guess the center frequency of a single frequency range.

    :param data: np.array
        data to guess the center frequency from
    :param freq: np.array
        frequency range of the data
    :return: np.array
        center frequency of the data
    """
    pixel = normalized_cumsum_pixel(pixel)
    idx = np.argmin(np.abs(pixel - 0.5))
    return freq[idx]


@numba.njit(parallel=True, fastmath=True)
def guess_width(data: NDArray, f_GHz: NDArray, vmin: float, vmax: float) -> NDArray:
    """
    Guess the width of a ODMR resonance peaks.

    :param data: np.array
        data to guess the width from
    :param f_GHz: np.array
        frequency range of the data
    :param vmin: float
        minimum value of normalized cumsum to be considered
    :param vmax: float
        maximum value of normalized cumsum to be considered

    :return: np.array
        width of the data
    """
    # width
    width = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        freq = f_GHz[f]
        for px in numba.prange(data.shape[2]):
            width[p, f, px] = guess_width_pixel(data[p, f, px], freq, vmin, vmax)

    return width


@numba.njit(fastmath=True)
def guess_width_pixel(
    pixel: NDArray, freq: NDArray, vmin: float, vmax: float
) -> NDArray:
    """
    Guess the width of a single frequency range.

    :param data: np.array
        data to guess the width from
    :param freq: np.array
        frequency range of the data

    :return: np.array
        width of the data

    Raises ValueError if the number of peaks is not 1, 2 or 3.
    """
    pixel = normalized_cumsum_pixel(pixel)
    lidx = np.argmin(np.abs(pixel - vmin))
    ridx = np.argmin(np.abs(pixel - vmax))
    return freq[lidx] - freq[ridx]


@numba.njit(parallel=True, fastmath=True)
def normalized_cumsum(data: NDArray) -> NDArray:
    """
    Guess the width of a ODMR resonance peaks.

    :param data: np.array
        data to guess the width from
    :param f_GHz: np.array
        frequency range of the data

    :return: np.array
        width of the data
    """
    # width
    width = np.zeros(data.shape)
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        for px in numba.prange(data.shape[2]):
            width[p, f, px, :] = normalized_cumsum_pixel(data[p, f, px])

    return width


@numba.njit
def normalized_cumsum_pixel(pixel):
    """Calculate the normalized cumulative sum of the data.

    Parameters
    ----------
    data : NDArray
        Data to calculate the normalized cumulative sum of.


    Returns
    -------
    NDArray
        Normalized cumulative sum of the data.
    """
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    pixel /= np.max(pixel)
    return pixel


def make_dummy_data(
    model: str = "esr14n",
    n_freqs: int = 50,
    scan_dimensions: Union[Tuple[int, int], None] = (120, 190),
    shift: float = 0,
    noise: float = 0,
) -> Tuple[NDArray, NDArray, NDArray]:
    model = model.upper()

    if model not in models.IMPLEMENTED:
        raise ValueError(f"Unknown model {model}")

    model_func = models.IMPLEMENTED[model]["func"]
    n_parameter = len(models.IMPLEMENTED[model]["params"])

    f0 = np.linspace(2.84, 2.85, n_freqs)
    f1 = np.linspace(2.89, 2.9, n_freqs)
    c0 = np.mean(f0)
    c1 = np.mean(f1)
    width = 0.0001
    contrast = 0.01
    offset = 0.0
    parameter = {
        4: [width, contrast, offset],
        5: [width, contrast, contrast, offset],
        6: [width, contrast, contrast, contrast, offset],
    }

    p = np.ones((scan_dimensions[0] * scan_dimensions[1], n_parameter))
    p00 = make_parameter_array(c0 - shift, n_parameter, p, parameter)
    p10 = make_parameter_array(c0 + shift, n_parameter, p, parameter)
    p01 = make_parameter_array(c1 - shift, n_parameter, p, parameter)
    p11 = make_parameter_array(c1 + shift, n_parameter, p, parameter)

    m00 = model_func(f0, p00)
    m10 = model_func(f0, p10)
    m01 = model_func(f1, p01)
    m11 = model_func(f1, p11)

    mall = np.stack([np.stack([m00, m01]), np.stack([m10, m11])])

    if noise:
        mall += np.random.normal(0, noise, size=mall.shape)

    pall = np.stack([np.stack([p00, p01]), np.stack([p10, p11])])
    f_ghz = np.stack([f0, f1])
    return mall, f_ghz, pall


def make_parameter_array(
    c0: float, n_params: int, p: NDArray, params: Dict[int, List[float]]
) -> np.ndarray:
    """Make a parameter array for a given center frequency.

    :param c0: float
        center frequency
    :param n_params: int
        number of parameters
    :param p: np.array
        parameter array
    :param params: dict
        parameter dictionary
    :return: np.array
        parameter array
    """
    p00 = p.copy()
    p00[:, 0] *= c0 - 0.0001
    p00[:, 1:] *= params[n_params]
    return p00


def write_test_qdmio_file(path: Union[str, os.PathLike], **kwargs: Any) -> None:
    path = Path(path)
    path.mkdir(exist_ok=True)

    data, freq, true_parameter = make_dummy_data(n_freqs=100, model="esr15n", **kwargs)

    data = np.swapaxes(data, -1, -2)

    for i in range(2):
        out = {
            "disp1": data[i, 0, 0],
            "disp2": data[i, 0, 0],
            "freqList": np.concatenate(freq) * 1e9,
            "imgNumCols": 190,
            "imgNumRows": 120,
            "imgStack1": data[i, 0],
            "imgStack2": data[i, 1],
            "numFreqs": 100,
        }

        savemat(os.path.join(path, f"run_0000{i}.mat"), out)

    led = pd.DataFrame(data=np.random.normal(0, 1, size=(120, 190)))

    led.to_csv(
        os.path.join(path, "LED.csv"),
        header=False,
        index=False,
        sep="\t",
    )
    laser = pd.DataFrame(data=np.random.normal(0, 1, size=(120, 190)))

    laser.to_csv(
        os.path.join(path, "laser.csv"),
        header=False,
        index=False,
        sep="\t",
    )
