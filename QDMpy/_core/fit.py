import logging
import os.path
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import pandas as pd
from numba import float64, guvectorize


import QDMpy
from QDMpy._core.models import guess_model

if QDMpy.PYGPUFIT_PRESENT:
    import pygpufit.gpufit as gf

from scipy.io import savemat

from QDMpy._core import models

UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}
CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}


class Fit:
    """Fitting class to calculate the fit parameters and the fit results from a QDM measurement."""

    LOG = logging.getLogger(__name__)

    def __init__(
        self,
        data: np.ndarray,
        frequencies: np.ndarray,
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
            f"Initializing Fit instance with data: {data.shape} at {frequencies.shape} frequencies."
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
        )  # structure is: type: [float(vmin), float(vmax), str(constraint_type), str(unit)]

        self.estimator_id = ESTIMATOR_ID[QDMpy.SETTINGS["fit"]["estimator"]]  # 0 for LSE, 1 for MLE

    def __repr__(self) -> str:
        """Return a string representation of the fit."""
        return f"Fit(data: {self.data.shape},f: {self.f_ghz.shape}, model:{self.model_name})"

    @property
    def data(self) -> np.ndarray:
        """Get the data to fit.

        Returns:
            np.ndarray: The data to fit.
        """
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Set the data to fit.

        Args:
            data (np.ndarray): A 3D array of the data to fit.

        Returns:
            None
        """
        self.LOG.info("Data changed, fits need to be recalculated!")
        if np.all(self._data == data):
            return
        self._data = data
        self._initial_parameter = None
        self._reset_fit()

    ### MODEL RELATED METHODS ###
    def guess_model_name(self, check=False) -> str:
        """Guess the model name from the data.

        Returns:
          str: Name of the model.
        """

        data = np.median(self.data, axis=2)
        n_peaks, doubt, peaks = guess_model(data, check=check)

        if doubt:
            self.LOG.warning(
                "Doubt on the diamond type. Check using `guess_model(True)` and set manually if incorrect."
            )

        model = [mdict for m, mdict in models.IMPLEMENTED.items() if mdict["n_peaks"] == n_peaks][0]

        self.LOG.info(
            f"Guessed diamond type: {n_peaks} peaks -> {model['func_name']} ({model['name']})"
        )
        return model["func_name"]

    @property
    def model_func(self) -> Callable:
        """Return the model function."""
        return self._model["func"]

    @property
    def model_params(self) -> List[str]:
        """Return the model parameters."""
        return self._model["params"]

    @property
    def model(self) -> dict:
        """Return the model dictionary."""
        return self._model

    @property
    def model_name(self) -> Callable:
        """ """
        return self._model["func_name"]

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """Setter for the model name.

        Args:
          model_name: Name of the model to fit.
        """
        if model_name.upper() not in models.IMPLEMENTED:
            raise ValueError(
                f"Unknown model: {model_name} choose from {list(models.IMPLEMENTED.keys())}"
            )

        self._model = models.IMPLEMENTED[model_name]
        self._constraints = self._set_initial_constraints()

        self.LOG.info(
            f"Setting model to {model_name}, resetting all fit results and initial parameters."
        )
        self._reset_fit()
        self._initial_parameter: Union[np.ndarray, None] = self.get_initial_parameter()

    @property
    def model_id(self) -> int:
        """Return the model id."""
        return self._model["model_id"]

    @property
    def model_params_unique(self) -> List[str]:
        """
        Return the unique model parameters.
        Contrast is a non unique parameter, as we can not tell what peak it belongs to.
        To make it unique, a number is added to the parameter name.

        Returns:
            List[str]: List of unique model parameters.

        Example:
            >>> some_instance = SomeClass()
            >>> some_instance.model_params = ["contrast", "width", "contrast", "location"]
            >>> some_instance.model_params_unique
            ['contrast_0', 'width', 'contrast_1', 'location']
        """
        lst = []
        for v in self.model_params:
            # Check if the parameter appears more than once in the list
            if self.model_params.count(v) > 1:
                # Iterate through numbers from 0 to 9
                for n in range(10):
                    # Check if the parameter with the current number is not already in the list
                    if f"{v}_{n}" not in lst:
                        # Add the parameter with the current number to the list
                        lst.append(f"{v}_{n}")
                        break
            else:
                # Add the unique parameter to the list
                lst.append(v)
        return lst

    @property
    def n_parameter(self) -> int:
        """
        Return the number of parameters in the model.

        This method calculates the number of parameters in the model by returning the length of the
        `model_params` attribute, which is assumed to be a list of model parameters.

        Returns:
            int: The number of parameters in the model.

        Example:
            >>> model = ExampleModel([1, 2, 3, 4])
            >>> model.n_parameter
            4
        """
        return len(self.model_params)

    ### INITIAL PARAMETER RELATED METHODS ###
    @property
    def initial_parameter(self) -> Union[np.ndarray, None]:
        """
        Return the initial parameter.

        This property returns the initial parameter for the model fitting process.
        If the initial parameter is not set, it calls the `get_initial_parameter` method to generate one.

        Returns:
            Union[np.ndarray, None]: The initial parameter as a NumPy array or None if not set.

        Example:
            model = Model()
            initial_param = model.initial_parameter
        """
        # Check if the initial parameter is already set
        if self._initial_parameter is None:
            # If not, call the `get_initial_parameter` method to generate one
            self._initial_parameter = self.get_initial_parameter()

        return self._initial_parameter

    def get_initial_parameter(self) -> np.ndarray:
        """
        Constructs an initial guess for the fit.

        This method creates a NumPy array with initial parameter guesses for the model fitting process
        by iterating over the model's parameters and calling their respective guess methods.

        Returns:
            np.ndarray: A NumPy array containing the initial parameter guesses.

        Example:
            model = Model()
            initial_param_guess = model.get_initial_parameter()
        """
        fit_parameter = []

        # Iterate over the model's parameters
        for p in self.model_params:
            # Call the corresponding guess method for each parameter
            param = getattr(self, f"_guess_{p}")()
            # Add the guessed parameter to the list
            fit_parameter.append(param)

        # Stack the guessed parameters along the last dimension of the last guessed parameter
        fit_parameter = np.stack(fit_parameter, axis=fit_parameter[-1].ndim)

        # Return the guessed parameters as a contiguous NumPy array with dtype float32
        return np.ascontiguousarray(fit_parameter, dtype=np.float32)

    ### COSTRAINTS RELATED METHODS ###
    def _set_initial_constraints(self) -> Dict[str, List[Any]]:
        """
        Get the default constraints dictionary for the fit.

        This method extracts the constraints from the QDMpy.SETTINGS["fit"]["constraints"] dictionary,
        and initializes the default constraints for each unique model parameter.

        Returns:
            A dictionary with model parameters as keys, and lists of constraints as values.
            Each list contains the following constraints:
            - Minimum value
            - Maximum value
            - Type of constraint
            - Unit

        Example:
            Assuming the following constraints and unique model parameters in the SETTINGS dictionary:
            {
                "a_min": 0,
                "a_max": 10,
                "a_type": "uniform",
                "b_min": -10,
                "b_max": 10,
                "b_type": "normal",
            }
            model_params_unique = ["a_param", "b_param"]

            The output will be:
            {
                "a_param": [0, 10, "uniform", "meters"],
                "b_param": [-10, 10, "normal", "seconds"],
            }
        """
        # Extract constraints from the SETTINGS dictionary
        constraints = QDMpy.SETTINGS["fit"]["constraints"]

        # Initialize an empty dictionary for storing default constraints
        defaults = {}
        for value in self.model_params_unique:
            # Extract the parameter prefix from the value
            v = value.split("_")[0]
            # Assign the default constraints for the parameter
            defaults[value] = [
                constraints[f"{v}_min"],
                constraints[f"{v}_max"],
                constraints[f"{v}_type"],
                self.UNITS[v],
            ]
        return defaults

    def set_constraints(
        self,
        param: str,
        vmin: Union[float, None] = None,
        vmax: Union[float, None] = None,
        constraint_type: Union[str, None] = None,
        reset_fit: bool = True,
    ) -> None:
        """
        Set the constraints for the fit.

        Args:
          param (str): The parameter to set the constraints for.
          vmin (float, optional): The minimum value to set the constraints to. The default is None.
          vmax (float, optional): The maximum value to set the constraints to. The default is None.
          constraint_type (str, optional): The bound type to set the constraints to. The default is None.
          reset_fit (bool, optional): Whether to reset the fit results. The default is True.

        Returns:
            None

        Raises:
            ValueError: If the constraint_type does not exist.

        Example:
            >>> model.set_constraints("param1", vmin=0, vmax=1, constraint_type="bounded")
        """

        # Validate the constraint_type provided
        if constraint_type is not None and constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(
                f"Unknown constraint type: {constraint_type} choose from {CONSTRAINT_TYPES}"
            )
        elif constraint_type is None:
            constraint_type = self._constraints[param][2]

        # Check if constraint_type is an integer and use it as an index of CONSTRAINT_TYPES
        if isinstance(constraint_type, int):
            constraint_type = CONSTRAINT_TYPES[constraint_type]

        # If a non-unique parameter is given, use it for all instances of the parameter
        # (i.e. contrast -> contrast_0, contrast_1, ...)
        if param == "contrast" and self.model_params_unique != self.model_params:
            for contrast in [v for v in self.model_params_unique if "contrast" in v]:
                self.set_constraints(
                    contrast, vmin=vmin, vmax=vmax, constraint_type=constraint_type
                )
        # Otherwise set the specific constraint for the given parameter
        else:
            self.LOG.info(
                f"Setting constraints for << {param} >> ({vmin}, {vmax}) with {constraint_type}"
            )
            self._constraints[param] = [
                vmin,
                vmax,
                constraint_type,
                UNITS[param.split("_")[0]],
            ]

        # Reset the fit results if requested
        if reset_fit:
            self._reset_fit()

    def set_free_constraints(self) -> None:
        """
        Set all unique model parameters to have free constraints.

        This method iterates through all unique parameters in the model and sets their
        constraints to be free. This allows each parameter to be adjusted freely during
        optimization, which may be useful when you want to explore a wide range of
        possible parameter values.

        Example:
            model = Model()
            model.set_free_constraints()  # All unique model parameters are now free.

        """
        # Iterate through all unique model parameters
        for param in self.model_params_unique:
            # Set the constraints for the current parameter to be free
            self.set_constraints(
                param,
                constraint_type="FREE",
                reset_fit=param == self.model_params_unique[-1],
            )
            # The reset_fit flag is set to True only for the last parameter in the list,
            # which resets the fit status after all constraints have been updated.

    @property
    def constraints(self) -> Dict[str, List[Union[float, str]]]:
        """
        Return a dictionary of the constraints.

        The constraints are stored as a dictionary where the keys are constraint names and the values are lists of constraint values.
        Constraint values can be either floats or strings.

        Example:
            >>> example_instance = Fit()
            >>> example_instance.constraints
            {'constraint1': [1.0, 2.0], 'constraint2': ['a', 'b']}

        Returns:
            dict: A dictionary of constraint names as keys and lists of constraint values.
        """
        # Return the private _constraints attribute
        return self._constraints

    def get_constraints_array(self, n_pixel: int) -> np.ndarray:
        """
        This function generates an array of constraints by repeating the given constraints for each unique model
        parameter for a specified number of pixels (n_pixel).

        Args:
          n_pixel (int): The number of pixels for which to generate constraints.

        Returns:
          np.ndarray: A NumPy array of shape (n_pixel, 2 * len(self.model_params_unique)) containing the constraints.

        Example:
          example_instance = ExampleClass()
          n_pixel = 3
          constraints_array = example_instance.get_constraints_array(n_pixel)
          print(constraints_array)
          # Output: [[ 0 10 -5  5]
          #          [ 0 10 -5  5]
          #          [ 0 10 -5  5]]
        """
        # Initialize an empty list to store constraints
        constraints_list: List[float] = []

        # Iterate over unique model parameters and add constraints for each parameter
        for k in self.model_params_unique:
            constraints_list.extend((self._constraints[k][0], self._constraints[k][1]))

        # Tile the constraints list to generate the constraints array with shape (n_pixel, 2 * len(self.model_params_unique))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self) -> np.np.ndarray:
        """
        Returns the indices of the constraint types for each model parameter in self.model_params_unique.

        Args:
            None

        Returns:
            np.np.ndarray: An integer NumPy array containing the indices of constraint types.

        Example:
            Assume CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"] and a class instance has
            self.model_params_unique = ['a', 'b'] and self._constraints = {'a': (1, 2, 'FREE'), 'b': (3, 4, 'LOWER_UPPER')}

            >>> obj = Fit()
            >>> constraint_types_indices = obj.get_constraint_types()
            >>> print(constraint_types_indices)
            [0 3]
        """
        # Create a list of the indices of constraint types for each model parameter
        fit_bounds = [
            CONSTRAINT_TYPES.index(self._constraints[k][2]) for k in self.model_params_unique
        ]

        # Return the fit_bounds list as an integer NumPy array
        return np.array(fit_bounds).astype(np.int32)

    # parameters
    @property
    def parameter(self) -> np.np.ndarray:
        """
        Returns the `_fit_results` attribute as the parameter of the class instance.

        Args:
            None

        Returns:
            np.np.ndarray: The NumPy array containing the fit results.

        Example:
            Assume a class instance has self._fit_results = np.array([1.2, 3.4, 5.6])

            >>> obj = Fit()
            >>> param = obj.parameter
            >>> print(param)
            [1.2 3.4 5.6]
        """
        return self._fit_results

    def get_param(self, param: str) -> Union[np.ndarray, None]:
        """
        Get the value of a parameter reshaped to the image dimensions.

        Args:
            param (str): The parameter name to retrieve. Acceptable values include:
                         - "chi2", "chi_squares", "chi_squared", "chi"
                         - "number_iterations", "num", "num_iter"
                         - "states", "state"
                         - "mean_contrast"

        Returns:
            Union[np.ndarray, None]: The requested parameter reshaped to the image dimensions, or None if not found.

        Raises:
            NotImplementedError: If no fit has been performed yet.

        Example:
            >>> example = ExampleClass()
            >>> example.fitted = True  # Assuming a fit has been performed
            >>> result = example.get_param("chi2")
            >>> print(result)
        """
        if not self.fitted:
            raise NotImplementedError("No fit has been performed yet. Run fit_odmr().")

        # Determine which parameter value to return based on the input string
        if param in {"chi2", "chi_squares", "chi_squared", "chi"}:
            return self._chi_squares
        elif param in {"number_iterations", "num", "num_iter"}:
            return self._number_iterations
        elif param in {"states", "state"}:
            return self._states

        idx = self._param_idx(param)

        # Calculate the mean contrast if requested
        if param == "mean_contrast":
            return np.mean(self._fit_results[:, :, :, idx], axis=-1)  # type: ignore[index]

        # Return the requested parameter value
        return self._fit_results[:, :, :, idx]  # type: ignore[index]

    def _param_idx(self, parameter: str) -> List[int]:
        """
        Get the indices of the specified fitted parameter.

        Args:
            parameter (str): The name of the fitted parameter. Acceptable values include:
                             - "resonance", "res"
                             - "center"
                             - "mean_contrast"
                             - "contrast"

        Returns:
            List[int]: A list of indices of the specified fitted parameter.

        Raises:
            ValueError: If the provided parameter is unknown.

        Example:
            >>> example = ExampleClass()
            >>> indices = example._param_idx("resonance")
            >>> print(indices)
        """
        # Map the input parameter to its canonical name
        if parameter in {"resonance", "res"}:
            parameter = "center"
        if parameter == "mean_contrast":
            parameter = "contrast"

        # Get the indices of the specified parameter in self.model_params
        idx = [i for i, v in enumerate(self.model_params) if v == parameter]

        # If not found, search in self.model_params_unique
        if not idx:
            idx = [i for i, v in enumerate(self.model_params_unique) if v == parameter]

        # Raise a ValueError if the parameter is still not found
        if not idx:
            raise ValueError(f"Unknown parameter: {parameter}")

        return idx

    def _guess_center(self) -> np.ndarray:
        """
        Guess the center of the ODMR spectra.

        Returns:
            np.ndarray: An array representing the estimated center frequencies of the ODMR spectra.

        Example:
            >>> example = ExampleClass()
            >>> center = example._guess_center()
            >>> print(center)
        """
        # Use the 'guess_center' function to estimate the center frequencies
        center = guess_center(self.data, self.f_ghz)

        # Log the shape of the estimated center frequencies
        self.LOG.debug(f"Guessing center frequency [GHz] of ODMR spectra {center.shape}.")

        return center

    def _guess_contrast(self) -> np.ndarray:
        """
        Guess the contrast of the ODMR spectra.

        Returns:
            np.ndarray: An array representing the estimated contrast of the ODMR spectra.

        Example:
            >>> example = ExampleClass()
            >>> contrast = example._guess_contrast()
            >>> print(contrast)
        """
        # Use the 'guess_contrast' function to estimate the contrast
        contrast = guess_contrast(self.data)

        # Log the shape of the estimated contrast
        self.LOG.debug(f"Guessing contrast of ODMR spectra {contrast.shape}.")

        return contrast

    def _guess_width(self) -> np.ndarray:
        """
        Guess the width of the ODMR spectra.

        Returns:
            np.ndarray: An array representing the estimated width of the ODMR spectra.

        Raises:
            ValueError: If n_peaks is not 1, 2, or 3.

        Example:
            >>> example = ExampleClass()
            >>> width = example._guess_width()
            >>> print(width)
        """
        # Set detection thresholds based on the number of peaks
        if self._model["n_peaks"] == 1:
            vmin, vmax = 0.3, 0.7
        elif self._model["n_peaks"] == 2:
            vmin, vmax = 0.4, 0.6
        elif self._model["n_peaks"] == 3:
            vmin, vmax = 0.35, 0.65
        else:
            raise ValueError("n_peaks must be 1, 2 or 3")

        # Use the 'guess_width' function to estimate the width
        width = guess_width(self.data, self.f_ghz, vmin, vmax)

        # Log the shape of the estimated width
        self.LOG.debug(f"Guessing width of ODMR spectra {width.shape}.")

        return width

    def _guess_offset(self) -> np.ndarray:
        """
        Guess the offset from 0 of the ODMR spectra. Usually, this is 1.

        Returns:
            np.ndarray: An array representing the estimated offset of the ODMR spectra.

        Example:
            >>> example = ExampleClass()
            >>> offset = example._guess_offset()
            >>> print(offset)
        """
        # Get the shape of the input data
        n_pol, n_frange, n_pixel, _ = self.data.shape

        # Create a zero-filled array with the same shape as the input data
        offset = np.zeros((n_pol, n_frange, n_pixel))

        # Log the shape of the estimated offset
        self.LOG.debug(f"Guessing offset {offset.shape}")

        return offset

    ### fitting related methods ###
    def _reset_fit(self) -> None:
        """
        Reset the fit results.

        Example:
            >>> example = ExampleClass()
            >>> example._reset_fit()
        """
        # Reset fit-related attributes
        self._fitted = False
        self._fit_results = None
        self._states = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

        # Log that the fit results have been reset
        self.LOG.info("Fit results have been reset.")

    @property
    def fitted(self) -> bool:
        """
        Check if a fit has been performed.

        Returns:
            bool: True if a fit has been performed, False otherwise.

        Example:
            >>> example = ExampleClass()
            >>> result = example.fitted
            >>> print(result)
        """
        return self._fitted

    def fit_odmr(self, refit=False) -> None:
        """
        Perform a fit on the ODMR data.

        Args:
            refit (bool, optional): If True, the function will reset the fit results before performing a new fit. Defaults to False.

        Example:
            >>> example = ExampleClass()
            >>> example.fit_odmr(refit=True)
        """
        # Check if a fit has already been performed
        if self._fitted and not refit:
            self.LOG.debug("Already fitted")
            return

        # Reset the fit results if 'refit' is True
        if self.fitted and refit:
            self._reset_fit()
            self.LOG.debug("Refitting the ODMR data")

        # Fit each frange separately
        for irange in np.arange(0, self.data.shape[1]):
            # Log the current frange being fitted
            self.LOG.info(
                f"Fitting frange {irange} from {self.f_ghz[irange].min():5.3f}-{self.f_ghz[irange].max():5.3f} GHz"
            )

            # Perform the fit on the current frange
            results = self.fit_frange(
                self.data[:, irange],
                self.f_ghz[irange],
                self.initial_parameter[:, irange],
            )
            results = self.reshape_results(results)

            # Update the fit results
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
                self._number_iterations = np.stack((self._number_iterations, results[3]))
                self._execution_time = np.stack((self._execution_time, results[4]))

            # Log the time it took to fit the current frange
            self.LOG.info(f"        finished in {results[4]:.2f} seconds")

        # Swap the first two axes of the fit results to match the shape of the input data
        self._fit_results = np.swapaxes(self._fit_results, 0, 1)  # type: ignore[call-overload]

        # Set the 'fitted' attribute to True
        self._fitted = True

    def fit_frange(
        self, data: np.ndarray, freq: np.ndarray, initial_parameters: np.ndarray
    ) -> List[np.ndarray]:
        """
        Perform a fit on a single frequency range of the ODMR data.

        Args:
            data (np.ndarray): Data for one frequency range, to be fitted. Array of size (n_pol, n_pixel, n_freqs) of the ODMR data.
            freq (np.ndarray): Array of size (n_freqs) of the frequencies.
            initial_parameters (np.ndarray): Initial guess for the fit, an array of size (n_pol * n_pixel, 2 * n_param) of the initial parameters.

        Returns:
            List[np.ndarray]: A list of arrays containing the fit results, including:
                - results: array of size (n_pol*n_pixel, n_param) of the fitted parameters
                - states: array of size (n_pol*n_pixel) of the fit states (i.e. did the fit work)
                - chi_squares: array of size (n_pol*n_pixel) of the chi squares
                - number_iterations: array of size (n_pol*n_pixel) of the number of iterations
                - execution_time: execution time

        Example:
            >>> example = ExampleClass()
            >>> data = np.random.rand(2, 3, 10)
            >>> freq = np.linspace(1, 10, 10)
            >>> initial_parameters = np.random.rand(6, 4)
            >>> results = example.fit_frange(data, freq, initial_parameters)
        """
        # Reshape the data and initial parameters to the appropriate dimensions
        n_pol, n_pix, n_freqs = data.shape
        data = data.reshape((-1, n_freqs))
        initial_parameters = initial_parameters.reshape((-1, self.n_parameter))

        # Get the number of pixels in the data
        n_pixel = data.shape[0]

        # Get the constraint arrays and types
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        # Perform the fit using the 'fit_constrained' function
        results = gf.fit_constrained(
            data=np.ascontiguousarray(data, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(initial_parameters, dtype=np.float32),
            weights=None,
            model_id=self.model_id,
            max_number_iterations=QDMpy.SETTINGS["fit"]["max_number_iterations"],
            tolerance=QDMpy.SETTINGS["fit"]["tolerance"],
        )

        return list(results)

    def reshape_results(self, results: List[np.ndarray]) -> List[np.ndarray]:
        """
        Reshape the results from the fit_constrained function into the correct shape.

        Args:
            results (List[np.ndarray]): A list of arrays containing the fit results, including:
                - parameters: array of size (n_pol*n_pixel, n_param) of the fitted parameters
                - states: array of size (n_pol*n_pixel) of the fit states (i.e. did the fit work)
                - chi_squares: array of size (n_pol*n_pixel) of the chi squares
                - number_iterations: array of size (n_pol*n_pixel) of the number of iterations
                - execution_time: execution time

        Returns:
            List[np.ndarray]: A list of arrays containing the reshaped fit results, including:
                - parameters: array of size (n_pol, n_pixel, n_param) of the fitted parameters
                - states: array of size (n_pol, n_pixel) of the fit states (i.e. did the fit work)
                - chi_squares: array of size (n_pol, n_pixel) of the chi squares
                - number_iterations: array of size (n_pol, n_pixel) of the number of iterations
                - execution_time: execution time

        Example:
            >>> example = ExampleClass()
            >>> results = [np.random.rand(6, 4), np.random.rand(6), np.random.rand(6), np.random.rand(6), 5.0]
            >>> reshaped_results = example.reshape_results(results)
        """
        for i in range(len(results)):
            # Skip float values (i.e. execution time)
            if isinstance(results[i], float):
                continue
            results[i] = self.reshape_result(results[i])
        return results

    def reshape_result(self, result: np.ndarray) -> np.ndarray:
        """
        Reshape the results to the original shape of (n_pol, n_pixel, -1).

        Args:
            result (np.ndarray): Array of size (n_pol * n_pixel, -1) of the fitted parameters.

        Returns:
            np.ndarray: Array of size (n_pol, n_pixel, -1) of the fitted parameters.

        Example:
            >>> example = ExampleClass()
            >>> result = np.random.rand(6, 4)
            >>> reshaped_result = example.reshape_result(result)
        """
        # Get the dimensions of the data
        n_pol, n_pix, _ = self.data[0].shape

        # Reshape the result to the original shape
        result = result.reshape((n_pol, n_pix, -1))

        # Remove any extra dimensions
        return np.squeeze(result)


@numba.njit(parallel=True)
def guess_contrast(data: np.ndarray) -> np.ndarray:
    """
    Calculate the contrast of each ODMR data point in a 3D NumPy array.

    The contrast is calculated using the `guess_contrast_pixel` function
    on each 2D slice along the third axis of the input array.

    Args:
      data (np.ndarray): A 3D NumPy array representing ODMR data.

    Returns:
      np.ndarray: A 3D NumPy array with the same shape as the input array,
               where each element represents the contrast of the corresponding
               data point.

    Example:
      >>> data = np.random.rand(2, 3, 4)
      >>> contrast_data = guess_contrast(data)
      >>> print(contrast_data.shape)
      (2, 3, 4)
    """
    amp = np.zeros(data.shape[:-1])

    for i, j in np.ndindex(data.shape[0], data.shape[1]):
        for p in numba.prange(data.shape[2]):
            amp[i, j, p] = guess_contrast_pixel(data[i, j, p])
    return amp


@numba.njit()
def guess_contrast_pixel(data: np.ndarray) -> float:
    """
    Calculate the contrast of an image represented by a 2D NumPy array.

    The contrast is calculated as the absolute difference between the maximum
    and minimum pixel values, divided by the maximum pixel value.

    Args:
      data (np.ndarray): A 2D NumPy array representing the pixel values of an image.

    Returns:
      float: The contrast of the image.

    Example:
      >>> image = np.array([[10, 20], [30, 40]])
      >>> contrast = guess_contrast_pixel(image)
      >>> print(contrast)
      0.75
    """
    mx = np.nanmax(data)
    mn = np.nanmin(data)
    return np.abs((mx - mn) / mx) if mx != 0 else 0.0


@numba.njit(parallel=True, fastmath=True)
def guess_center(data: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Guess the center frequency of ODMR data.

    Args:
      data: np.array
    data to guess the center frequency from
      freq: np.array
    frequency range of the data

    Returns:
      center frequency of the data

    """
    # center frequency
    center = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        for px in numba.prange(data.shape[2]):
            center[p, f, px] = guess_center_pixel(data[p, f, px], freq[f])
    return center


@numba.njit(fastmath=True)
def guess_center_pixel(pixel: np.ndarray, freq: np.ndarray) -> float:
    """
    Guess the center frequency of a single frequency range for a given pixel data.

    Args:
      pixel (np.ndarray): A 1D NumPy array representing the pixel data.
      freq (np.ndarray): A 1D NumPy array representing the frequency range of the data.

    Returns:
      float: The estimated center frequency of the data.

    Example:
      >>> pixel_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
      >>> freq_data = np.array([1, 2, 3, 4, 5])
      >>> center_frequency = guess_center_pixel(pixel_data, freq_data)
      >>> print(center_frequency)
      3.0
    """
    pixel = normalized_cumsum(pixel)
    idx = np.argmin(np.abs(pixel - 0.5))
    return freq[idx]


# @numba.njit(parallel=True, fastmath=True)
def guess_width(data: np.ndarray, f_ghz: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Guess the width of a ODMR resonance peaks.

    Args:
      data: np.array
    data to guess the width from
      f_ghz: np.array
    frequency range of the data
      vmin: float
    minimum value of normalized cumsum to be considered
      vmax: float
    maximum value of normalized cumsum to be considered
      data: np.ndarray:
      f_ghz: np.ndarray:
      vmin: float:
      vmax: float:

    Returns:
      np.array
      width of the data

    """
    # width
    width = np.zeros(data.shape[:-1])
    for p, f in np.ndindex(data.shape[0], data.shape[1]):
        freq = f_ghz[f]
        for px in numba.prange(data.shape[2]):
            width[p, f, px] = guess_width_pixel(data[p, f, px], freq, vmin, vmax)

    return width


@numba.njit(fastmath=True)
def guess_width_pixel(pixel: np.ndarray, freq: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Guess the width of a single frequency range.

    Args:
      data: np.array
    data to guess the width from
      freq: np.array
    frequency range of the data
      pixel: np.ndarray:
      freq: np.ndarray:
      vmin: float:
      vmax: float:

    Returns:
      np.array
      width of the data

      Raises ValueError if the number of peaks is not 1, 2 or 3.

    """
    pixel = normalized_cumsum_pixel(pixel)
    lidx = np.argmin(np.abs(pixel - vmin))
    ridx = np.argmin(np.abs(pixel - vmax))
    return freq[lidx] - freq[ridx]


@numba.njit(parallel=True, fastmath=True)
def normalized_cumsum(data: np.ndarray) -> np.ndarray:
    """Calculate the normalized cumsum of the data.

    Args:
      data: np.array
        data to guess the width from

    Returns:
      normalized (0-1) cumsum of the data

    """
    data_shape = data.shape
    # reshape for faster processing
    data = data.reshape(-1, data_shape[-1])

    # initialize array
    csum = np.zeros(data.shape)
    for px in numba.prange(data.shape[0]):
        csum[px, :] = normalized_cumsum_pixel(data[px])
    return csum.reshape(data_shape)


#


@numba.njit
def normalized_cumsum_pixel(pixel: np.ndarray) -> np.ndarray:
    """Calculate the normalized cumulative sum of the data.

    Args:
      data(np.ndarray): Data to calculate the normalized cumulative sum of.
      pixel:

    Returns:


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
      model: str:  (Default value = "esr14n")
      n_freqs: int:  (Default value = 50)
      scan_dimensions: Union[Tuple[int:
      int]:
      None]:  (Default value = (120)
      190):
      shift: float:  (Default value = 0)
      noise: float:  (Default value = 0)

    Returns:

    """
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
    center_freq: float, n_parameter: int, param_arr: np.ndarray, param_dict: Dict[int, List[float]]
) -> np.np.ndarray:
    """
    Make a parameter array for a given center frequency.

    Args:
        center_freq (float): The center frequency.
        num_params (int): The number of parameters.
        param_arr (np.ndarray): The parameter array.
        param_dict (Dict[int, List[float]]): The parameter dictionary.

    Returns:
        np.np.ndarray: The parameter array.

    Example:
        >>> center_freq = 1.0
        >>> num_params = 4
        >>> param_arr = np.random.rand(6, 2 * num_params)
        >>> param_dict = {num_params: [0.1, 0.1, 0.1, 0.1]}
        >>> parameter_array = make_parameter_array(center_freq, num_params, param_arr, param_dict)
    """
    # Create a copy of the parameter array
    new_param_arr = param_arr.copy()

    # Update the center frequency value
    new_param_arr[:, 0] *= center_freq - 0.0001

    # Scale the other parameters
    new_param_arr[:, 1:] *= param_dict[n_parameter]

    return new_param_arr


def write_test_qdmio_file(path: Union[str, os.PathLike], **kwargs: Any) -> None:
    """
    Write a test QDM file to the given path.

    Args:
        path (Union[str, os.PathLike]): Path to write the file to.
        **kwargs (Any): Optional arguments for make_dummy_data.

    Returns:
        None

    Example:
        >>> write_test_qdmio_file('test_files/', model='esr15n', width=0.2, amplitude=1.0, snr=10)
    """

    # Create the directory if it doesn't exist
    path = Path(path)
    path.mkdir(exist_ok=True)

    # Generate dummy data with make_dummy_data and swap axes to match QDM file format
    data, freq, true_parameter = make_dummy_data(n_freqs=100, model="esr15n", **kwargs)
    data = np.swapaxes(data, -1, -2)

    # Write two QDM files with different data
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

    # Write the LED and laser files with random data
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


if __name__ == "__main__":
    from QDMpy._core.qdm import QDM

    q = QDM.from_qdmio("/Users/mike/github/QDMpy/tests/data/utrecht_test_data")

    o = q.odmr
    # aux = guess_center_pixel(o.data[0], o.f_ghz)

    # q.fit_odmr()
