import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import pyqdm
from pyqdm import plotting, pygpufit_present
from pyqdm.core import fitting
from pyqdm.core.odmr import ODMR
from pyqdm.exceptions import CantImportError, WrongFileNumber
from pyqdm.utils import get_image, idx2rc, rc2idx

if pygpufit_present:
    import pygpufit.gpufit as gf

import pandas as pd
from scipy.io import savemat


class QDM:
    DIAMOND_TYPES = ["NaN", "MISC.", "N15", "N14"]
    FIT_TYPES = ["GAUSS_1D", "ESRSINGLE", "ESR15N", "ESR14N"]
    POLARITIES = ["positive", "negative"]
    FRANGES = ["high", "low"]

    FIT_PARAMETER = {
        "GAUSS_1D": ["contrast", "center", "width", "offset"],
        "ESR14N": ["center", "width", "contrast", "contrast", "contrast", "offset"],
        "ESR15N": ["center", "width", "contrast", "contrast", "offset"],
        "ESRSINGLE": ["center", "width", "contrast", "offset"],
    }
    LOG = logging.getLogger(f"pyQDM.QDM")

    if pygpufit_present:
        CONSTRAINT_TYPES = {
            "FREE": gf.ConstraintType.FREE,
            "LOWER": gf.ConstraintType.LOWER,
            "UPPER": gf.ConstraintType.UPPER,
            "LOWER_UPPER": gf.ConstraintType.LOWER_UPPER,
        }
    else:
        CONSTRAINT_TYPES = {
            "FREE": 0,
            "LOWER": 1,
            "UPPER": 2,
            "LOWER_UPPER": 3,
        }

    BOUND_TYPES = list(CONSTRAINT_TYPES.keys())
    UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}

    # GAMMA = 0.0028 # GHz/G
    GAMMA = 28.024 / 1e6  # GHz/muT;

    def __init__(
        self,
        ODMRobj,
        light,
        laser,
        working_directory,
        diamond_type=None,
    ):

        self.LOG.info("Initializing QDM object.")
        self.LOG.info(f'Working directory: "{working_directory}"')
        self.working_directory = working_directory

        self.LOG.debug("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.debug(f"read parameter shape: data: {ODMRobj.data.shape}")
        self.LOG.debug(f"                      scan_dimensions: {ODMRobj.scan_dimensions}")
        self.LOG.debug(f"                      frequencies: {ODMRobj.frequencies.shape}")
        self.LOG.debug(f"                      n_freqs: {ODMRobj.n_freqs}")

        self.ODMRobj = ODMRobj
        self._initial_guess = None

        self._fitted = False
        self._fitting_constrains = None
        self._constraints = {}
        self._outliers = None

        self.light = light
        self.laser = laser

        self._B111 = None

        if diamond_type is None:
            diamond_type = self.guess_diamond_type()

        self.pixel_size = 4e-6  # 4 um
        self.set_diamond_type(diamond_type)

        self._check_bin_factor()

    @property
    def outliers(self):
        """
        Return the outliers mask.
        :return: ndarray of boolean
        """
        if self._outliers is None:
            self._outliers = self.detect_outliers()
        return self._outliers

    @property
    def outliers_idx(self):
        """
        Return the indices of the outliers pixels.
        Indices are in reference to the binned ODMR data.

        :return: np.array
        """
        return np.where(self.outliers)[0]

    @property
    def outliers_xy(self):
        """
        Return the xy coordinates of the outliers pixels.
        In reference to the binned ODMR data.

        :return: np.array of shape (n_outlier, 2)
        """
        y, x = self.idx2rc(self.outliers_idx)
        return np.stack([x, y], axis=1)

    def detect_outliers(self, dtype="chi_squared", method="LocalOutlierFactor", **outlier_props):
        """
        Detect outliers in the ODMR data.
        The outliers are detected using 'method'.
        The method can be either 'LocalOutlierFactor' or 'IsolationForest'.
        The LocalOutlierFactor is a sklearn method.
        The IsolationForest is a scikit-learn method.
        The method can be passed as a keyword argument.

        :param dtype:
        :param outlier_props:
        :return:
        """

        if dtype == "chi_squared":
            data = self._chi_squares
        elif dtype == "B111_remanent":
            data = self.B111_remanent.reshape(1, -1)
        elif dtype == "B111_induced":
            data = self.B111_remanent.reshape(1, -1)
        elif dtype in self._fitting_params + self._fitting_params_unique:
            data = self.get_param(dtype, reshape=False)
        else:
            raise ValueError(f"dtype {dtype} not recognized")

        if method == "LocalOutlierFactor":
            clf = LocalOutlierFactor(**outlier_props).fit(data)
        elif method == "IsolationForest":
            outlier_props["contamination"] = outlier_props.pop("contamination", 0.001)
            clf = IsolationForest(n_jobs=-1, **outlier_props)
        else:
            raise ValueError(f"Method {method} not recognized.")

        shape = data.shape
        self.LOG.debug(f"Detecting outliers in <<{dtype}>> data of shape {shape}")
        outliers = (clf.fit_predict(data.reshape(-1, 1)) + 1) / 2
        # collapse the first dimensions so that the product applies to all except the pixel dimension
        outliers = outliers.reshape(-1, shape[-1])
        outliers = ~np.product(outliers, axis=0).astype(bool)
        self.LOG.info(
            f"Outlier detection using LocalOutlierFactor of <<{dtype}>> detected {outliers.sum()} outliers pixels.\n"
            f"                                      Indixes can be accessed using 'outlier_idx' and 'outlier_xy'"
        )
        self.LOG.debug(f"returning {outliers.shape}")
        self._outliers = outliers
        return self._outliers

    @property
    def bin_factor(self):
        return self.ODMRobj.bin_factor

    def diamond_type(self):
        """
        Return the diamond type.
        """
        return self.DIAMOND_TYPES[self._diamond_type]

    def set_diamond_type(self, diamond_type):
        """
        Set the diamond type.

        :param diamond_type:
        """
        self.LOG.debug(f'Setting diamond type to "{diamond_type}"')
        self._diamond_type = {"N14": 3, "N15": 2, "SINGLE": 1}[self.DIAMOND_TYPES[diamond_type]]
        self.reset_constraints()
        self._construct_initial_guess()

    def _check_bin_factor(self):
        bin_factors = self.light.shape / self.odmr.scan_dimensions

        if np.all(self.ODMRobj._scan_dimensions != self.light.shape):
            self.LOG.warning(
                f"Scan dimensions of ODMR ({self.ODMRobj._scan_dimensions}) and LED ({self.light.shape}) are not equal. Setting pre_binfactor to {bin_factors[0]}."
            )
            # set the true bin factor
            self.ODMRobj._pre_bin_factor = bin_factors[0]
            self.ODMRobj._scan_dimensions = np.array(self.light.shape)

    @property
    def odmr(self):
        return self.ODMRobj

    @property
    def fitted(self):
        return self._fitted

    @property
    def fitted_parameter(self):
        return self._fitted_parameter

    @property
    def initial_guess(self):
        """
        Return the initial guess for the fit.
        """
        if self._initial_guess is None:
            self.LOG.debug("Initial guess is None. Calculating...")
            self._construct_initial_guess()
        return self._initial_guess

    @property
    def scan_dimensions(self):
        return self.ODMRobj.scan_dimensions

    def guess_diamond_type(self, *args):
        """
        Guess the diamond type based on the number of peaks.

        :return: diamond_type (int)
        """
        from scipy.signal import find_peaks

        indices = []

        # Find the indices of the peaks
        for p in range(self.ODMRobj.n_pol - 1):
            for f in range(self.ODMRobj.n_frange - 1):
                peaks = find_peaks(-self.ODMRobj.mean_odmr[p, f], prominence=0.003)
                indices.append(peaks[0])

        n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))

        doubt = np.std([len(idx) for idx in indices]) != 0

        if doubt:
            self.LOG.warning(
                "Doubt on the diamond type. Check using `QDMdata.guess_diamond_type('debug')` "
                "and set manually if incorrect."
            )
        if "debug" in args:
            self.LOG.debug(f"Indices of peaks: {indices}")
            n = 0
            for i in range(self.ODMRobj.mean_odmr.shape[0]):
                for j in range(self.ODMRobj.mean_odmr.shape[1]):
                    plt.plot(self.ODMRobj.mean_odmr[i, j], color=f"C{n}")
                    plt.plot(
                        indices[n],
                        self.ODMRobj.mean_odmr[i, j, indices[n]],
                        "X",
                        color=f"C{n}",
                        label=f"{n}: {len(indices[n])}",
                    )
                    n += 1
            plt.legend()
        self.LOG.info(f"Guessed diamond type: {n_peaks} peaks -> {self.DIAMOND_TYPES[n_peaks]}")

        return n_peaks

    ## from METHODS ##
    @classmethod
    def from_matlab(cls, matlab_files, dialect="QDM.io"):
        """
        Loads QDM data from a Matlab file.
        """

        match dialect:
            case "QDM.io":
                return cls.from_QDMio(matlab_files)

        raise NotImplementedError('Dialect "{}" not implemented.'.format(dialect))

    @classmethod
    def from_QDMio(cls, data_folder, diamond_type=None, **kwargs):
        """
        Loads QDM data from a Matlab file.
        """
        files = os.listdir(data_folder)
        light_files = [f for f in files if "led" in f.lower()]
        laser_files = [f for f in files if "laser" in f.lower()]
        cls.LOG.info(f"Reading {len(light_files)} led, {len(laser_files)} laser files.")

        try:
            odmr_obj = ODMR.from_qdmio(data_folder=data_folder)
            light = get_image(data_folder, light_files)
            laser = get_image(data_folder, laser_files)
        except WrongFileNumber as e:
            raise CantImportError(f'Cannot import QDM data from "{data_folder}"') from e

        return cls(odmr_obj, light=light, laser=laser, diamond_type=diamond_type, working_directory=data_folder)

    def bin_data(self, bin_factor):
        """
        Bin the data.
        :param bin_factor:
        :return:
        """
        if bin_factor == self.bin_factor:
            return
        self.odmr.bin_data(bin_factor=bin_factor)
        self.reset_constraints()  # resets the initial constraints with new dimensions
        self._initial_guess = None  # resets the initial guess

    # EXPORT METHODS ###
    def export_QDMio(self, path_to_file=None):
        """
        Export the data to a QDM.io file. This is a Matlab file named B111dataToPlot.mat. With the following variables:

        ['negDiff', 'posDiff', 'B111ferro', 'B111para', 'chi2Pos1', 'chi2Pos2', 'chi2Neg1', 'chi2Neg2', 'ledImg',
         'laser', 'pixelAlerts']
        """
        if path_to_file is None:
            path_to_file = os.path.join(
                self.working_directory, f"{self.ODMRobj._bin_factor}x{self.ODMRobj._bin_factor}Binned"
            )
            if not os.path.exists(path_to_file):
                self.LOG.warning(f"Path does not exist, creating directory {path_to_file}")
                os.makedirs(path_to_file)

        neg_diff, pos_diff = self.delta_resonance
        b111_remanent, b111_induced = self.B111
        chi_squares = self.ODMRobj.reshape_data(self._chi_squares)
        chi2_pos1, chi2_pos2 = chi_squares[0]
        chi2_neg1, chi2_neg2 = chi_squares[1]
        led_img = self.light
        laser_img = self.laser
        pixel_alerts = np.zeros(b111_remanent.shape)

        return savemat(
            os.path.join(path_to_file, "B111dataToPlot.mat"),
            {
                "negDiff": neg_diff,
                "posDiff": pos_diff,
                "B111ferro": b111_remanent,
                "B111para": b111_induced,
                "chi2Pos1": chi2_pos1,
                "chi2Pos2": chi2_pos2,
                "chi2Neg1": chi2_neg1,
                "chi2Neg2": chi2_neg2,
                "ledImg": led_img,
                "laser": laser_img,
                "pixelAlerts": pixel_alerts,
                "bin_factor": self.bin_factor,
                "pyqdm_version": pyqdm.__version__,
            },
        )

    # FITTING ##
    @property
    def _fitting_params(self):
        return self.FIT_PARAMETER[self.FIT_TYPES[self._diamond_type]]

    def _guess_center(self):
        """
        Guess the center of the ODMR spectra.
        """
        center = fitting.guess_center(self.ODMRobj.data, self.ODMRobj.f_ghz)
        self.LOG.debug(f"Guessing center frequency [GHz] of ODMR spectra {center.shape}.")
        return center

    def _guess_contrast(self):
        """
        Guess the contrast of the ODMR spectra.
        """
        contrast = fitting.guess_contrast(self.ODMRobj.data)
        self.LOG.debug(f"Guessing contrast of ODMR spectra {contrast.shape}.")
        # np.ones((self.n_pol, self.n_frange, self.n_pixel)) * 0.03
        return contrast

    def _guess_width(self):
        """
        Guess the width of the ODMR spectra.
        """
        width = fitting.guess_width(self.ODMRobj.data, self.ODMRobj.f_ghz)
        self.LOG.debug(f"Guessing width of ODMR spectra {width.shape}.")
        return width

    def _guess_offset(self):
        """
        Guess the offset from 0 of the ODMR spectra. Usually this is 1
        """
        offset = np.zeros((self.ODMRobj.n_pol, self.ODMRobj.n_frange, self.ODMRobj.n_pixel))
        self.LOG.debug(f"Guessing offset {offset.shape}")
        return offset

    def _construct_initial_guess(self, **kwargs):
        """
        Constructs an initial guess for the fit.
        """
        fit_parameter = []

        for p in self._fitting_params:
            param = getattr(self, f"_guess_{p}")()
            fit_parameter.append(param)

        fit_parameter = np.stack(fit_parameter, axis=fit_parameter[-1].ndim)
        self._initial_guess = np.ascontiguousarray(fit_parameter, dtype=np.float32)

    def correct_glob_fluorescence(self, glob_fluo):
        """
        Corrects the global fluorescence.
        """
        self.LOG.debug(f"Correcting global fluorescence {glob_fluo}.")
        self.ODMRobj.correct_glob_fluorescence(glob_fluo)
        self._construct_initial_guess()

    @property
    def global_factor(self):
        return self.ODMRobj.global_factor

    @property
    def constraints(self):
        """
        Return the constraints for the fit.
        """
        return self._constraints

    def set_free_constraints(self):
        """
        Set the constraints to be free.
        """
        for param in set(self._fitting_params):
            self.set_constraints(param, bound_type="FREE")

    @property
    def constraints_array(self):
        """
        Return the constraints as an array (pixel, 2*fitting_parameters).

        :return: np.array
        """
        constraints = np.zeros((self.ODMRobj.n_pixel * 2, 2 * len(self._fitting_params)), dtype=np.float32)

        constraints_list = []
        for k in self._fitting_params_unique:
            constraints_list.extend((self._constraints[k][0], self._constraints[k][1]))
        constraints[:, :] = constraints_list
        return constraints

    @property
    def constraint_types(self):
        """
        Return the constraint types.
        :return: np.array
        """
        fit_bounds = [self.BOUND_TYPES.index(self._constraints[k][2]) for k in self._fitting_params_unique]

        return np.array(fit_bounds).astype(np.int32)

    def set_constraints(self, param, values=None, bound_type=None):
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
            bound_type = self.BOUND_TYPES[bound_type]
        if bound_type is not None and bound_type not in self.BOUND_TYPES:
            raise ValueError(f"Unknown constraint type: {bound_type} choose from {self.BOUND_TYPES}")

        if param == "contrast":
            for contrast in [v for v in self._fitting_params_unique if "contrast" in v]:
                self.set_constraints(contrast, values=values, bound_type=bound_type)
        else:
            self.LOG.debug(f"Setting constraints for {param}: {values} with {bound_type}")
            self._constraints[param] = [values[0], values[1], bound_type, self.UNITS[param.split("_")[0]]]

    def reset_constraints(self):
        """
        Reset the constraints to the default values.
        """
        if pygpufit_present:
            self._set_initial_constraints()

    def _set_initial_constraints(self):
        """
        Set the initial constraints for the fit.
        """
        default_constraints = pyqdm.config["default_fitconstraints"]
        self.set_constraints(
            "center",
            [default_constraints["center_min"], default_constraints["center_max"]],
            bound_type=default_constraints["center_type"],
        )
        self.set_constraints(
            "width",
            [default_constraints["width_min"], default_constraints["width_max"]],
            bound_type=default_constraints["width_type"],
        )
        self.set_constraints(
            "contrast",
            [default_constraints["contrast_min"], default_constraints["contrast_max"]],
            bound_type=default_constraints["contrast_type"],
        )
        self.set_constraints(
            "offset",
            [default_constraints["offset_min"], default_constraints["offset_max"]],
            bound_type=default_constraints["offset_type"],
        )

    def _reshape_parameter(self, data, n_pol, n_frange):
        """
        Reshape data so that all data for a frange are in series (i.e. [low_freq(B+), low_freq(B-)]).
        Input data must have format: [polarity, frange, n_pixel, n_freqs]
        """
        out = np.array(data)
        out = np.reshape(out, (n_frange, n_pol, -1, data.shape[-1]))
        out = np.swapaxes(out, 0, 1)  # swap polarity and frange
        return out

    def fit_ODMR(self):
        """
        Fit the data using the current fit type.
        """
        estimator_id = gf.EstimatorID.MLE  # maximum likelihood estimator
        # ESR14N is the default model
        model_id = getattr(gf.ModelID, self.FIT_TYPES[self._diamond_type])

        parameters, states, chi_squares, number_iterations, execution_time = (
            [],
            [],
            [],
            [],
            [],
        )

        # calculate the fit for each f_range -> 1st dimension is f_range not n_pol
        # needs to be swapped later

        for i in np.arange(0, self.ODMRobj.n_frange):
            self.LOG.info(f"Fitting {self.POLARITIES[i]} polarization")
            data = self.ODMRobj.data[:, i].reshape(-1, self.ODMRobj.n_freqs).astype(np.float32)
            initial_guess = self.initial_guess[:, i].reshape(-1, len(self._fitting_params)).astype(np.float32)

            # fit the data
            p, s, c, n, t = gf.fit_constrained(
                data=data,
                user_info=self.ODMRobj.f_ghz[i].astype(np.float32),
                constraints=self.constraints_array,
                constraint_types=self.constraint_types,
                weights=None,
                initial_parameters=initial_guess,
                model_id=model_id,
                estimator_id=estimator_id,
                max_number_iterations=100,
                tolerance=1e-10,
            )
            parameters.append(p)
            states.append(s)
            chi_squares.append(c)
            number_iterations.append(n)
            execution_time.append(t)

        if np.ndim(parameters) == 3:
            parameters = np.array(parameters)[:, np.newaxis, :, :]
            states = np.array(states)[:, np.newaxis, :]
            chi_squares = np.array(chi_squares)[:, np.newaxis, :]
            number_iterations = np.array(number_iterations)[:, np.newaxis, :]

        # swapping the f_range and n_pol axes
        parameters = self._reshape_parameter(parameters, self.ODMRobj.n_pol, self.ODMRobj.n_frange)

        # check if the reshaping is correct
        assert np.all(parameters[0, 1] == p[: int(p.shape[0] / self.ODMRobj.n_pol)])

        # reshape the states and chi_squares
        shape = (self.ODMRobj.n_pol, self.ODMRobj.n_frange, self.ODMRobj.n_pixel)
        states = np.reshape(states, shape)
        chi_squares = np.reshape(chi_squares, shape)
        number_iterations = np.reshape(number_iterations, shape)

        # flip pol / frange
        states = np.swapaxes(states, 0, 1)
        chi_squares = np.swapaxes(chi_squares, 0, 1)
        number_iterations = np.swapaxes(number_iterations, 0, 1)

        self._fitted_parameter = np.rollaxis(parameters, -1)
        self._states = states
        self._chi_squares = chi_squares
        self._number_iterations = number_iterations
        self.execution_time = execution_time

        t = "; ".join([f"{v:.2f}" for v in execution_time])
        self.LOG.debug(f"Fitting took {np.sum(execution_time):.3f} ({t}) seconds.")
        if np.any(np.flatnonzero(states)):
            n = len(np.flatnonzero(states))
            self.LOG.warning(f"Fit did not converge in {n} pixels.")

        self._fitted = True
        self._fitting_constraints = self.constraints

    def get_param(self, param, reshape=True):
        """
        Get the value of a parameter reshaped to the image dimesions.
        """
        if not self.fitted:
            raise NotImplementedError("No fit has been performed yet. Run fit_ODMR().")

        if param == "resonance":
            param = "center"

        idx = [i for i, v in enumerate(self._fitting_params) if v == param]
        if not idx:
            idx = [i for i, v in enumerate(self._fitting_params_unique) if v == param]
        if not idx:
            raise ValueError(f"Unknown parameter: {param}")

        out = self._fitted_parameter[idx]

        if reshape:
            out = out.reshape(-1, self.ODMRobj.n_pol, self.ODMRobj.n_frange, *self.ODMRobj.scan_dimensions)

        return np.squeeze(out)

    def print_fit_infos(self):
        self.LOG.info(f"ODMR fitted: {self.fitted}")
        self.LOG.info(f"Diamond type: {self._diamond_type}")
        self.LOG.info(f"Fitting type: {self.FIT_TYPES[self._diamond_type]}")
        self.LOG.info(f"Fitted parameters: {self._fitting_params}")
        self.LOG.info("Fitting constraints:")
        for i, p in enumerate(self._fitting_params):
            self.LOG.info(
                f"\t{p:10}: {self._fitting_constraints[i * 2]:8.2e} - {self._fitting_constraints[i * 2 + 1]:8.2e} | {self._constraint_types[i]}"
            )

        self.LOG.info(f"Number of failed pixels: {np.flatnonzero(self._states).size}")
        self.LOG.info(
            f"Chi-square: mean: {np.mean(self._chi_squares):.2e}, median {np.median(self._chi_squares):.2e}, max: {np.max(self._chi_squares):.2e}"
        )

    @property
    def _fitting_params_unique(self):
        """
        Return a list of unique fitting parameters.
        :return: list
        """
        lst = []
        for v in self._fitting_params:
            if self._fitting_params.count(v) > 1:
                for n in range(10):
                    if f"{v}_{n}" not in lst:
                        lst.append(f"{v}_{n}")
                        break
            else:
                lst.append(v)
        return lst

    def pd_fit_data(self, ipol, irange):
        """
        Create a pandas dataframe with the fit data.

        :param ipol: index of the polarity
        :param irange: index of the frange
        :return: pandas dataframe
        """
        return pd.DataFrame(
            data=self._fitted_parameter[ipol, irange].astype(np.float), columns=self._fitting_params_unique
        )

    @property
    def resonance(self):
        """
        Return the resonance of the fit.
        """
        return self.get_param("center")

    @property
    def width(self):
        """
        Return the resonance of the fit.
        """
        return self.get_param("width")

    @property
    def contrast(self):
        """
        Return the resonance of the fit.
        """
        return np.mean(self.get_param("contrast"), axis=0)

    @property
    def offset(self):
        """
        Return the resonance of the fit.
        """
        return self.get_param("offset")

    # QUALITY CHECKS ###

    def median_fit(self):
        """
        Return the median fit parameters.
        """
        return np.median(self._fitted_parameter.reshape((self._fitted_parameter.shape[0], -1)), axis=1)

    def std_fit(self):
        """
        Return the median fit parameters.
        """
        return np.std(self._fitted_parameter.reshape((self._fitted_parameter.shape[0], -1)), axis=1)

    # CALCULATIONS ###
    @property
    def delta_resonance(self):
        """
        Return the difference between low and high freq. resonance of the fit.

        Returns:
            numpy.ndarray: negative difference
            numpy.ndarray: positive difference

        """
        d = np.expand_dims(np.array([-1, 1]), axis=[1, 2])
        return (self.resonance[:, 1] - self.resonance[:, 0]) / 2 / self.GAMMA * d

    @property
    def B111(self):
        """
        Return the B111 of the fit.
        """
        neg_difference, pos_difference = self.delta_resonance
        return (neg_difference + pos_difference) / 2, (neg_difference - pos_difference) / 2

    @property
    def B111_remanent(self):
        """
        Return the remanent component of the B111 of the fit.
        :return: numpy.ndarray
        """
        return self.B111[0]

    @property
    def B111_induced(self):
        """
        Return the Induced component of B111 of the fit.
        :return: numpy.ndarray
        """
        return self.B111[1]

    # PLOTTING
    def rc2idx(self, rc, ref="data"):
        """
        Convert the xy coordinates to the index of the data.
        If the reference is 'data', the index is relative to the data.
        If the reference is 'img', the index is relative to the LED/laser image.
        Only data -> data and img -> img are supported.
        :param rc: numpy.ndarray [[row], [column]] -> [[y], [x]]
        :param ref: str 'data' or 'img'
        :return: numpy.ndarray [idx]
        """
        if ref == "data":
            idx = rc2idx(rc, self.scan_dimensions)
        elif ref == "img":
            idx = rc2idx(rc, self.light.shape)
        else:
            raise ValueError(f"Reference {ref} not supported.")
        return idx

    def idx2rc(self, idx, ref="data"):
        """
        Convert an index to a rc coordinate of the reference.
        If the reference is 'data', the index is relative to the data.
        If the reference is 'img', the index is relative to the LED/laser image.
        Only data -> data and img -> img are implemented.

        :param idx: int or numpy.ndarray [idx]
        :param ref: 'data' or 'img'

        :return: numpy.ndarray ([row], [col]) -> [[y], [x]]
        """
        if ref == "data":
            rc = idx2rc(idx, self.scan_dimensions)
        elif ref == "img":
            rc = idx2rc(idx, self.light.shape)
        else:
            raise ValueError(f"Reference {ref} not supported.")
        return rc

    def plot_fluorescence(self, f_idx=10):
        """
        Plot the fluorescence of a given (idx) frequency.
        """
        plotting.plot_fluorescence(self, f_idx)

    def plot_parameter(self, param, save=False):
        """
        Plot the resonances of the fit.
        """
        plotting.plot_fit_params(self, param)

    def plot_fits(self, idx=10):
        """
        Plot the fits.
        """
        f, ax = plotting.check_fit_pixel(self, idx)
        return f, ax


from dataclasses import dataclass


@dataclass
class fit:
    _parameter = None
    _fitted_parameter = None
    _states = None
    _chi_squares = None
    _fitting_params = None
    _fitted_parameter_unique = None
    _constraints = None
    _constraint_types = None
