import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import QDMpy
import QDMpy.core.fit
from QDMpy.core.fit import MODELS, Fit
from QDMpy.core.odmr import ODMR
from QDMpy.exceptions import CantImportError, WrongFileNumber
from QDMpy.utils import get_image, idx2rc, rc2idx

GAMMA = 28.024 / 1e6  # GHz/muT;
DIAMOND_TYPES = ["NaN", "MISC.", "N15", "N14"]
MODELS = ["gauss1d", "esrsingle", "esr15n", "esr14n"]
POLARITIES = ["positive", "negative"]
FRANGES = ["high", "low"]

from pathlib import Path

import pandas as pd
from scipy.io import savemat


class QDM:
    LOG = logging.getLogger(__name__)

    @property
    def outliers(self):
        """
        Return the outliers mask.
        :return: ndarray of boolean
        """
        if self._outliers is None:
            self._outliers = self.detect_outliers()
        return self._outliers

    # outliers
    def __init__(
            self,
            odmr_instance,
            light,
            laser,
            working_directory,
            pixel_size=4e-6,
            diamond_type=None,
    ):

        self.LOG.info("Initializing QDM object.")
        self.LOG.info(f'Working directory: "{working_directory}"')
        self.working_directory = Path(working_directory)

        self.LOG.debug("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.debug(f"read parameter shape: data: {odmr_instance.data.shape}")
        self.LOG.debug(f"                      scan_dimensions: {odmr_instance.data_shape}")
        self.LOG.debug(f"                      frequencies: {odmr_instance.f_ghz.shape}")
        self.LOG.debug(f"                      n_freqs: {odmr_instance.n_freqs}")

        self.odmr = odmr_instance

        self._outliers = None

        self.light = light
        self.laser = laser

        self._B111 = None

        if diamond_type is None:
            diamond_type = self.guess_diamond_type()

        self._fit = None
        self.set_diamond_type(diamond_type)
        self._fit = Fit(self.odmr.data, self.odmr.f_ghz, model=MODELS[self._diamond_type])
        self.pixel_size = pixel_size  # 4 um

        self._check_bin_factor()

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

    @property
    def outlier_pdf(self):
        """
        Return the outlier pandas Dataframe.
        :return: pandas.DataFrame
        """
        outlier_pdf = pd.DataFrame(columns=["idx", "x", "y"])
        outlier_pdf["x"] = self.outliers_xy[:, 0]
        outlier_pdf["y"] = self.outliers_xy[:, 1]
        outlier_pdf["idx"] = self.outliers_idx
        return outlier_pdf

    def detect_outliers(self, dtype="width", method="LocalOutlierFactor", **outlier_props):
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
        outlier_props["n_jobs"] = -1
        d1 = self.get_param("chi2", reshape=False)
        d1 = np.sum(d1, axis=tuple(range(0, d1.ndim - 1)))

        if dtype in self.fit.fitting_parameter + self.fit.fitting_parameter_unique:
            d2 = self.get_param(dtype, reshape=False)
        else:
            raise ValueError(f"dtype {dtype} not recognized")

        d2 = np.sum(d2, axis=tuple(range(0, d2.ndim - 1)))
        data = np.stack([d1, d2], axis=0)

        outlier_props["contamination"] = outlier_props.pop("contamination", 0.05)

        if method == "LocalOutlierFactor":
            clf = LocalOutlierFactor(**outlier_props)
        elif method == "IsolationForest":
            outlier_props = {
                k: v
                for k, v in outlier_props.items()
                if k
                   in [
                       "n_estimators",
                       "max_samples",
                       "contamination",
                       "max_features",
                       "bootstrap",
                       "n_jobs",
                       "random_state",
                   ]
            }
            clf = IsolationForest(**outlier_props)
        else:
            raise ValueError(f"Method {method} not recognized.")

        shape = data.shape
        self.LOG.debug(f"Detecting outliers in <<{dtype}>> data of shape {shape}")
        outliers = (clf.fit_predict(data.T) + 1) / 2
        # collapse the first dimensions so that the product applies to all except the pixel dimension
        outliers = outliers.reshape(-1, shape[-1])
        outliers = ~np.product(outliers, axis=0).astype(bool)
        self.LOG.info(
            f"Outlier detection using {method} of <<{dtype}>> detected {outliers.sum()} outliers pixels.\n"
            f"                                      Indixes can be accessed using 'outlier_idx' and 'outlier_xy'"
        )
        self.LOG.debug(f"returning {outliers.shape}")
        self._outliers = outliers
        return self._outliers

    def apply_outlier_mask(self, outlier=None):
        if outlier is None:
            outlier = self.outliers
        self.LOG.debug(f"Applying outlier mask of shape {outlier.shape}")
        self.odmr.apply_outlier_mask(outlier)

    # binning
    @property
    def bin_factor(self):
        return self.odmr.bin_factor

    def bin_data(self, bin_factor):
        """
        Bin the data.
        :param bin_factor:
        :return:
        """
        if bin_factor == self.bin_factor:
            return
        self.odmr.bin_data(bin_factor=bin_factor)
        self._fit.data = self.odmr.data
        if self._fit.fitted:
            self.LOG.info("Binning changed, fits need to be recalculated!")
            self._fit._reset_fit()

    def _check_bin_factor(self):
        bin_factors = self.light.shape / self.odmr.data_shape

        if np.all(self.odmr._img_shape != self.light.shape):
            self.LOG.warning(
                f"Scan dimensions of ODMR ({self.odmr._img_shape}) and LED ({self.light.shape}) are not equal. Setting pre_binfactor to {bin_factors[0]}."
            )
            # set the true bin factor
            self.odmr._pre_bin_factor = bin_factors[0]
            self.odmr._img_shape = np.array(self.light.shape)

    # global fluorescence related functions
    def correct_glob_fluorescence(self, glob_fluo):
        """
        Corrects the global fluorescence.
        """
        self.LOG.debug(f"Correcting global fluorescence {glob_fluo}.")
        self.odmr.correct_glob_fluorescence(glob_fluo)
        self._fit.data = self.odmr.data

    @property
    def global_factor(self):
        return self.odmr.global_factor

    # diamond related
    def guess_diamond_type(self, *args):
        """
        Guess the diamond type based on the number of peaks.

        :return: diamond_type (int)
        """
        from scipy.signal import find_peaks

        indices = []

        # Find the indices of the peaks
        for p in range(self.odmr.n_pol - 1):
            for f in range(self.odmr.n_frange - 1):
                peaks = find_peaks(-self.odmr.mean_odmr[p, f], prominence=0.003)
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
            for i in range(self.odmr.mean_odmr.shape[0]):
                for j in range(self.odmr.mean_odmr.shape[1]):
                    plt.plot(self.odmr.mean_odmr[i, j], color=f"C{n}")
                    plt.plot(
                        indices[n],
                        self.odmr.mean_odmr[i, j, indices[n]],
                        "X",
                        color=f"C{n}",
                        label=f"{n}: {len(indices[n])}",
                    )
                    n += 1
            plt.legend()
        self.LOG.info(f"Guessed diamond type: {n_peaks} peaks -> {DIAMOND_TYPES[n_peaks]}")

        return n_peaks

    @property
    def diamond_type(self):
        """
        Return the diamond type.
        """
        return DIAMOND_TYPES[self._diamond_type]

    def set_diamond_type(self, diamond_type):
        """
        Set the diamond type.

        :param diamond_type:
        """
        self.LOG.debug(f'Setting diamond type to "{diamond_type}"')

        if isinstance(diamond_type, int):
            diamond_type = DIAMOND_TYPES[diamond_type]

        self._diamond_type = {"N14": 3, "N15": 2, "SINGLE": 1, "MISC.": 1}[diamond_type]

        if self._fit is not None:
            self._fit.model = MODELS[self._diamond_type]

    @property
    def data_shape(self):
        return self.odmr.data_shape

    # fitting
    @property
    def fit(self):
        return self._fit

    @property
    def fitted(self):
        return self._fit.fitted

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
        self._fit.set_constraints(param, vmin, vmax, bound_type)

    def reset_constraints(self):
        """
        Reset the constraints to the default values.
        """
        self._fit._set_initial_constraints()

    def fit_odmr(self):
        """
        Fit the data using the current fit type.
        """
        if not QDMpy.PYGPUFIT_PRESENT:
            self.LOG.error("pygpufit not installed. Skipping fitting.")
            raise ImportError("pygpufit not installed.")
        self._fit.fit_odmr()

    def get_param(self, param, reshape=True):
        """
        Get the value of a parameter reshaped to the image dimesions.
        """
        out = self._fit.get_param(param)

        if reshape:
            out = out.reshape(
                -1,
                self.odmr.n_pol,
                self.odmr.n_frange,
                *self.odmr.data_shape,
            )

        return np.squeeze(out)

    def _reshape_parameter(self, data, n_pol, n_frange):
        """
        Reshape data so that all data for a frange are in series (i.e. [low_freq(B+), low_freq(B-)]).
        Input data must have format: [polarity, frange, n_pixel, n_freqs]
        """
        out = np.array(data)
        out = np.reshape(out, (n_frange, n_pol, -1, data.shape[-1]))
        out = np.swapaxes(out, 0, 1)  # swap polarity and frange
        return out

    ## from METHODS ##
    @classmethod
    def from_matlab(cls, matlab_files, dialect="QDM.io"):
        """
        Loads QDM data from a Matlab file.
        """

        match dialect:
            case "QDM.io":
                return cls.from_qdmio(matlab_files)

        raise NotImplementedError(f'Dialect "{dialect}" not implemented.')

    @classmethod
    def from_qdmio(cls, data_folder, diamond_type=None):
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

        return cls(
            odmr_obj,
            light=light,
            laser=laser,
            diamond_type=diamond_type,
            working_directory=data_folder,
        )

    # EXPORT METHODS ###
    def export_qdmio(self, path_to_file=None):
        """
        Export the data to a QDM.io file. This is a Matlab file named B111dataToPlot.mat. With the following variables:

        ['negDiff', 'posDiff', 'B111ferro', 'B111para', 'chi2Pos1', 'chi2Pos2', 'chi2Neg1', 'chi2Neg2', 'ledImg',
         'laser', 'pixelAlerts']
        """

        path_to_file = Path(path_to_file) if path_to_file is not None else self.working_directory
        full_folder = path_to_file / f"{self.odmr.bin_factor}x{self.odmr.bin_factor}Binned"
        full_folder.mkdir(parents=True, exist_ok=True)
        data = self._save_data(dialect="QDMio")

        return savemat(
            full_folder / "B111dataToPlot.mat",
            data,
        )

    def export_qdmpy(self, path_to_file):
        print(path_to_file)
        path_to_file = Path(path_to_file)
        savemat(path_to_file, self._save_data(dialect="QDMpy"))

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
        resonance = self.get_param("resonance")
        return (resonance[:, 1] - resonance[:, 0]) / 2 / GAMMA * d

    @property
    def b111(self):
        """
        Return the B111 of the fit.
        """
        neg_difference, pos_difference = self.delta_resonance
        return (neg_difference + pos_difference) / 2, (neg_difference - pos_difference) / 2

    @property
    def b111_remanent(self):
        """
        Return the remanent component of the B111 of the fit.
        :return: numpy.ndarray
        """
        return self.b111[0]

    @property
    def b111_induced(self):
        """
        Return the Induced component of B111 of the fit.
        :return: numpy.ndarray
        """
        return self.b111[1]

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
            idx = rc2idx(rc, self.data_shape)
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
            rc = idx2rc(idx, self.data_shape)
        elif ref == "img":
            rc = idx2rc(idx, self.light.shape)
        else:
            raise ValueError(f"Reference {ref} not supported.")
        return rc

    def _save_data(self, dialect="QDMpy") -> dict:
        """
        Return the data structure that can be saved to a file.

        :param dialect: str 'QDMpy' or 'QDMio'
        :return: dict

        """

        if dialect == "QDMpy":
            return {
                "remanent": self.b111[0],
                "induced": self.b111[1],
                "chi_squares": self.get_param("chi2"),
                "resonance": self.get_param("resonance"),
                "width": self.get_param("width"),
                "contrast": self.get_param("contrast"),
                "offset": self.get_param("offset"),
                "fit.constraints": self.fit.constraints,
                "diamond_type": self.diamond_type,
                "laser": self.laser,
                "light": self.light,
                "bin_factor": self.bin_factor,
            }

        elif dialect == "QDMio":
            neg_diff, pos_diff = self.delta_resonance
            b111_remanent, b111_induced = self.b111
            chi_squares = self.get_param("chi2")
            chi2_pos1, chi2_pos2 = chi_squares[0]
            chi2_neg1, chi2_neg2 = chi_squares[1]
            led_img = self.light
            laser_img = self.laser
            pixel_alerts = np.zeros(b111_remanent.shape)

            out = dict(
                negDiff=neg_diff,
                posDiff=pos_diff,
                B111ferro=b111_remanent,
                B111para=b111_induced,
                chi2Pos1=chi2_pos1,
                chi2Pos2=chi2_pos2,
                chi2Neg1=chi2_neg1,
                chi2Neg2=chi2_neg2,
                ledImg=led_img,
                laser=laser_img,
                pixelAlerts=pixel_alerts,
                bin_factor=self.bin_factor,
                QDMpy_version=QDMpy.__version__,
            )
            return out

        else:
            raise ValueError(f"Dialect {dialect} not supported.")
