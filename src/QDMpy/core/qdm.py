import logging
import os
from typing import Union, Tuple, Optional, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import QDMpy
import QDMpy.core.fit
from QDMpy.core import models
from QDMpy.core.fit import Fit
from QDMpy.core.odmr import ODMR
from QDMpy.exceptions import CantImportError, WrongFileNumber
from QDMpy.utils import get_image, idx2rc, rc2idx

GAMMA = 28.024 / 1e6  # GHz/muT;

POLARITIES = ["positive", "negative"]
FRANGES = ["high", "low"]
DIAMOND_TYPES = ["", "MISC.", "N15", "N14"]

from pathlib import Path

import pandas as pd
from scipy.io import savemat


class QDM:
    """ """

    LOG = logging.getLogger(__name__)

    # outliers
    def __init__(
            self,
            odmr_instance: ODMR,
            light: np.ndarray,
            laser: np.ndarray,
            working_directory: Union[str, os.PathLike],
            pixel_size: float = 4e-6,
            model_name: str = 'auto',
    ) -> None:
        """Initialize the QDM object.

        Args:
            odmr_instance: ODMR instance
            light: light image
            laser: laser image
            working_directory: working directory
            pixel_size: pixel size in m
            model_name: model name (Default value = 'auto')
                If 'auto' the model is chosen based on the mean ODMR data.
                See Also: QDMpy.core.models.guess_model_name

        """

        self.LOG.info("Initializing QDM object.")
        self.LOG.info(f'Working directory: "{working_directory}"')
        self.working_directory = Path(working_directory)

        self.LOG.debug("ODMR data format is [polarity, f_range, n_pixels, n_freqs]")
        self.LOG.debug(f"read parameter shape: data: {odmr_instance.data.shape}")
        self.LOG.debug(f"                      scan_dimensions: {odmr_instance.data_shape}")
        self.LOG.debug(f"                      frequencies: {odmr_instance.f_ghz.shape}")
        self.LOG.debug(f"                      n_freqs: {odmr_instance.n_freqs}")

        self.odmr = odmr_instance

        self._outliers = np.ones(self.odmr.data_shape, dtype=bool)

        self.light = light
        self.laser = laser

        self._B111 = None

        self._fit = Fit(data=self.odmr.data,
                        frequencies=self.odmr.f_ghz,
                        model_name=model_name)

        self.pixel_size = pixel_size  # 4 um

        self._check_bin_factor()

    @property
    def outliers(self) -> NDArray:
        """

        Args:

        Returns:
          :return: ndarray of boolean

        """
        return self._outliers

    @property
    def outliers_idx(self) -> NDArray:
        """

        Args:

        Returns:
          Indices are in reference to the binned ODMR data.

          :return: np.array

        """
        return np.where(self.outliers)[0]

    @property
    def outliers_xy(self) -> NDArray:
        """

        Args:

        Returns:
          In reference to the binned ODMR data.

          :return: np.array of shape (n_outlier, 2)

        """
        y, x = self.idx2rc(self.outliers_idx)
        return np.stack([x, y], axis=1)

    @property
    def outlier_pdf(self) -> pd.DataFrame:
        """

        Args:

        Returns:
          :return: pandas.DataFrame

        """
        outlier_pdf = pd.DataFrame(columns=["idx", "x", "y"])
        outlier_pdf["x"] = self.outliers_xy[:, 0]
        outlier_pdf["y"] = self.outliers_xy[:, 1]
        outlier_pdf["idx"] = self.outliers_idx
        return outlier_pdf

    def detect_outliers(self, dtype: str = "width", method: str = "LocalOutlierFactor",
                        **outlier_props: Any) -> np.ndarray:
        """Detect outliers in the ODMR data.

        The outliers are detected using 'method'. The method can be either 'LocalOutlierFactor' or 'IsolationForest'.

        The LocalOutlierFactor is a scikit-learn method.
        The IsolationForest is a scikit-learn method.

        The method arguments can be passed as a keyword argument.

        Args:
          dtype: the data type the method should be used on (Default value = "width")
          method: the outlier detection method (Default value = "LocalOutlierFactor")
          **outlier_props: keyword arguments for the outlier detection method

        Returns: outlier mask [bool] of shape ODMR.data_shape

        """
        outlier_props["n_jobs"] = -1
        d1 = self.get_param("chi2", reshape=False)
        d1 = np.sum(d1, axis=tuple(range(0, d1.ndim - 1)))

        if dtype in self.fit.model_params + self.fit.model_params_unique:
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

    def apply_outlier_mask(self, outlier: Union[NDArray, None] = None) -> None:
        """

        Args:
          outlier:  (Default value = None)

        Returns:

        """
        if outlier is None:
            outlier = self.outliers

        self.LOG.debug(f"Applying outlier mask of shape {outlier.shape}")
        self.odmr.apply_outlier_mask(outlier)

    # binning
    @property
    def bin_factor(self) -> int:
        """mirrors the bin_factor of the ODMR instance"""
        return self.odmr.bin_factor

    def bin_data(self, bin_factor: int) -> None:
        """Bin the data.

        Args:
          bin_factor: return:

        Returns:

        """
        if bin_factor == self.bin_factor:
            return
        self.odmr.bin_data(bin_factor=bin_factor)
        self._fit.data = self.odmr.data
        if self._fit.fitted:
            self.LOG.info("Binning changed, fits need to be recalculated!")
            self._fit._reset_fit()

    def _check_bin_factor(self) -> None:
        """check if the bin factor is correct

        If there was some form pf "prebinning" in the ODMR data, the initial bin factor is not 1.
        Therefore, we add a pre_binning factor.
        """
        bin_factors = self.light.shape / self.odmr.data_shape

        if np.all(self.odmr._img_shape != self.light.shape):
            self.LOG.warning(
                f"Scan dimensions of ODMR ({self.odmr._img_shape}) and LED ({self.light.shape}) are not equal. Setting pre_binfactor to {bin_factors[0]}."
            )
            # set the true bin factor
            self.odmr._pre_bin_factor = bin_factors[0]
            self.odmr._img_shape = np.array(self.light.shape)

    # global fluorescence related functions
    def correct_glob_fluorescence(self, glob_fluo: float) -> None:
        """Corrects the global fluorescence.

        Args:
          glob_fluo: global fluorescence correction factor
        """
        self.LOG.debug(f"Correcting global fluorescence {glob_fluo}.")
        self.odmr.correct_glob_fluorescence(glob_fluo)
        self._fit.data = self.odmr.data

    @property
    def global_factor(self) -> float:
        """Global fluorescence factor used for correction"""
        return self.odmr.global_factor

    # MODEL related
    @property
    def model_names(self) -> str:
        """List of available models"""
        return self.fit.model['func_name']

    def set_model_name(self, model_name: Union[str, int]) -> None:
        """Set the diamond type.

        Args:
          model_name: type of diamond used (int or str) e.g. N15 of 2 as in 2 peaks
        """

        if isinstance(model_name, int):
            # get str of diamond type from number of peaks (e.g. 2 -> ESR15N)
            model_name = models.PEAK_TO_TYPE[model_name]

        if model_name not in models.IMPLEMENTED:
            raise NotImplementedError('diamond type has not been implemented, yet')

        self.LOG.debug(f'Setting model to "{model_name}"')

        if hasattr(self, "_fit") and self._fit is not None:
            self._fit.model_name = models.IMPLEMENTED[self._model_name]['func_name']

    @property
    def data_shape(self) -> NDArray:
        """ """
        return self.odmr.data_shape

    # fitting
    @property
    def fit(self) -> Fit:
        """ """
        return self._fit

    @property
    def fitted(self) -> bool:
        """ """
        return self._fit.fitted

    def set_constraints(
            self,
            param: str,
            vmin: Optional[Union[str, None]] = None,
            vmax: Optional[Union[str, None]] = None,
            bound_type: Optional[Union[str, None]] = None,
    ) -> None:
        """Set the constraints for the fit.

        Args:
          param:
          vmin:  (Default value = None)
          vmax:  (Default value = None)
          bound_type:  (Default value = None)

        Returns:


        """
        self._fit.set_constraints(param, vmin, vmax, bound_type)

    def reset_constraints(self) -> None:
        """Reset the constraints to the default values."""
        self._fit._set_initial_constraints()

    def fit_odmr(self, refit=False) -> None:
        """Fit the data using the current fit type."""
        if not QDMpy.PYGPUFIT_PRESENT:
            self.LOG.error("pygpufit not installed. Skipping fitting.")
            raise ImportError("pygpufit not installed.")
        self._fit.fit_odmr(refit=refit)

    def get_param(self, param: str, reshape: bool = True) -> NDArray:
        """Get the value of a parameter reshaped to the image dimesions.

        Args:
          param:
          reshape:  (Default value = True)

        Returns:

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

    def _reshape_parameter(
            self,
            data: NDArray,
            n_pol: int,
            n_frange: int,
    ) -> NDArray:
        """Reshape data so that all data for a frange are in series (i.e. [low_freq(B+), low_freq(B-)]).
        Input data must have format: [polarity, frange, n_pixel, n_freqs]

        Args:
          data:
          n_pol:
          n_frange:

        Returns:

        """
        out = np.array(data)
        out = np.reshape(out, (n_frange, n_pol, -1, data.shape[-1]))
        out = np.swapaxes(out, 0, 1)  # swap polarity and frange
        return out

    ## from METHODS ##
    @classmethod
    def from_matlab(cls, matlab_files: Union[os.PathLike[Any], str], dialect: str = "QDM.io") -> Any:
        """Loads QDM data from a Matlab file.

        Args:
          matlab_files:
          dialect:  (Default value = "QDM.io")

        Returns:

        """

        match dialect:
            case "QDM.io":
                return cls.from_qdmio(matlab_files)

        raise NotImplementedError(f'Dialect "{dialect}" not implemented.')

    @classmethod
    def from_qdmio(cls, data_folder: Union[os.PathLike[Any], str], model_name: str = "auto") -> Any:
        """Loads QDM data from a Matlab file.

        Args:
          data_folder:
          model_name:  (Default value = None)

        Returns:

        """
        cls.LOG.info(f"Initializing QDMpy object from QDMio data in {data_folder}")
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
            model_name=model_name,
            working_directory=data_folder,
        )

    # EXPORT METHODS ###
    def export_qdmio(self, path_to_file: Union[os.PathLike, str, None] = None) -> None:
        """Export the data to a QDM.io file. This is a Matlab file named B111dataToPlot.mat. With the following variables:

        ['negDiff', 'posDiff', 'B111ferro', 'B111para', 'chi2Pos1', 'chi2Pos2', 'chi2Neg1', 'chi2Neg2', 'ledImg',
         'laser', 'pixelAlerts']

        Args:
          path_to_file:  (Default value = None)

        Returns:

        """

        path_to_file = Path(path_to_file) if path_to_file is not None else self.working_directory
        full_folder = path_to_file / f"{self.odmr.bin_factor}x{self.odmr.bin_factor}Binned"
        full_folder.mkdir(parents=True, exist_ok=True)
        data = self._save_data(dialect="QDMio")

        savemat(
            full_folder / "B111dataToPlot.mat",
            data,
        )

    def export_qdmpy(self, path_to_file: Union[os.PathLike, str]) -> None:
        """

        Args:
          path_to_file:

        Returns:

        """
        path_to_file = Path(path_to_file)
        savemat(path_to_file, self._save_data(dialect="QDMpy"))

    # CALCULATIONS ###
    @property
    def delta_resonance(self) -> NDArray:
        """Return the difference between low and high freq. resonance of the fit.

        Args:

        Returns:
          numpy.ndarray: negative difference
          numpy.ndarray: positive difference

        """
        d = np.expand_dims(np.array([-1, 1]), axis=[1, 2])
        resonance = self.get_param("resonance")
        return (resonance[:, 1] - resonance[:, 0]) / 2 / GAMMA * d

    @property
    def b111(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        neg_difference, pos_difference = self.delta_resonance
        return (neg_difference + pos_difference) / 2, (neg_difference - pos_difference) / 2

    @property
    def b111_remanent(self) -> np.ndarray:
        """

        Args:

        Returns:
          :return: numpy.ndarray

        """
        return self.b111[0]

    @property
    def b111_induced(self) -> np.ndarray:
        """

        Args:

        Returns:
          :return: numpy.ndarray

        """
        return self.b111[1]

    # PLOTTING
    def rc2idx(self, rc: np.ndarray, ref: str = "data") -> NDArray:
        """Convert the xy coordinates to the index of the data.

        If the reference is 'data', the index is relative to the data.
        If the reference is 'img', the index is relative to the LED/laser image.
        Only data -> data and img -> img are supported.

        Args:
          rc: numpy.ndarray [[row], [column]] -> [[y], [x]]
          ref: str 'data' or 'img' (Default value = "data")
          rc:np.ndarray:

        Returns:
          numpy.ndarray [idx]

        """
        if ref == "data":
            shape = self.odmr.data_shape
        elif ref == "img":
            shape = self.odmr.img_shape
        else:
            raise ValueError(f"Reference {ref} not supported.")
        return rc2idx(rc, shape)  # type: ignore[arg-type]

    def idx2rc(
            self, idx: Union[int, np.ndarray], ref: str = "data"
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Convert an index to a rc coordinate of the reference.

        If the reference is 'data', the index is relative to the data.
        If the reference is 'img', the index is relative to the LED/laser image.
        Only data -> data and img -> img are implemented.

        Args:
          idx: int or numpy.ndarray [idx] or [idx, idx]
          ref: data' or 'img' (Default value = "data")

        Returns:
          numpy.ndarray ([row], [col]) -> [[y], [x]]

        """
        if ref == "data":
            rc = idx2rc(idx, self.data_shape)  # type: ignore[arg-type]
        elif ref == "img":
            rc = idx2rc(idx, self.light.shape)
        else:
            raise ValueError(f"Reference {ref} not supported.")
        return rc

    def _save_data(self, dialect: str = "QDMpy") -> dict:
        """Return the data structure that can be saved to a file.

        Args:
          dialect: str 'QDMpy' or 'QDMio'
          dialect:str:  (Default value = "QDMpy")

        Returns:
          dict

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
                "diamond_type": self.model_name,
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
