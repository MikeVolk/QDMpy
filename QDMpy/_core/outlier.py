import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, np.ndarray

import QDMpy
import QDMpy.utils

class Outlier(ABC):
    LOG = logging.getLogger(f"QDMpy.{__name__}")

    @property
    def settings(self) -> dict:
        if self._settings is None:
            if self.__class__.__name__ in QDMpy.SETTINGS["outlier_detection"]:
                self._settings = QDMpy.SETTINGS["outlier_detection"][
                    self.__class__.__name__
                ]
            else:
                self.LOG.warning(
                    f"Settings for {self.__class__.__name__} not found in QDMpy.SETTINGS"
                )
                self._settings = {}
        return self._settings

    def __init__(self, data_shape: Tuple[int, ...]) -> None:
        self.data_shape = data_shape
        self.outliers: np.ndarray = np.zeros(self.data_shape, dtype=bool)
        self.LOG.debug(
            f"{self.__class__.__name__} initialized with data of shape {self.data_shape}"
        )
        self._settings = None

    @abstractmethod
    def detect_outlier(self, **kwargs) -> None:
        pass

    @property
    def n(self):
        return self.outliers.sum()

    def reset(self):
        """Reset the outliers array to all False values."""
        self.outliers = np.zeros(self.data_shape, dtype=bool)

    def __repr__(self) -> str:
        return f"Outlier(b111={self.data_shape})"
class StatisticsPercentile(Outlier):
    def __repr__(self):
        return (
            f"Outlier(b111={self.data_shape}, chi2={self.chi2.shape}, "
            f"width={self.width.shape}, mean_contrast={self.mean_contrast.shape})"
        )

    def __init__(
        self,
        b111: np.ndarray,
        chi2: np.ndarray,
        width: np.ndarray,
        mean_contrast: np.ndarray,
        **kwargs,
    ):
        """
        Initialize the outlier detection using the fit statistics.

        Args:
            b111: B111 values of the fit
            chi2: Chi2 values of the fit
            width:  Width of the fit
            mean_contrast:  Mean contrast of the fit
            **kwargs:  Additional keyword arguments. Allowed are:
                chi2_percentile: Tuple[float, float]: The percentile for the chi2 statistic
                width_percentile: Tuple[float, float]: The percentile for the width
                contrast_percentile: Tuple[float, float]: The percentile for the contrast

        """
        self.b111 = b111
        self.chi2 = chi2
        self.width = width
        self.mean_contrast = mean_contrast
        super().__init__(data_shape=b111.shape)
        self.LOG.info("initializing outlier detection with statistics")

        self.data_shape = self.b111.shape
        self.detected = False
        self._chi2_percentile = kwargs.pop(
            "chi2_percentile", self.settings["chi2_percentile"]
        )
        self._width_percentile = kwargs.pop(
            "width_percentile", self.settings["width_percentile"]
        )
        self._contrast_percentile = kwargs.pop(
            "contrast_percentile", self.settings["contrast_percentile"]
        )
        self._chi2_range = [self.chi2.min(), self.chi2.max()]
        self._width_range = [self.width.min(), self.width.max()]
        self._contrast_range = [self.mean_contrast.min(), self.mean_contrast.max()]
        self.chi2_outlier = np.zeros(self.b111.shape, dtype=bool)
        self.width_outlier = np.zeros(self.b111.shape, dtype=bool)
        self.contrast_outlier = np.zeros(self.b111.shape, dtype=bool)
        self.detect_outlier(
            self._chi2_percentile, self._width_percentile, self._contrast_percentile
        )

    # PROPERTIES AND SETTERS
    @property
    def chi2_percentile(self):
        """Getter for the percentile range of the chi2 statistic"""
        return self._chi2_percentile

    @chi2_percentile.setter
    def chi2_percentile(self, chi2_percentile):
        """Setter for the percentile range of the chi2 statistic"""
        self._chi2_percentile = chi2_percentile
        self.set_range("chi2", self.chi2, chi2_percentile)

    @property
    def width_percentile(self):
        """Getter for the percentile range of the width"""
        return self._width_percentile

    @width_percentile.setter
    def width_percentile(self, width_percentile):
        """Setter for the percentile range of the width"""
        self._width_percentile = width_percentile
        self.set_range("width", self.width, width_percentile)

    @property
    def contrast_percentile(self):
        """Getter for the percentile range of the contrast"""
        return self._contrast_percentile

    @contrast_percentile.setter
    def contrast_percentile(self, contrast_percentile):
        self._contrast_percentile = contrast_percentile
        self.set_range("contrast", self.mean_contrast, contrast_percentile)

    @property
    def chi2_range(self):
        return self._chi2_range

    @property
    def width_range(self):
        return self._width_range

    @property
    def contrast_range(self):
        return self._contrast_range

    def set_ranges(self, chi2_percentile, contrast_percentile, width_percentile):
        """
        Set the percentiles for chi2, contrast, and width.

        Args:
            chi2_percentile: Tuple of two floats that represent the lower and upper
                             percentiles for chi2.
            contrast_percentile: Tuple of two floats that represent the lower and upper
                                  percentiles for mean contrast.
            width_percentile: Tuple of two floats that represent the lower and upper
                              percentiles for width.
        """
        self.set_range("chi2", self.chi2, chi2_percentile)
        self.set_range("contrast", self.mean_contrast, contrast_percentile)
        self.set_range("width", self.width, width_percentile)

    def detect_outlier(
        self, chi2_percentile=None, width_percentile=None, contrast_percentile=None
    ):
        """
        Detect outliers in the statistics.

        Args:
            chi2_percentile: Tuple of two floats that represent the lower and upper
                             percentiles for chi2 (Default value = None).
            width_percentile: Tuple of two floats that represent the lower and upper
                              percentiles for width (Default value = None).
            contrast_percentile: Tuple of two floats that represent the lower and upper
                                  percentiles for mean contrast (Default value = None).

        Returns:
            np.ndarray: Array of booleans indicating whether each pixel is an outlier or not.
        """
        if not any(
            [
                chi2_percentile is None,
                width_percentile is None,
                contrast_percentile is None,
            ]
        ):
            self.set_ranges(chi2_percentile, contrast_percentile, width_percentile)

        self.chi2_outlier = self.get_outlier_from(self.chi2, self.chi2_range)
        self.width_outlier = self.get_outlier_from(self.width, self.width_range)
        self.contrast_outlier = self.get_outlier_from(
            self.mean_contrast, self.contrast_range
        )

        self.outliers = self.chi2_outlier | self.width_outlier | self.contrast_outlier
        self.detected = True
        return self.outliers

    def get_outlier_from(self, data, percentiles):
        """
        Determine whether each pixel is an outlier or not.

        Args:
            data: The statistics data to check for outliers.
            percentiles: The percentiles to use for outlier detection.

        Returns:
            np.ndarray: Array of booleans indicating whether each pixel is an outlier or not.
        """
        smaller = data < percentiles[0]
        larger = data > percentiles[1]
        return np.any(smaller, axis=(0, 1)) | np.any(larger, axis=(0, 1))

    def set_range(self, dtype: str, data: np.ndarray, percentile: Tuple[float, float]) -> np.ndarray:
        """
        Set the range for the specified dtype.

        Args:
            dtype: The data type for which the range should be set.
            data: The data from which the range should be set.
            percentile: The percentile values to be used for setting the range.

        Returns:
            The new range for the specified dtype.

        Raises:
            None.

        """
        data_range = np.percentile(data, percentile)
        self.LOG.debug(
            f"Setting {dtype} range to {data_range} out of {[data.min(), data.max()]} ({percentile})"
        )
        setattr(self, f"_{dtype}_range", data_range)
        if self.detected:
            self.LOG.warning(
                f"Parameter range for {dtype} changed, outlier detection needs to be rerun."
            )
            self.detected = False
        return data_range


class LocalOutlierFactor(Outlier):
    def concat_data(self, data: np.np.ndarray) -> np.np.ndarray:
        self.LOG.debug("concatenating data:")
        for i, d in enumerate(data):
            if d.ndim != 4:
                self.LOG.warning(f"dimension of data {i} is not 4")
                data[i] = d[..., np.newaxis]
            self.LOG.debug(f"shape of data {i}: {data[i].shape}")
        data = np.concatenate(data, axis=-1)
        self.LOG.debug(f"shape of concatenated data: {data.shape}")
        return data

    def __init__(self, data: ArrayLike, data_shape, **kwargs) -> None:
        data = self.concat_data(data)
        super().__init__(data_shape=data_shape, **kwargs)
        self.settings.update(kwargs)
        self.data = data

    def detect_outlier(self, **kwargs) -> np.ndarray:
        """
        Detect outliers using the LocalOutlierFactor algorithm.
        A pixel is considered and outlier if it is considered an outlier in one polarization or frequency range.

        Args:
            **kwargs: keyword arguments for the LocalOutlierFactor algorithm
                All keyword arguments are passed to the sklearn.neighbors.LocalOutlierFactor algorithm.
                See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

        Returns:
            np.ndarray: boolean array with the same shape as the data ([y,x]) with True for outliers
            and False for inliers

        """
        from sklearn.neighbors import LocalOutlierFactor as LOF

        self.settings.update(kwargs)
        initial_shape = self.data.shape
        data = self.data.reshape(-1, self.data.shape[-1])
        self.LOG.debug(f"detecting outliers with LOF on data {data.shape}.")
        lof = LOF(**self.settings, n_jobs=-1)
        self.outliers = lof.fit_predict(data)
        self.outliers = self.outliers.reshape(initial_shape[:-1])
        self.outliers = np.any(self.outliers == -1, axis=(0, 1))
        self.LOG.info(
            f"detected {self.outliers.sum()} outliers ({self.outliers.shape})"
        )
        self.outliers = self.outliers.reshape(self.data_shape)
        return self.outliers


def main():
    from scipy.io import loadmat

    d = loadmat("/home/mike/Desktop/b111test.b111")
    out = StatisticsPercentile(
        d["remanent"], d["chi_squares"], d["width"], np.mean(d["contrast"], axis=2)
    )


if __name__ == "__main__":
    main()
