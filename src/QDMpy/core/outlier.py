import logging
from dataclasses import dataclass

import numpy as np

import QDMpy
import QDMpy.utils


@dataclass
class Outlier:
    b111: np.ndarray
    chi2: np.ndarray
    width: np.ndarray
    mean_contrast: np.ndarray
    LOG = logging.getLogger(f"QDMpy.{__name__}")


    def __post_init__(self):
        self.outlier: np.ndarray = np.zeros(self.b111.shape, dtype=bool)
        self.settings = None

    def detect_outlier(self, **kwargs):
        pass

    def __repr__(self):
        return f"Outlier(b111={self.b111.shape}, chi2={self.chi2.shape}, width={self.width.shape}, mean_contrast={self.mean_contrast.shape})"


class StatisticsPercentile(Outlier):

    def __post_init__(self):
        """
        Initialize the outlier detection.
        """
        self.LOG.info(f"initializing outlier detection using default settings")
        self.settings = QDMpy.settings["outlier_detection"][self.__class__.__name__]
        self.detected = False

        # set the percentiles
        self._chi2_percentile = self.settings["chi2_percentile"]
        self._width_percentile = self.settings["width_percentile"]
        self._contrast_percentile = self.settings["contrast_percentile"]

        # initialize the ranges
        self._chi2_range = [self.chi2.min(), self.chi2.max()]
        self._width_range = [self.width.min(), self.width.max()]
        self._contrast_range = [self.mean_contrast.min(), self.mean_contrast.max()]

        # initialize arrays
        self.chi2_outlier = np.zeros(self.b111.shape, dtype=bool)
        self.width_outlier = np.zeros(self.b111.shape, dtype=bool)
        self.contrast_outlier = np.zeros(self.b111.shape, dtype=bool)
        # detect outliers
        self.detect_outlier(self._chi2_percentile, self._width_percentile, self._contrast_percentile)

    # PROPERTIES AND SETTERS
    @property
    def chi2_percentile(self):
        return self._chi2_percentile

    @chi2_percentile.setter
    def chi2_percentile(self, chi2_percentile):
        self._chi2_percentile = chi2_percentile
        self.set_range("chi2", self.chi2, chi2_percentile)

    @property
    def width_percentile(self):
        return self._width_percentile

    @width_percentile.setter
    def width_percentile(self, width_percentile):
        self._width_percentile = width_percentile
        self.set_range("width", self.width, width_percentile)

    @property
    def contrast_percentile(self):
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

    @property
    def n(self):
        return sum(self.outliers)
    # METHODS

    def set_ranges(self, chi2_percentile, contrast_percentile, width_percentile):
        self.set_range("chi2", self.chi2, chi2_percentile)
        self.set_range("contrast", self.mean_contrast, contrast_percentile)
        self.set_range("width", self.width, width_percentile)

    def detect_outlier(self, chi2_percentile=None, width_percentile=None, contrast_percentile=None):
        """
        Detect outliers in the statistics.

        Parameters
        ----------
        chi2_percentile : [float, float], optional
            The upper and lower percentile for the chi2 statistic. The default is None.
        width_percentile : [float, float], optional
            The upper and lower percentile for the width. The default is None.
        contrast_percentile : [float, float], optional
            The upper and lower percentile for the contrast. The default is None.
        """

        # if all are None it uses the internal values
        # so that the outliers can be recalculated
        if not all([chi2_percentile is None, width_percentile is None, contrast_percentile is None]):
            self.set_ranges(chi2_percentile, contrast_percentile, width_percentile)

        self.chi2_outlier = self.get_outlier_from(self.chi2, self.chi2_range)
        self.width_outlier = self.get_outlier_from(self.width, self.width_range)
        self.contrast_outlier = self.get_outlier_from(self.mean_contrast, self.contrast_range)

        self.outliers = self.chi2_outlier | self.width_outlier | self.contrast_outlier
        self.detected = True
        return self.outliers

    def get_outlier_from(self, data, percentiles):
        smaller = data < percentiles[0]
        larger = data > percentiles[1]
        return np.any(smaller, axis=(0, 1)) | np.any(larger, axis=(0, 1))

    def set_range(self, dtype, data, percentile):
        data_range = np.percentile(data, percentile)
        self.LOG.debug(f"setting {dtype} range to {data_range} out of {[data.min(), data.max()]} ({percentile})")
        setattr(self, f"_{dtype}_range", data_range)
        if self.detected:
            self.LOG.warning(f"parameter range for {dtype} changed, outlier detection needs to be rerun")
            self.detected = False
        return data_range

def main():
    from scipy.io import loadmat

    d = loadmat("/home/mike/Desktop/b111test.b111")
    out = StatisticsPercentile(d["remanent"], d["chi_squares"], d["width"], np.mean(d["contrast"], axis=2))


if __name__ == "__main__":
    main()
