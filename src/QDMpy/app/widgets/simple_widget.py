import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider

from QDMpy.app.assets.GuiElements import LabeledDoubleSpinBox
from QDMpy.app.canvas import SimpleCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget


class SimpleWidget(QDMWidget):
    def __init__(self, dtype, set_main_window=True, *args, **kwargs):
        canvas = SimpleCanvas(dtype=dtype)

        super().__init__(canvas=canvas, *args, **kwargs)

        if dtype == "laser":
            self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)
            self.canvas.fig.subplots_adjust(top=0.97, bottom=0.124, left=0.049, right=0.951, hspace=0.2, wspace=0.2)
            self.setWindowTitle("Laser scan")
        elif dtype == "light":
            self.canvas.add_light(self.qdm.light, self.qdm.data_shape)
            self.setWindowTitle("Reflected light")
        elif dtype == "outlier":
            self.canvas.add_data(self.qdm.b111[0], self.qdm.data_shape)
            self.setWindowTitle("Reflected light")
        else:
            raise ValueError(f"dtype {dtype} not recognized")

        if set_main_window:
            self.set_main_window()
        self.update_clims()
        self.add_scalebars()
        self.canvas.draw_idle()


class StatisticsWidget(SimpleWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(dtype="outlier", set_main_window=False, *args, **kwargs)
        self.chi2 = self.qdm.get_param("chi_squares")
        self.width = self.qdm.get_param("width")
        self.contrast = self.qdm.get_param("mean_contrast")

        box = QGroupBox("Outlier removal")
        vlayout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.add_chi_box()
        self.add_width_box()
        self.add_contrast_box()
        vlayout.addLayout(self.grid_layout)

        # PIXEL COUNTER
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Number of detected outliers: "))
        self.outlier_count = QLabel("0")
        hbox.addWidget(self.outlier_count)
        vlayout.addLayout(hbox)
        box.setLayout(vlayout)

        self.mainVerticalLayout.addWidget(box)
        self.set_main_window()
        self.update_outlier_select()
        self.canvas.draw()

    def add_contrast_box(self):
        # contrast BOX
        self.contrast_checkbox = QtWidgets.QCheckBox("  contrast: ")
        self.contrast_checkbox.setChecked(True)

        self.contrast_checkbox.stateChanged.connect(self.update_outlier_select)
        label_min, self.contrast_min = LabeledDoubleSpinBox(
            "min", value=0, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        label_max, self.contrast_max = LabeledDoubleSpinBox(
            "max", value=100.1, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        contrast_label2 = QLabel("contrast range: ")
        label_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        contrast_label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.contrast_range_min = QLabel("")
        self.contrast_range_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.contrast_range_max = QLabel("")

        for i, w in enumerate(
            [
                self.contrast_checkbox,
                label_min,
                self.contrast_min,
                label_max,
                self.contrast_max,
                contrast_label2,
                self.contrast_range_min,
                self.contrast_range_max,
            ]
        ):
            self.grid_layout.addWidget(w, 2, i)

    def add_width_box(self):
        self.width = self.qdm.get_param("width")
        self.width_checkbox = QtWidgets.QCheckBox("  width: ")
        self.width_checkbox.setChecked(True)
        self.width_checkbox.stateChanged.connect(self.update_outlier_select)

        label_min, self.width_min = LabeledDoubleSpinBox(
            "min", value=0, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        label_max, self.width_max = LabeledDoubleSpinBox(
            "max", value=100.1, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        width_label2 = QLabel("width range [MHz]: ")
        label_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        width_label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.width_range_min = QLabel("")
        self.width_range_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.width_range_max = QLabel("")

        for i, w in enumerate(
            [
                self.width_checkbox,
                label_min,
                self.width_min,
                label_max,
                self.width_max,
                width_label2,
                self.width_range_min,
                self.width_range_max,
            ]
        ):
            self.grid_layout.addWidget(w, 1, i)

    def add_chi_box(self):

        self.chi_checkbox = QtWidgets.QCheckBox("  Χ²: ")
        self.chi_checkbox.setChecked(True)
        self.chi_checkbox.stateChanged.connect(self.update_outlier_select)

        label_min, self.chi_min = LabeledDoubleSpinBox(
            "min", value=0, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        label_max, self.chi_max = LabeledDoubleSpinBox(
            "max", value=100.1, decimals=2, step=0.1, vmin=0, vmax=100, callback=self.update_outlier_select
        )
        chi_label2 = QLabel("Χ² range: ")
        label_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        chi_label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.chi_range_min = QLabel("")
        self.chi_range_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.chi_range_max = QLabel("")

        for i, w in enumerate(
            [
                self.chi_checkbox,
                label_min,
                self.chi_min,
                label_max,
                self.chi_max,
                chi_label2,
                self.chi_range_min,
                self.chi_range_max,
            ]
        ):
            self.grid_layout.addWidget(w, 0, i)

    def update_outlier_select(self):
        chi_square_percentile = self.get_chi_percentile()
        width_percentile = self.get_width_percentile()
        contrast_percentile = self.get_contrast_percentile()

        self.set_range_labels(chi_square_percentile, contrast_percentile, width_percentile)

        smaller_chi_square = self.chi2 < chi_square_percentile[0]
        larger_chi_square = self.chi2 > chi_square_percentile[1]
        smaller_width = self.width < width_percentile[0]
        larger_width = self.width > width_percentile[1]
        smaller_contrast = self.contrast < contrast_percentile[0]
        larger_contrast = self.contrast > contrast_percentile[1]

        outliers = (
            np.any(smaller_chi_square, axis=(0, 1))
            | np.any(larger_chi_square, axis=(0, 1))
            | np.any(smaller_width, axis=(0, 1))
            | np.any(larger_width, axis=(0, 1))
            | np.any(smaller_contrast, axis=(0, 1))
            | np.any(larger_contrast, axis=(0, 1))
        )

        for i in [
            smaller_contrast,
            larger_contrast,
            smaller_width,
            larger_width,
            smaller_chi_square,
            larger_chi_square,
        ]:
            self.canvas.update_outlier(outliers.reshape(*self.qdm.data_shape))
            self.outlier_count.setText(f"{np.sum(outliers)}")

    def set_range_labels(self, chi_square_percentile, contrast_percentile, width_percentile):
        self.chi_range_min.setText(f"{chi_square_percentile[0]:5.2e} - ")
        self.chi_range_max.setText(f"{chi_square_percentile[1]:5.2e}")
        self.width_range_min.setText(f"{width_percentile[0]*1e3:5.2f} - ")
        self.width_range_max.setText(f"{width_percentile[1]*1e3:5.2f}")
        self.contrast_range_min.setText(f"{contrast_percentile[0]*100:5.2f} - ")
        self.contrast_range_max.setText(f"{contrast_percentile[1]*100:5.2f}")

    def get_contrast_percentile(self):
        if self.contrast_checkbox.isChecked():
            contrast_min, contrast_max = self.contrast_min.value(), self.contrast_max.value()
        else:
            contrast_min, contrast_max = 0, 100
        contrast_percentile = np.percentile(self.contrast, [contrast_min, contrast_max])
        return contrast_percentile

    def get_width_percentile(self):
        if self.width_checkbox.isChecked():
            width_min, width_max = self.width_min.value(), self.width_max.value()
        else:
            width_min, width_max = 0, 100
        width_percentile = np.percentile(self.width, [width_min, width_max])
        return width_percentile

    def get_chi_percentile(self):
        if self.chi_checkbox.isChecked():
            chi_min, chi_max = self.chi_min.value(), self.chi_max.value()
        else:
            chi_min, chi_max = 0, 100
        chi_square_percentile = np.percentile(self.chi2, [chi_min, chi_max])
        return chi_square_percentile
