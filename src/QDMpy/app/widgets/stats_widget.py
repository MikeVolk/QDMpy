import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout

from QDMpy.app.assets.GuiElements import LabeledDoubleSpinBox
from QDMpy.app.canvas import StatCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget


class StatisticsWidget(QDMWidget):
    def __init__(self, *args, **kwargs):
        canvas = StatCanvas(self)
        super().__init__(canvas=canvas, *args, **kwargs)
        self._add_b111_select_box(self.main_toolbar)
        self.setWindowTitle("Fit Statistics")

        self.chi2 = self.qdm.get_param("chi_squares")
        self.width = self.qdm.get_param("width") * 1000
        self.contrast = self.qdm.get_param("mean_contrast") * 100

        self.canvas.chi_ax.hist(self.chi2.flatten(), bins=400)
        self.canvas.chi_ax.set_yscale("log")
        self.canvas.width_ax.hist(self.width.flatten(), bins=400)
        self.canvas.width_ax.set_yscale("log")
        self.canvas.contrast_ax.hist(self.contrast.flatten(), bins=400)
        self.canvas.contrast_ax.set_yscale("log")

        for span in [
            "chi_min_span",
            "chi_max_span",
            "width_min_span",
            "width_max_span",
            "contrast_min_span",
            "contrast_max_span",
        ]:
            setattr(self, span, None)

        self.update_data()

        box = QGroupBox("Outlier removal")
        vlayout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.chi_checkbox, self.chi_min, self.chi_max, self.chi_rmin, self.chi_rmax = self.add_box("Χ²", 0)
        self.width_checkbox, self.width_min, self.width_max, self.width_rmin, self.width_rmax = self.add_box(
            "width", 1, "[MHz]"
        )
        (
            self.chi_checkbox,
            self.chi_min,
            self.chi_max,
            self.chi_rmin,
            self.chi_rmax,
        ) = self.add_box("Χ²", 0)
        (
            self.width_checkbox,
            self.width_min,
            self.width_max,
            self.width_rmin,
            self.width_rmax,
        ) = self.add_box("width", 1, "[MHz]")
        (
            self.contrast_checkbox,
            self.contrast_min,
            self.contrast_max,
            self.contrast_rmin,
            self.contrast_rmax,
        ) = self.add_box("contrast", 2, "[%]")

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

    def add_box(self, dtype, row, unit=""):
        # contrast BOX
        checkbox = QtWidgets.QCheckBox(f"  {dtype:>10}: ")
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.update_outlier_select)
        label_min, min = LabeledDoubleSpinBox(
            "min",
            value=0,
            decimals=2,
            step=0.1,
            vmin=0,
            vmax=100,
            callback=self.update_outlier_select,
        )
        label_max, max = LabeledDoubleSpinBox(
            "max",
            value=100,
            decimals=2,
            step=0.1,
            vmin=0,
            vmax=100,
            callback=self.update_outlier_select,
        )
        label2 = QLabel(f"{dtype:>10} range {unit}: ")
        label_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        rmin = QLabel("")
        rmin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        rmax = QLabel("")

        for i, w in enumerate(
            [
                checkbox,
                label_min,
                min,
                label_max,
                max,
                label2,
                rmin,
                rmax,
            ]
        ):
            self.grid_layout.addWidget(w, row, i)
        return checkbox, min, max, rmin, rmax

    def update_range_spans(
        self, chi_min, chi_max, width_min, width_max, contrast_min, contrast_max
    ):
        axes = [
            self.canvas.chi_ax,
            self.canvas.chi_ax,
            self.canvas.width_ax,
            self.canvas.width_ax,
            self.canvas.contrast_ax,
            self.canvas.contrast_ax,
        ]

        values = [chi_min, chi_max, width_min, width_max, contrast_min, contrast_max]
        minval = [
            self.chi2.min(),
            self.chi2.min(),
            self.width.min(),
            self.width.min(),
            self.contrast.min(),
            self.contrast.min(),
        ]

        maxval = [
            self.chi2.max(),
            self.chi2.max(),
            self.width.max(),
            self.width.max(),
            self.contrast.max(),
            self.contrast.max(),
        ]

        for i, lname in enumerate(
            [
                "chi_min_span",
                "chi_max_span",
                "width_min_span",
                "width_max_span",
                "contrast_min_span",
                "contrast_max_span",
            ]
        ):
            poly = getattr(self, lname)
            if "min" in lname:
                vmin, vmax = minval[i], values[i]
            else:
                vmin, vmax = values[i], maxval[i]
            if poly is None and vmin != vmax:
                poly = axes[i].axvspan(vmin, vmax, color="C03", alpha=0.1, ls="")
                setattr(self, lname, poly)
            elif poly is not None:
                xy = poly.get_xy()
                xy[0][0] = vmin
                xy[1][0] = vmin
                xy[2][0] = vmax
                xy[3][0] = vmax
                xy[4][0] = vmin
                poly.set_xy(xy)

    def update_outlier_select(self):
        chi_square_percentile = self.get_chi_percentile()
        width_percentile = self.get_width_percentile()
        contrast_percentile = self.get_contrast_percentile()
        self.update_range_spans(
            chi_square_percentile[0],
            chi_square_percentile[1],
            width_percentile[0],
            width_percentile[1],
            contrast_percentile[0],
            contrast_percentile[1],
        )

        self.LOG.debug(
            f"Percentiles have changed {chi_square_percentile} {width_percentile} {contrast_percentile}"
        )

        self.set_range_labels(
            chi_square_percentile, contrast_percentile, width_percentile
        )

        smaller_chi_square = self.chi2 < chi_square_percentile[0]
        larger_chi_square = self.chi2 > chi_square_percentile[1]
        smaller_width = self.width < width_percentile[0]
        larger_width = self.width > width_percentile[1]
        smaller_contrast = self.contrast < contrast_percentile[0]
        larger_contrast = self.contrast > contrast_percentile[1]
        outliers = (
            (
                np.any(smaller_chi_square, axis=(0, 1))
                | np.any(larger_chi_square, axis=(0, 1))
            )
            | np.any(smaller_width, axis=(0, 1))
            | np.any(larger_width, axis=(0, 1))
            | np.any(smaller_contrast, axis=(0, 1))
            | np.any(larger_contrast, axis=(0, 1))
        )

        self.canvas.update_outlier(outliers.reshape(*self.qdm.data_shape))
        self.outlier_count.setText(
            f"{np.sum(outliers)} ({(np.sum(outliers)/outliers.size)*100:.1f} %)"
        )

    def set_range_labels(
        self, chi_square_percentile, contrast_percentile, width_percentile
    ):
        self.chi_rmin.setText(f"{chi_square_percentile[0]:5.2e} -")
        self.chi_rmax.setText(f"{chi_square_percentile[1]:5.2e}")
        self.width_rmin.setText(f"{width_percentile[0] :5.2f} -")
        self.width_rmax.setText(f"{width_percentile[1] :5.2f}")
        self.contrast_rmin.setText(f"{contrast_percentile[0] :5.2f} -")
        self.contrast_rmax.setText(f"{contrast_percentile[1] :5.2f}")

    def get_contrast_percentile(self):
        if self.contrast_checkbox.isChecked():
            contrast_min, contrast_max = (
                self.contrast_min.value(),
                self.contrast_max.value(),
            )

        else:
            contrast_min, contrast_max = 0, 100
        return np.percentile(self.contrast, [contrast_min, contrast_max])

    def get_width_percentile(self):
        if self.width_checkbox.isChecked():
            width_min, width_max = self.width_min.value(), self.width_max.value()
        else:
            width_min, width_max = 0, 100
        return np.percentile(self.width, [width_min, width_max])

    def get_chi_percentile(self):
        if self.chi_checkbox.isChecked():
            chi_min, chi_max = self.chi_min.value(), self.chi_max.value()
        else:
            chi_min, chi_max = 0, 100
        return np.percentile(self.chi2, [chi_min, chi_max])
