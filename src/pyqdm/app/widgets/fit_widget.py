import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from pyqdm.app.canvas import FitCanvas
from pyqdm.app.widgets.qdm_widget import PyQdmWindow
from pyqdm.app.widgets.quality_widget import QualityWidget
from pyqdm.utils import polyfit2d

B111 = "B$_{111}$"


class FitWidget(PyQdmWindow):
    def __init__(self, *args, **kwargs):
        canvas = FitCanvas(self, width=12, height=12, dpi=100)

        super().__init__(canvas=canvas, *args, **kwargs)

        self._add_b111_select_box(self.mainToolbar)
        self._add_subtract_box(self.mainToolbar)
        self._add_quality_button(self.mainToolbar)

        self.update_data()
        self.add_light()
        self.add_laser()
        self.add_odmr()

        self.quad_background = [None, None]  # for BG of ferro/induced

        self.set_main_window()
        self.update_clims()

    def _calculate_quad_background(self):
        for i in range(2):
            if self.quad_background[i] is None or any(self.quad_background[i].shape != self.qdm.odmr.data_shape):
                self.LOG.debug(f'Calculating quad background for {["remanent", "induced"][i]} component')
                x = np.arange(self.qdm.odmr.data_shape[0])
                y = np.arange(self.qdm.odmr.data_shape[1])
                kx = ky = 2
                solution = polyfit2d(x, y, self.qdm.b111[i], kx=kx, ky=ky)
                self.quad_background[i] = np.polynomial.polynomial.polygrid2d(
                    x, y, solution[0].reshape((kx + 1, ky + 1))
                )

    def add_odmr(self, mean=False):
        self.canvas.update_odmr(
            self.qdm.odmr.f_ghz,
            data=self.get_current_odmr(),
            fit=self.get_current_fit(),
            mean=self.qdm.odmr.mean_odmr if mean else None,
        )

    def update_odmr(self):
        """
        Update the marker position on the image plots.
        """
        print(self.canvas.odmr)
        self.canvas.update_odmr(freq=self.qdm.odmr.f_ghz, data=self.get_corrected_odmr(), fit=self.get_current_fit())
        self.canvas.update_odmr_lims()

    def on_subtract_median_clicked(self):
        if self.subtract_median.isChecked() and self.subtract_quad.isChecked():
            self.subtract_quad.setChecked(False)
        self.update_data()

    def on_subtract_quad_clicked(self):
        if self.subtract_median.isChecked() and self.subtract_quad.isChecked():
            self.subtract_median.setChecked(False)
        self._calculate_quad_background()
        self.update_data()

    def update_data(self):

        d = self.qdm.b111[self.b111_select.currentIndex()].copy()

        if self.subtract_median.isChecked():
            self.LOG.debug("Subtracting median")
            d -= np.median(d)

        if self.subtract_quad.isChecked():
            self.LOG.debug("Subtracting Quad")
            d -= self.quad_background[self.b111_select.currentIndex()]

        self.canvas.add_data(d, self.qdm.data_shape)

        # set the colorbar label
        self.canvas.data_ax.set_title(f"{B111}({self.b111_select.currentText()[:3]}.)")

        self.update_clims()
        self.canvas.draw()

    def _add_quality_button(self, toolbar):
        self.qualityButton = QPushButton("Quality")
        # self.qualityButton.clicked.connect(self.on_quality_clicked)
        toolbar.addWidget(self.qualityButton)

    def _add_b111_select_box(self, toolbar):
        b111_widget = QWidget()
        b111select_box = QHBoxLayout()
        b111label = QLabel("B111: ")
        self.b111_select = QComboBox()
        self.b111_select.addItems(["remanent", "induced"])
        self.b111_select.currentIndexChanged.connect(self.update_data)
        b111select_box.addWidget(b111label)
        b111select_box.addWidget(self.b111_select)
        b111_widget.setLayout(b111select_box)
        toolbar.addWidget(b111_widget)

    def _add_subtract_box(self, toolbar):
        subtract_widget = QWidget()
        bg_check_box = QHBoxLayout()
        subtract_label = QLabel("Subtract: ")
        self.subtract_median = QCheckBox("median")
        self.subtract_median.stateChanged.connect(self.on_subtract_median_clicked)
        self.subtract_quad = QCheckBox("quadratic")
        self.subtract_quad.stateChanged.connect(self.on_subtract_quad_clicked)
        bg_check_box.addWidget(subtract_label)
        bg_check_box.addWidget(self.subtract_median)
        bg_check_box.addWidget(self.subtract_quad)
        subtract_widget.setLayout(bg_check_box)
        toolbar.addWidget(subtract_widget)


class FitWidgetOLD(PyQdmWindow):
    def __init__(self, caller, *args, **kwargs):
        canvas = FittingPropertyCanvas(self, width=12, height=12, dpi=100)
        self._spectra_ax = [
            [canvas.left_odmr_ax, canvas.right_odmr_ax],
            [canvas.left_odmr_ax, canvas.right_odmr_ax],
        ]

        super().__init__(caller, canvas, *args, **kwargs)
        self.data_img = None
        self._includes_fits = True
        self.quad_background = [None, None]  # for BG of ferro/induced

        self._add_b111_select_box(self.mainToolbar)
        self._add_subtract_box(self.mainToolbar)
        self._add_quality_button(self.mainToolbar)

        self._pixel_ax = np.array(
            [
                [self.canvas.left_odmr_ax, self.canvas.right_odmr_ax],
                [self.canvas.left_odmr_ax, self.canvas.right_odmr_ax],
            ]
        )
        self.data_ax = self.canvas.data_ax
        self._outlier_masks = {self.data_ax: None}

        self._init_lines()

        self.add_light_img(self.canvas.light_ax)
        self.add_laser_img(self.canvas.laser_ax)
        self.add_mean_odmr()

        self.qualityWindow = None

        self.update_img_plots()
        self.update_pixel_lines()
        self.update_fit_lines()
        self.canvas.draw()

    def _add_quality_button(self, toolbar):
        self.qualityButton = QPushButton("Quality")
        self.qualityButton.clicked.connect(self.on_quality_clicked)
        toolbar.addWidget(self.qualityButton)

    def _add_b111_select_box(self, toolbar):
        b111_widget = QWidget()
        b111select_box = QHBoxLayout()
        b111label = QLabel("B111: ")
        self.b111select = QComboBox()
        self.b111select.addItems(["remanent", "induced"])
        self.b111select.currentIndexChanged.connect(self.update_img_plots)
        b111select_box.addWidget(b111label)
        b111select_box.addWidget(self.b111select)
        b111_widget.setLayout(b111select_box)
        toolbar.addWidget(b111_widget)

    def _add_subtract_box(self, toolbar):
        subtract_widget = QWidget()
        bg_check_box = QHBoxLayout()
        subtract_label = QLabel("Subtract: ")
        self.subtractMedian = QCheckBox("median")
        self.subtractMedian.stateChanged.connect(self.on_subtract_median_clicked)
        self.subtractQuad = QCheckBox("quadratic")
        self.subtractQuad.stateChanged.connect(self.on_subtract_quad_clicked)
        bg_check_box.addWidget(subtract_label)
        bg_check_box.addWidget(self.subtractMedian)
        bg_check_box.addWidget(self.subtractQuad)
        subtract_widget.setLayout(bg_check_box)
        toolbar.addWidget(subtract_widget)

    def update_img_plots(self):
        d = self.qdm.b111[self.b111select.currentIndex()].copy()

        if self.subtractMedian.isChecked():
            self.LOG.debug("Subtracting median")
            d -= np.median(d)

        if self.subtractQuad.isChecked():
            self.LOG.debug("Subtracting Quad")
            d -= self.quad_background[self.b111select.currentIndex()]

        vmin, vmax = np.min(d), np.max(d)

        if self.fix_clim_check_box.isChecked():
            vmin, vmax = np.percentile(
                d,
                [
                    (100 - self.clims_selector.value()) / 2,
                    (100 + self.clims_selector.value()) / 2,
                ],
            )

        vcenter = 0 if vmin < 0 < vmax else (vmin + vmax) / 2

        if self.data_img is None:
            self.data_img = self.data_ax.imshow(
                d,
                cmap="RdBu",
                interpolation="none",
                origin="lower",
                norm=colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter),
            )
        else:
            self.data_img.set_data(d)
            self.data_img.set(norm=colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter))

        self.canvas.cbar.clear()
        self.canvas.cbar.set_axes_locator(self.canvas.original_cax_locator)

        # noinspection PyPep8
        self.data_cbar = plt.colorbar(
            self.data_img,
            cax=self.canvas.cax,
            extend="both" if self.need_extend() else "neither",
            label=r"B$_{111}$ [$\mu$T]",
        )

        self.canvas.draw()

    def on_subtract_median_clicked(self):
        if self.subtractMedian.isChecked() and self.subtractQuad.isChecked():
            self.subtractQuad.setChecked(False)
        self.update_img_plots()

    def on_subtract_quad_clicked(self):
        if self.subtractMedian.isChecked() and self.subtractQuad.isChecked():
            self.subtractMedian.setChecked(False)

        self._calculate_quad_background()

        self.update_img_plots()

    def _calculate_quad_background(self):
        for i in range(2):
            if self.quad_background[i] is None or any(self.quad_background[i].shape != self.qdm.odmr.scan_dimensions):
                self.LOG.debug(f'Calculating quad background for {["remanent", "induced"][i]} component')
                x = np.arange(self.qdm.odmr.scan_dimensions[0])
                y = np.arange(self.qdm.odmr.scan_dimensions[1])
                kx = ky = 2
                solution = polyfit2d(x, y, self.qdm.b111[i], kx=kx, ky=ky)
                self.quad_background[i] = np.polynomial.polynomial.polygrid2d(
                    x, y, solution[0].reshape((kx + 1, ky + 1))
                )

    def on_quality_clicked(self):
        if self.qualityWindow is None:
            self.qualityWindow = QualityWidget(self.caller, self.qdm)
            self.caller.qualityWindow = self.qualityWindow
        if self.qualityWindow.isVisible():
            self.qualityWindow.hide()
        else:
            self.qualityWindow.show()
