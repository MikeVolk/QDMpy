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

from pyqdm.app.canvas import FittingPropertyCanvas
from pyqdm.app.windows.pyqdm_plot_window import PyQdmWindow
from pyqdm.app.windows.quality_window import QualityWindow
from pyqdm.utils import polyfit2d


class FitWindow(PyQdmWindow):
    def __init__(self, caller, *args, **kwargs):
        canvas = FittingPropertyCanvas(self, width=12, height=12, dpi=100)
        self._spectra_ax = [
            [canvas.left_ODMR_ax, canvas.right_ODMR_ax],
            [canvas.left_ODMR_ax, canvas.right_ODMR_ax],
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
                [self.canvas.left_ODMR_ax, self.canvas.right_ODMR_ax],
                [self.canvas.left_ODMR_ax, self.canvas.right_ODMR_ax],
            ]
        )
        self.data_ax = self.canvas.main_ax
        self._outlier_masks = {self.data_ax: None}

        self._init_lines()

        self.add_light_img(self.canvas.led_ax)
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
                    (100 - self.cLimSelector.value()) / 2,
                    (100 + self.cLimSelector.value()) / 2,
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

        self.canvas.cax.clear()
        self.canvas.cax.set_axes_locator(self.canvas.original_cax_locator)

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

        for i in range(2):
            if self.quad_background[i] is None or self.quad_background[i].shape != self.qdm.odmr.scan_dimensions:
                self.LOG.debug(f'Calculating quad background for {["remanent", "induced"][i]} component')
                x = np.arange(self.qdm.odmr.scan_dimensions[0])
                y = np.arange(self.qdm.odmr.scan_dimensions[1])
                kx = ky = 2
                solution = polyfit2d(x, y, self.qdm.b111[i], kx=kx, ky=ky)
                self.quad_background[i] = np.polynomial.polynomial.polygrid2d(
                    x, y, solution[0].reshape((kx + 1, ky + 1))
                )

        self.update_img_plots()

    def on_quality_clicked(self):
        if self.qualityWindow is None:
            self.qualityWindow = QualityWindow(self.caller, self.qdm)
            self.caller.qualityWindow = self.qualityWindow
        if self.qualityWindow.isVisible():
            self.qualityWindow.hide()
        else:
            self.qualityWindow.show()
