import logging
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from PySide6.QtWidgets import (
    QLabel, QPushButton, QComboBox,
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
)
from quality_window import QualityWindow
from canvas import FittingPropertyCanvas

# sns.set_theme(font_scale=0.7)
from matplotlib import colors

from utils import polyfit2d
from pyqdm_plot_window import pyqdmWindow


class FitWindow(pyqdmWindow):
    def __init__(self, caller, *args, **kwargs):
        canvas = FittingPropertyCanvas(self, width=12, height=12, dpi=100)
        self._spectra_ax = [[canvas.left_ODMR_ax, canvas.right_ODMR_ax], [canvas.left_ODMR_ax, canvas.right_ODMR_ax]]

        super(FitWindow, self).__init__(caller, canvas, *args, **kwargs)
        self.data_img = None
        self._includes_fits = True
        self.quad_background = [None, None]  # for BG of ferro/induced

        self._add_b111_select_box(self.mainToolbar)
        self._add_subtract_box(self.mainToolbar)
        self._add_quality_button(self.mainToolbar)

        self._pixel_ax = np.array([[self.canvas.left_ODMR_ax, self.canvas.right_ODMR_ax],
                                   [self.canvas.left_ODMR_ax, self.canvas.right_ODMR_ax]])
        self.data_ax = self.canvas.main_ax
        self._outlier_masks = {self.data_ax: None}

        self._init_lines()

        self.add_light_img(self.canvas.led_ax)
        self.add_laser_img(self.canvas.laser_ax)
        self.add_mean_ODMR()

        self.qualityWindow = None

        self.update_img_plots()
        self.update_pixel_lines()
        self.update_fit_lines()
        self.canvas.draw()

    def _add_quality_button(self, toolbar):
        self.qualityButton = QPushButton('Quality')
        self.qualityButton.clicked.connect(self.onQualityClicked)
        toolbar.addWidget(self.qualityButton)

    def _add_b111_select_box(self, toolbar):
        b111Widget = QWidget()
        b111selectBox = QHBoxLayout()
        b111label = QLabel('B111: ')
        self.b111select = QComboBox()
        self.b111select.addItems(['remanent', 'induced'])
        self.b111select.currentIndexChanged.connect(self.update_img_plots)
        b111selectBox.addWidget(b111label)
        b111selectBox.addWidget(self.b111select)
        b111Widget.setLayout(b111selectBox)
        toolbar.addWidget(b111Widget)

    def _add_subtract_box(self, toolbar):
        subtractWidget = QWidget()
        bgCheckBox = QHBoxLayout()
        subtractLabel = QLabel('Subtract: ')
        self.subtractMedian = QCheckBox('median')
        self.subtractMedian.stateChanged.connect(self.onSubtractMedianClicked)
        self.subtractQuad = QCheckBox('quadratic')
        self.subtractQuad.stateChanged.connect(self.onSubtractQuadClicked)
        bgCheckBox.addWidget(subtractLabel)
        bgCheckBox.addWidget(self.subtractMedian)
        bgCheckBox.addWidget(self.subtractQuad)
        subtractWidget.setLayout(bgCheckBox)
        toolbar.addWidget(subtractWidget)

    def update_img_plots(self):
        d = self.QDMObj.B111[self.b111select.currentIndex()].copy()

        if self.subtractMedian.isChecked():
            self.LOG.debug('Subtracting median')
            d -= np.median(d)

        if self.subtractQuad.isChecked():
            self.LOG.debug('Subtracting Quad')
            d -= self.quad_background[self.b111select.currentIndex()]

        vmin, vmax = np.min(d), np.max(d)

        if self.fixClimCheckBox.isChecked():
            vmin, vmax = np.percentile(d, [(100 - self.cLimSelector.value()) / 2,
                                           (100 + self.cLimSelector.value()) / 2])

        if not (vmin < 0 < vmax):
            vcenter = (vmin + vmax) / 2
        else:
            vcenter = 0

        if self.data_img is None:
            self.data_img = self.data_ax.imshow(d, cmap='RdBu', interpolation='none', origin='lower',
                                                       norm=colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter))
        else:
            self.data_img.set_data(d)
            self.data_img.set(norm=colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter))

        self.canvas.cax.clear()
        self.canvas.cax.set_axes_locator(self.canvas.original_cax_locator)

        im_ratio = d.shape[0] / d.shape[1]
        self.data_cbar = plt.colorbar(self.data_img, cax=self.canvas.cax,  # fraction=0.047,# * im_ratio, pad=0.01,
                                      extend='both' if self.need_extend() else 'neither',
                                      label='B$_{111}$ [$\mu$T]')

        self.canvas.draw()

    def onSubtractMedianClicked(self):
        if self.subtractMedian.isChecked() and self.subtractQuad.isChecked():
            self.subtractQuad.setChecked(False)
        self.update_img_plots()

    def onSubtractQuadClicked(self):
        if self.subtractMedian.isChecked() and self.subtractQuad.isChecked():
            self.subtractMedian.setChecked(False)

        for i in range(2):
            if self.quad_background[i] is None or self.quad_background[i].shape != self.QDMObj.odmr.scan_dimensions:
                self.LOG.debug(f'Calculating quad background for {["remanent", "induced"][i]} component')
                x = np.arange(self.QDMObj.odmr.scan_dimensions[0])
                y = np.arange(self.QDMObj.odmr.scan_dimensions[1])
                kx = ky = 2
                solution = polyfit2d(x, y, self.QDMObj.B111[i], kx=kx, ky=ky)
                self.quad_background[i] = np.polynomial.polynomial.polygrid2d(x, y,
                                                                              solution[0].reshape((kx + 1, ky + 1)))

        self.update_img_plots()

    def onQualityClicked(self):
        if self.qualityWindow is None:
            self.qualityWindow = QualityWindow(self.caller, self.QDMObj)
            self.caller.qualityWindow = self.qualityWindow
        if self.qualityWindow.isVisible():
            self.qualityWindow.hide()
        else:
            self.qualityWindow.show()


class FluorescenceWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent,
    it will appear as a free-floating window.
    """
    LOG = logging.getLogger('pyqdm.app')

    def __init__(self, QDMObj):
        self.QDMObj = QDMObj
        print(self.QDMObj.odmr.f_Hz)
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Fluorescence Plots")

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.LOG.debug("FluorescenceWindow created")

        self.canvas.axes.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        self.setLayout(layout)
        self.resize(920, 600)
        self.canvas.draw()
