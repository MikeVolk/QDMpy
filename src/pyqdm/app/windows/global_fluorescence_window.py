import logging
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QSlider, QPushButton, QDoubleSpinBox,
    QWidget, QVBoxLayout, QHBoxLayout
)
from pyqdm.app.canvas import GlobalFluorescenceCanvas

import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
from pyqdm.app.windows.misc import GFAppliedWindow

matplotlib.rcParams.update({  # 'font.size': 8,
    # 'axes.labelsize': 8,
    'grid.linestyle': '-',
    'grid.alpha': 0.5})


class GlobalFluorescenceWindow(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    def on_press(self, event):
        if event.inaxes in self.pixel_axes:
            self.LOG.debug(f'clicked in {event.inaxes}')
            return
        if event.xdata is None or event.ydata is None:
            self.LOG.debug('clicked outside of axes')
            return
        if event.button == MouseButton.LEFT and not self.toolbar.mode:
            bin_factor = self.QDMObj.bin_factor
            xy = [event.xdata / bin_factor, event.ydata / bin_factor]
            x, y = np.round(xy).astype(int)
            self.xselect.valueChanged.disconnect(self.onXYValueChange)
            self.xselect.setValue(x)
            self.xselect.valueChanged.connect(self.onXYValueChange)
            self.yselect.valueChanged.disconnect(self.onXYValueChange)
            self.yselect.setValue(y)
            self.yselect.valueChanged.connect(self.onXYValueChange)
            self.LOG.debug(f'clicked in {event.inaxes} with new index: {self._current_idx}')

            self._current_idx = self.QDMObj.odmr.rc2idx([y, x])
            self.update_marker()
            self.update_plots()

    def get_label_box(self, label, value, decimals, step, min, max, callback):
        label = QLabel(label)
        selector = QDoubleSpinBox()
        selector.setValue(value)
        selector.setDecimals(decimals)
        selector.setSingleStep(step)
        selector.setMinimum(min)
        selector.setMaximum(max)
        selector.setKeyboardTracking(False)
        selector.valueChanged.connect(callback)
        return label, selector

    @property
    def _current_xy(self):
        return self.QDMObj.odmr.idx2rc(self._current_idx)[::-1]


    def __init__(self, main_window, QDMObj=None, pixelsize=1e-6, *args, **kwargs):
        self.LOG = logging.getLogger(f'pyqdm.{self.__class__.__name__}')
        self.main_window = main_window
        self.QDMObj = QDMObj
        self.pixelsize = pixelsize

        super(GlobalFluorescenceWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Global Fluorescence Estimation')

        # Create the maptlotlib FigureCanvas object,
        self.canvas = GlobalFluorescenceCanvas(self, width=12, height=6, dpi=100)
        cid = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.pixel_axes = [self.canvas.left_meanODMR_ax, self.canvas.right_meanODMR_ax]
        self.img_axes = [self.canvas.led_ax, self.canvas.laser_ax]


        self._current_idx = self.QDMObj.odmr.get_most_divergent_from_mean()[-1]
        self.LOG.debug(f'setting index of worst pixel to {self._current_xy} ({self._current_idx})')

        verticalLayout = QVBoxLayout()
        horizontalLayoutTop = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        horizontalLayoutTop.addWidget(self.toolbar)

        label = QLabel('Data pixel:')
        horizontalLayoutTop.addWidget(label)
        self.xlabel, self.xselect = self.get_label_box('x', int(self._current_xy[0]), 0, 1, 0, self.QDMObj.odmr.scan_dimensions[0], self.onXYValueChange)
        horizontalLayoutTop.addWidget(self.xlabel)
        horizontalLayoutTop.addWidget(self.xselect)
        self.ylabel, self.yselect = self.get_label_box('y', int(self._current_xy[1]), 0, 1, 0, self.QDMObj.odmr.scan_dimensions[1], self.onXYValueChange)

        self.xselect.valueChanged.disconnect(self.onXYValueChange)
        self.xselect.setValue(int(self._current_xy[0]))
        self.xselect.valueChanged.connect(self.onXYValueChange)

        self.yselect.valueChanged.disconnect(self.onXYValueChange)
        self.yselect.setValue(int(self._current_xy[1]))
        self.yselect.valueChanged.connect(self.onXYValueChange)

        self.indexLabel = QLabel(f'({self._current_idx}')

        horizontalLayoutTop.addWidget(self.ylabel)
        horizontalLayoutTop.addWidget(self.yselect)
        horizontalLayoutTop.addWidget(self.indexLabel)

        verticalLayout.addLayout(horizontalLayoutTop)
        verticalLayout.addWidget(self.canvas)

        self.gf_label = QLabel(f'Global Fluorescence: {self.QDMObj.global_factor:.2f}')
        self.gfSlider = QSlider()
        self.gfSlider.setValue(self.main_window.gf_select.value())
        self.gfSlider.setRange(0, 100)
        self.gfSlider.setOrientation(Qt.Horizontal)
        self.gfSlider.valueChanged.connect(self.onSliderValueChanged)
        self.applyButton = QPushButton('Apply')
        self.applyButton.clicked.connect(self.apply_global_factor)

        horizontalLayout = QHBoxLayout()
        horizontalLayout.addWidget(self.gf_label)
        horizontalLayout.addWidget(self.gfSlider)
        horizontalLayout.addWidget(self.applyButton)

        verticalLayout.addLayout(horizontalLayout)
        mainWidget = QWidget()
        mainWidget.setLayout(verticalLayout)

        self.setCentralWidget(mainWidget)

        self._corrected_lines = [[None, None], [None, None]]
        self._pixel_lines = [[None, None], [None, None]]
        self._uncorrected_lines = [[None, None], [None, None]]
        self._marker_line = [None, None]

        self.init_plots()
        self.resize(1000, 700)
        self.show()

    def init_plots(self):
        pixel_spectra, uncorrected, corrected, mn, mx= self.get_pixel_data()

        pols = ['+', '-']

        for f in np.arange(self.QDMObj.odmr.n_frange):
            for p in np.arange(self.QDMObj.odmr.n_pol):
                ul, = self.pixel_axes[f].plot(self.QDMObj.odmr.f_GHz[f], uncorrected[p, f],
                                             '.--', mfc='w', label=f"{pols[p]} original", lw=0.8)
                self._uncorrected_lines[p][f] = ul

                cl, = self.pixel_axes[f].plot(self.QDMObj.odmr.f_GHz[f], corrected[p, f],
                                             '.-', label=f"{pols[p]} corrected", color=ul.get_color())
                self._corrected_lines[p][f] = cl

                if self.QDMObj.odmr._gf_factor != 0:
                    pl, = self.pixel_axes[f].plot(self.QDMObj.odmr.f_GHz[f], pixel_spectra[p,f], ':',
                                            label=f"{pols[p]} current: GF={self.QDMObj.odmr._gf_factor}",
                                            color = ul.get_color(), lw=0.8)
                    self._pixel_lines[p][f] = pl

            if self.QDMObj.odmr._gf_factor != 0:
                h, l = self.pixel_axes[f].get_legend_handles_labels()
                h = np.array(h).reshape((2, -1)).T.flatten()
                l = np.array(l).reshape((2, -1)).T.flatten()
                self.pixel_axes[f].legend(h, l, ncol=3,
                                          bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0., frameon=False,
                                          prop={'family': 'DejaVu Sans Mono'})
            else:
                self.pixel_axes[f].legend(ncol=2, bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0., frameon=False,
                                          prop={'family': 'DejaVu Sans Mono'})

            self.pixel_axes[f].set(ylabel="ODMR contrast", xlabel="Frequency [GHz]", ylim=(mn, mx))

        self.img_axes[0].imshow(self.QDMObj.led, cmap='gray', interpolation='none', origin='lower')
        self.img_axes[1].imshow(self.QDMObj.laser, cmap='inferno', interpolation='none', origin='lower')
        self._marker_line[0], = self.img_axes[0].plot(self._current_xy[1]*self.QDMObj.bin_factor, self._current_xy[0]*self.QDMObj.bin_factor, 'cx', markersize=5,
                                                      zorder=10)
        self._marker_line[1], = self.img_axes[1].plot(self._current_xy[1]*self.QDMObj.bin_factor, self._current_xy[0]*self.QDMObj.bin_factor, 'cx', markersize=5,
                                                      zorder=10)
        self.img_axes[0].set(xlabel="x [px]", ylabel="y [px]")
        self.img_axes[1].set(xlabel="x [px]", ylabel="y [px]")
        self._add_scalebars()
        self.canvas.draw()

    def get_pixel_data(self):
        gf_factor = self.gfSlider.value() / 100
        new_correct = self.QDMObj.odmr._get_gf_correction(gf=gf_factor)
        old_correct = self.QDMObj.odmr._get_gf_correction(gf=self.QDMObj.odmr._gf_factor)
        pixel_spectra = self.QDMObj.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        uncorrected = pixel_spectra + old_correct
        corrected = uncorrected - new_correct
        mn = np.min([np.min(pixel_spectra), np.min(corrected), np.min(uncorrected)]) * 0.998
        mx = np.max([np.max(pixel_spectra), np.max(corrected), np.max(uncorrected)]) * 1.002
        return  pixel_spectra, uncorrected, corrected, mn, mx

    def _add_scalebars(self):
        for ax in self.img_axes:
            # Create scale bar
            scalebar = ScaleBar(self.pixelsize, "m", length_fraction=0.25,
                                location="lower left")
            ax.add_artist(scalebar)

    def update_pixel_spectrum(self, x, y):
        """
        Update the pixel spectrum plot with the current pixel position.
        Adds a marker to the plot and adds the pixel spectrum to the mean ODMR plot.

        """
        if self.toolbar.mode:
            return

        idx = self.QDMObj.odmr.rc2idx([y, x])  # get the index of the current pixel
        labels = ['p(<+', 'p(<-', 'p(>+', 'p(>-']
        # update the pixel spectrum plot
        for l in [self.low_pos_pixel, self.low_neg_pixel, self.high_pos_pixel, self.high_neg_pixel]:
            l.set_data(x, y)

        # update the mean ODMR plot legend
        h, l = self.canvas.lowF_meanODMR_ax.get_legend_handles_labels()
        h.extend([self.low_pos_pixel_line, self.low_neg_pixel_line])
        l.extend([f'{labels[0]},{x},{y})', f'{labels[1]},{x},{y})'])
        self.canvas.lowF_meanODMR_ax.legend(h, l, loc='lower left', fontsize=8)

        h, l = self.canvas.highF_meanODMR_ax.get_legend_handles_labels()
        h.extend([self.high_pos_pixel_line, self.high_neg_pixel_line])
        l.extend([f'{labels[2]},{x},{y})', f'{labels[3]},{x},{y})'])
        self.canvas.highF_meanODMR_ax.legend(h, l, loc='lower left', fontsize=8)

        # add lines to mean ODMR plot
        self.low_pos_pixel_line.set_ydata(self.QDMObj.odmr.data[0, 0, idx])
        self.low_neg_pixel_line.set_ydata(self.QDMObj.odmr.data[0, 1, idx])
        self.high_pos_pixel_line.set_ydata(self.QDMObj.odmr.data[1, 0, idx])
        self.high_neg_pixel_line.set_ydata(self.QDMObj.odmr.data[1, 1, idx])

        self.canvas.draw()

    def update_plots(self):
        """
        Update the plot with the current index.
        """
        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()

        for p in np.arange(self.QDMObj.odmr.n_pol):
            for f in np.arange(self.QDMObj.odmr.n_frange):
                self._uncorrected_lines[p][f].set_ydata(uncorrected[p, f])
                self._corrected_lines[p][f].set_ydata(corrected[p, f])
                if self.QDMObj.odmr._gf_factor != 0:
                    self._pixel_lines[p][f].set_ydata(pixel_spectra[p,f])

        for a in self.pixel_axes:
            a.set_ylim(mn, mx)

        self.canvas.draw()

    def update_marker(self):
        """
        Update the marker position on the image plots.
        """
        self.LOG.debug(f'Updating marker to ({self._current_xy[0] * self.QDMObj.bin_factor},{self._current_xy[1] * self.QDMObj.bin_factor})')

        for i, a in enumerate(self.img_axes):
            self._marker_line[i].set_data(self._current_xy[0] * self.QDMObj.bin_factor,
                                          self._current_xy[1] * self.QDMObj.bin_factor)
        self.canvas.draw()

    def on_frange_selector_changed(self, frange):
        self.LOG.info(f'frange changed to {frange}')

    def onSliderValueChanged(self):
        self.gf_label.setText(f'Global Fluorescence: {self.gfSlider.value() / 100:.2f}')
        self.update_plots()

    def onXYValueChange(self):
        self._current_idx = self.QDMObj.odmr.rc2idx([int(self.yselect.value()),
                                                     int(self.xselect.value())])
        self.LOG.debug(f'XY value changed to {self._current_xy} ({self._current_idx})')
        self.indexLabel.setText(f'({self._current_idx})')
        self.update_plots()
        self.update_marker()

    def apply_global_factor(self):
        self.LOG.debug(f'applying global factor {self.gfSlider.value() / 100:.2f}')
        self.QDMObj.odmr.correct_glob_fluorescence(self.gfSlider.value()/100)
        GFAppliedWindow(self.gfSlider.value() / 100)

        self.main_window.gf_select.setValue(self.gfSlider.value() / 100)
        self.close()

# class MplCanvas(FigureCanvas):
#     LOG = logging.getLogger('pyqdm.app')
#
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111)
#         super(MplCanvas, self).__init__(self.fig)
#         self.LOG.debug("MplCanvas created")
#
#
# class FluorescenceWindow(QWidget):
#     """
#     This "window" is a QWidget. If it has no parent,
#     it will appear as a free-floating window.
#     """
#     LOG = logging.getLogger('pyqdm.app')
#
#     def __init__(self, QDMObj):
#         self.QDMObj = QDMObj
#         print(self.QDMObj.odmr.f_Hz)
#         super().__init__()
#         layout = QVBoxLayout()
#         self.setWindowTitle("Fluorescence Plots")
#
#         self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
#         self.LOG.debug("FluorescenceWindow created")
#
#         self.canvas.axes.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
#         self.setLayout(layout)
#         self.resize(920, 600)
#         self.canvas.draw()
