import logging

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QDoubleSpinBox, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSizePolicy, QToolBar,
)
from PySide6.QtCore import Qt, QSize
import models
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton

matplotlib.rcParams.update({  # 'font.size': 8,
    # 'axes.labelsize': 8,
    'grid.linestyle': '-',
    'grid.alpha': 0.5})


class pyqdmWindow(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """
    _pixel_marker = []  # Line2D objects that contain the pixel marker
    _pixel_lines = [[None, None], [None, None]]  # Line2D for selected pixel
    _fit_lines = [[None, None], [None, None]]  # Line2D for fit line

    _need_cLim_update = []

    POL = ['+', '-']
    RANGE = ['<', '>']

    @property
    def _is_img(self):
        return self.canvas._is_img

    @property
    def _is_data(self):
        return self.canvas._is_data

    @property
    def _is_spectra(self):
        return self.canvas._is_spectra

    def __init__(self, caller, canvas, QDMObj=None, includes_fits=False, *args, **kwargs):
        self.LOG = logging.getLogger('pyqdm.' + self.__class__.__name__)
        self.caller = caller
        self.QDMObj = QDMObj
        self._includes_fits = includes_fits
        super().__init__(*args, **kwargs)

        self.setContentsMargins(0, 0, 0, 0)
        # Create the maptlotlib FigureCanvas object,
        self.canvas = canvas
        cid = self.canvas.mpl_connect('button_press_event', self.on_press)

        self.mainToolbar = QToolBar("Toolbar")
        self.mainToolbar.setStyleSheet("QToolBar{spacing:0px;padding:0px;}")

        self._add_plt_toolbar()
        self.addToolBar(self.mainToolbar)
        self._outlier_masks = {}

        self.mainVerticalLayout = QVBoxLayout()
        self.toolbarLayout = QHBoxLayout()

        self.mainToolbar.addSeparator()
        self._add_cLim_selector(self.mainToolbar)
        self.mainToolbar.addSeparator()
        self._add_pixel_box(self.mainToolbar)

        self.mainVerticalLayout.addWidget(self.canvas)

        centralWidget = QWidget()
        centralWidget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(centralWidget)

    def _add_cLim_selector(self, toolbar):
        cLimWidget = QWidget()
        cLimSelectLayout = QHBoxLayout()
        cLimLabel, self.cLimSelector = self.get_label_box(label='clim', value=99,
                                                          decimals=1, step=1, min=1, max=100,
                                                          callback=self.update_img_plots)
        cLimLabelUnit = QLabel('[%]')
        self.fixClimCheckBox = QCheckBox(f'set')
        self.fixClimCheckBox.setStatusTip('Fix the color scale')
        self.fixClimCheckBox.stateChanged.connect(self.update_img_plots)
        cLimSelectLayout.addWidget(cLimLabel)
        cLimSelectLayout.addWidget(self.cLimSelector)
        cLimSelectLayout.addWidget(cLimLabelUnit)
        cLimSelectLayout.addWidget(self.fixClimCheckBox)
        cLimWidget.setLayout(cLimSelectLayout)
        toolbar.addWidget(cLimWidget)

    def _add_plt_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMinimumWidth(370)
        self.toolbar.addSeparator()
        self.addToolBar(self.toolbar)

    def _add_pixel_box(self, toolbar):
        pixelBoxWidget = QWidget()
        coordBox = QHBoxLayout()
        self.xlabel, self.xselect = self.get_label_box('x', int(self._current_xy[0]), 0, 1, 0,
                                                       self.QDMObj.odmr.scan_dimensions[1],
                                                       self.onXYValueChange)
        self.ylabel, self.yselect = self.get_label_box('y', int(self._current_xy[1]), 0, 1, 0,
                                                       self.QDMObj.odmr.scan_dimensions[0],
                                                       self.onXYValueChange)
        self.xselect.valueChanged.disconnect(self.onXYValueChange)
        self.xselect.setValue(int(self._current_xy[0]))
        self.xselect.valueChanged.connect(self.onXYValueChange)
        self.yselect.valueChanged.disconnect(self.onXYValueChange)
        self.yselect.setValue(int(self._current_xy[1]))
        self.yselect.valueChanged.connect(self.onXYValueChange)
        self.indexLabel = QLabel(f'[{self._current_idx}]')
        self.indexLabel.setFixedWidth(60)
        coordBox.addWidget(self.xlabel)
        coordBox.addWidget(self.xselect)
        coordBox.addWidget(self.ylabel)
        coordBox.addWidget(self.yselect)
        coordBox.addWidget(self.indexLabel)
        pixelBoxWidget.setLayout(coordBox)
        toolbar.addWidget(pixelBoxWidget)

    def _add_outlier_mask(self):
        for ax, img in self._outlier_masks.items():
            self.LOG.debug('Adding outlier mask to axis {}'.format(ax))
            if img is None:
                self._outlier_masks[ax] = ax.imshow(
                    self.QDMObj.outliers.reshape(self.QDMObj.scan_dimensions),
                    cmap='gist_rainbow',
                    alpha=self.QDMObj.outliers.reshape(self.QDMObj.scan_dimensions).astype(float),
                    vmin=0, vmax=1, interpolation='none', origin='lower', aspect='equal',
                    zorder=2)
            else:
                img.set_data(self.QDMObj.outliers.reshape(self.QDMObj.scan_dimensions))
        self.canvas.draw()

    def _toggle_outlier_mask(self, onoff = 'on'):
        for ax, img in self._outlier_masks.items():
            if onoff == 'on':
                if img is None:
                    self._add_outlier_mask()
                    img = self._outlier_masks[ax]
                img.set_visible(True)
            if onoff == 'off':
                img.set_visible(False)
        self.canvas.draw()

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

    def set_current_idx(self, x=None, y=None, idx=None):
        self.caller.set_current_idx(x=x, y=y, idx=idx)

    def add_laser_img(self, ax, cax=None):
        im = ax.imshow(self.QDMObj.laser, cmap='magma', interpolation='none', origin='lower', aspect='equal',
                       extent=[0, self.QDMObj.odmr.scan_dimensions[1], 0, self.QDMObj.odmr.scan_dimensions[0]])
        ax.set(xlabel='px', ylabel='px', title='Laser',
               )
        if cax is not None:
            self.cbar = plt.colorbar(im, cax=cax)

    def add_light_img(self, ax):
        ax.imshow(self.QDMObj.led, cmap='bone', interpolation='none', origin='lower', aspect='equal',
                  extent=[0, self.QDMObj.odmr.scan_dimensions[1], 0, self.QDMObj.odmr.scan_dimensions[0]])
        ax.set(xlabel='px', ylabel='px', title='Light',
               xlim=(0, self.QDMObj.odmr.scan_dimensions[1]), ylim=(0, self.QDMObj.odmr.scan_dimensions[0]))

    @property
    def _current_xy(self):
        """
        Returns the current xy-coordinates in data coordinates.
        :return:
        """
        return self.caller._current_xy

    @property
    def _current_idx(self):
        return self.caller._current_idx

    def on_press(self, event):
        if event.inaxes in self._is_spectra:
            self.LOG.debug(f'clicked in {event.inaxes}')
            return

        if event.xdata is None or event.ydata is None:
            self.LOG.debug(f'clicked outside of axes')
            return

        if event.button == MouseButton.LEFT and not self.toolbar.mode:
            bin_factor = self.QDMObj.bin_factor
            # event is in image coordinates
            xy = [event.xdata, event.ydata]
            x, y = np.round(xy).astype(int)

            self.xselect.valueChanged.disconnect(self.onXYValueChange)
            self.xselect.setValue(x)
            self.xselect.valueChanged.connect(self.onXYValueChange)

            self.yselect.valueChanged.disconnect(self.onXYValueChange)
            self.yselect.setValue(y)
            self.yselect.valueChanged.connect(self.onXYValueChange)

            self.set_current_idx(x, y)
            self.indexLabel.setText(f'[{self._current_idx}]')
            self.LOG.debug(f'clicked in {event.inaxes} with new index: {self._current_idx}')

            self.caller.update_marker()
            self.caller.update_pixel()
            self.update_fit_lines()

    def onXYValueChange(self):
        self.set_current_idx(x=self.xselect.value(), y=self.yselect.value())
        self.LOG.debug(f'XY value changed to {self._current_xy} ({self._current_idx})')
        self.indexLabel.setText(f'[{self._current_idx}]')

        self.caller.update_marker()
        self.caller.update_pixel()

    def set_current_idx(self, x=None, y=None, idx=None):
        self.caller.set_current_idx(x=x, y=y, idx=idx)

    def init_plots(self):
        """
        needs to be implemented by classes that inherit from pyqdm_window
        """
        pass

    def _add_scalebars(self):
        for ax in self.img_axes:
            # Create scale bar
            if ax in self._is_image:
                scalebar = ScaleBar(self.pixelsize, "m", length_fraction=0.25,
                                    frameon=True, box_alpha=0.5, location="lower left")
            else:
                scalebar = ScaleBar(self.pixelsize * self.QDMObj.bin_factor, "m", length_fraction=0.25,
                                    frameon=True, box_alpha=0.5, location="lower left")
            ax.add_artist(scalebar)

    def _init_lines(self):
        if self._is_spectra is None:
            return
        # init the lists for marker, pixel and fit lines
        self._pixel_marker = np.array([None for a in self._is_img])
        self._pixel_lines = np.array(
            [[None for f in np.arange(self.QDMObj.odmr.n_pol)] for p in np.arange(self.QDMObj.odmr.n_frange)])
        self._fit_lines = np.array(
            [[None for p in np.arange(self.QDMObj.odmr.n_pol)] for f in np.arange(self.QDMObj.odmr.n_frange)])

    def add_mean_ODMR(self):
        for f in np.arange(self.QDMObj.odmr.n_frange):
            for p in np.arange(self.QDMObj.odmr.n_pol):
                self._pixel_ax[p][f].plot(self.QDMObj.odmr.f_GHz[f], self.QDMObj.odmr.mean_odmr[p, f],
                                          marker='', ls='--', alpha=0.5, lw=1,
                                          label=f'mean ({self.POL[p]})')

    def update_pixel_lines(self):
        if len(self._pixel_ax) == 0:
            return

        pixel_spectra = self.QDMObj.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        for f in np.arange(self.QDMObj.odmr.n_frange):
            for p in np.arange(self.QDMObj.odmr.n_pol):
                if self._pixel_lines[p][f] is None:
                    self._pixel_lines[p][f], = self._pixel_ax[p][f].plot(self.QDMObj.odmr.f_GHz[f], pixel_spectra[p, f],
                                                                         marker='.', markersize=6, mfc='w',
                                                                         linestyle='' if self._includes_fits else '-')
                else:
                    self._pixel_lines[p][f].set_data(self.QDMObj.odmr.f_GHz[f], pixel_spectra[p, f])
                self._pixel_lines[p][f].set_label(f'p({self.POL[p]},{self._current_xy[0]},{self._current_xy[1]})')
                self._pixel_ax[p][f].legend(loc='lower left', fontsize=8, ncol=2)
        self.update_pixel_lims()
        self.canvas.draw()

    def update_fit_lines(self):

        if not self._includes_fits:
            return

        parameter = self.QDMObj.fitted_parameter[:, :, :, [self._current_idx]]
        parameter = np.rollaxis(parameter, axis=0, start=4)

        for f in np.arange(self.QDMObj.odmr.n_frange):
            f_new = np.linspace(min(self.QDMObj.ODMRobj.f_GHz[f]), max(self.QDMObj.ODMRobj.f_GHz[f]), 200)
            for p in np.arange(self.QDMObj.odmr.n_pol):
                m_fit = self.model(parameter=parameter[p, f], x=f_new)
                if self._fit_lines[p][f] is None:
                    self._fit_lines[p][f], = self._pixel_ax[p][f].plot(f_new, m_fit[0], marker='', linestyle='-',
                                                                       lw=0.8,
                                                                       color=self._pixel_lines[p][f].get_color())
                else:
                    self._fit_lines[p][f].set_ydata(m_fit[0])
                self._fit_lines[p][f].set_label(f'fit ({self.POL[p]},{self._current_xy[0]},{self._current_xy[1]})')
                self._pixel_ax[p][f].legend(loc='lower left', fontsize=8, ncol=3)

        self.update_pixel_lims()
        self.canvas.draw()

    def update_pixel_lims(self):
        self.LOG.debug(f'updating xy limits for the pixel plots')
        for ax in self._pixel_ax.flatten():
            mn = np.min([np.min(l.get_ydata()) for l in ax.get_lines()])
            mx = np.max([np.max(l.get_ydata()) for l in ax.get_lines()])
            ax.set(ylim=(mn * 0.999, mx * 1.001))

    def update_marker(self):
        """
        Update the marker position on the image plots.
        """
        self.LOG.debug(f'Updating marker to ({self._current_xy[0] * self.QDMObj.bin_factor},'
                       f'{self._current_xy[1] * self.QDMObj.bin_factor})')

        x, y = self._current_xy

        for i, a in enumerate(self._is_img):
            if self._pixel_marker[i] is None:
                self._pixel_marker[i], = a.plot(x, y, marker='x', color='chartreuse', zorder=100)
            else:
                self._pixel_marker[i].set_data(self._current_xy[0], self._current_xy[1])

        self.canvas.draw()

    def need_extend(self):
        return self.fixClimCheckBox.isChecked() and self.cLimSelector.value() != 100

    def update_img_plots(self):
        """
        needs to be implemented by classes that inherit from pyqdm_window
        """
        pass

    def update_pixel(self):
        self.caller.update_pixel()

    def redraw_all_plots(self):
        self.update_img_plots()
        self.update_marker()

    @property
    def model(self):
        return [None, models.ESRSINGLE, models.ESR15N, models.ESR14N][self.QDMObj._diamond_type]

    @property
    def pixel_size(self):
        return self.QDMObj.pixel_size
