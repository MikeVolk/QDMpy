import logging

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pyqdm.app.models import Pix
from pyqdm.app.widgets.tools import get_label_box
from pyqdm.core import models


class PyQdmWindow(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    POL = ["+", "-"]
    RANGE = ["<", ">"]

    @property
    def _is_img(self):
        return self.canvas.img_axes

    @property
    def _is_data(self):
        return self.canvas.data_axes

    @property
    def _is_spectra(self):
        return self.canvas.odmr_axes

    @property
    def qdm(self):
        return self.parent().qdm

    # canvas wrapper methods
    def add_light(self):
        self.canvas.add_light(self.qdm.light, self.qdm.data_shape)

    def add_laser(self):
        self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)

    def update_data(self):
        # empty, needs to be implemented in the child
        pass

    def add_scalebars(self):
        self.canvas.add_scalebars(self.qdm.pixel_size)

    def add_mean_odmr(self):
        self.canvas.add_mean_odmr(self.qdm.odmr.f_ghz, self.qdm.odmr.mean_odmr)

    def add_odmr(self, mean=False):
        self.canvas.update_odmr(
            self.qdm.odmr.f_ghz, self.get_current_odmr(), mean=self.qdm.odmr.mean_odmr if mean else None
        )

    def __init__(self, canvas, includes_fits=False, clim_select=True, pixel_select=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LOG = logging.getLogger(f"pyqdm.{self.__class__.__name__}")
        self.caller = self.parent()

        self.setContentsMargins(0, 0, 0, 0)
        self.canvas = canvas
        self._xy_box = [[0, 0], [0, 0]]

        self.pix = Pix()
        self.data_shape = self.qdm.data_shape
        self.img_shape = self.qdm.light.shape

        self.data_ax_img = {}
        self.led_ax_img = {}
        self.laser_ax_img = {}

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # layout
        self.mainToolbar = QToolBar("Toolbar")
        self.mainToolbar.setStyleSheet("QToolBar{spacing:0px;padding:0px;}")
        self._add_plt_toolbar()
        self.addToolBar(self.mainToolbar)
        self.mainVerticalLayout = QVBoxLayout()
        self.toolbarLayout = QHBoxLayout()
        self.mainToolbar.addSeparator()

        if clim_select:
            self._add_cLim_select(self.mainToolbar)
            self.mainToolbar.addSeparator()

        if pixel_select:
            self._add_pixel_select(self.mainToolbar)

        self.mainVerticalLayout.addWidget(self.canvas)

    def set_main_window(self):
        """
        Sets the final widget to the layout.
        THis is separate so you can add additional things to the toplayout ....
        """
        central_widget = QWidget()
        central_widget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(central_widget)

    def add_bottom_infobar(self):
        bottom_info_layout = QHBoxLayout()
        xy_txt_label = QLabel("dimensions: ")
        xy_label = QLabel()
        xy_unit = QLabel("[mum]")
        mean_txt_label = QLabel("mean: ")
        mean_label = QLabel("nan")
        min_txt_label = QLabel("min: ")
        min_label = QLabel("nan")
        max_txt_label = QLabel("max: ")
        max_label = QLabel("nan")
        rms_txt_label = QLabel("RMS: ")
        rms_label = QLabel("nan")

        bottom_info_layout.addWidget(xy_txt_label)
        bottom_info_layout.addWidget(xy_label)
        bottom_info_layout.addWidget(xy_unit)
        bottom_info_layout.addWidget(mean_txt_label)
        bottom_info_layout.addWidget(mean_label)
        bottom_info_layout.addWidget(min_txt_label)
        bottom_info_layout.addWidget(min_label)
        bottom_info_layout.addWidget(max_txt_label)
        bottom_info_layout.addWidget(max_label)
        bottom_info_layout.addWidget(rms_txt_label)
        bottom_info_layout.addWidget(rms_label)
        self.mainVerticalLayout.addLayout(bottom_info_layout)

        self.infobar_labels = {
            "min": min_label,
            "max": max_label,
            "mean": mean_label,
            "rms": rms_label,
            "dimensions": xy_label,
        }

    def update_bottom_info(self):
        np.sort(self._xy_box[0][0], self._xy_box[1][0])
        np.sort(self._xy_box[0][1], self._xy_box[1][1])

        self.data_img

    def _add_cLim_select(self, toolbar):
        clim_widget = QWidget()
        clim_selection_layout = QHBoxLayout()
        clim_label, self.clims_selector = get_label_box(
            label="clim",
            value=99.5,
            decimals=1,
            step=1,
            vmin=1,
            vmax=100,
            callback=self.update_clims,
        )
        clim_label_unit = QLabel("[%]")
        self.fix_clim_check_box = QCheckBox("set ")
        self.fix_clim_check_box.setStatusTip("Fix the color scale")
        self.fix_clim_check_box.stateChanged.connect(self.update_clims)
        clim_selection_layout.addWidget(clim_label)
        clim_selection_layout.addWidget(self.clims_selector)
        clim_selection_layout.addWidget(clim_label_unit)
        clim_selection_layout.addWidget(self.fix_clim_check_box)
        clim_widget.setLayout(clim_selection_layout)
        toolbar.addWidget(clim_widget)

    def _add_plt_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMinimumWidth(370)
        self.toolbar.addSeparator()
        self.addToolBar(self.toolbar)

    def _add_pixel_select(self, toolbar):
        pixel_box_widget = QWidget()
        coord_box = QHBoxLayout()
        self.xlabel, self.xselect = get_label_box(
            "x",
            int(self._current_xy[0]),
            0,
            1,
            0,
            self.qdm.odmr.data_shape[1],
            self.on_xy_value_change,
        )
        self.ylabel, self.yselect = get_label_box(
            "y",
            int(self._current_xy[1]),
            0,
            1,
            0,
            self.qdm.odmr.data_shape[0],
            self.on_xy_value_change,
        )
        self.xselect.valueChanged.disconnect(self.on_xy_value_change)
        self.xselect.setValue(int(self._current_xy[0]))
        self.xselect.valueChanged.connect(self.on_xy_value_change)
        self.yselect.valueChanged.disconnect(self.on_xy_value_change)
        self.yselect.setValue(int(self._current_xy[1]))
        self.yselect.valueChanged.connect(self.on_xy_value_change)
        self.indexLabel = QLabel(f"[{self._current_idx}]")
        self.indexLabel.setFixedWidth(60)
        coord_box.addWidget(self.xlabel)
        coord_box.addWidget(self.xselect)
        coord_box.addWidget(self.ylabel)
        coord_box.addWidget(self.yselect)
        coord_box.addWidget(self.indexLabel)
        pixel_box_widget.setLayout(coord_box)
        toolbar.addWidget(pixel_box_widget)

    def add_outlier_mask(self):
        for ax, img in self._outlier_masks.items():
            self.LOG.debug(f"Adding outlier mask to axis {ax}")
            if img is None:
                self._outlier_masks[ax] = ax.imshow(
                    self.qdm.outliers.reshape(self.qdm.data_shape),
                    cmap="gist_rainbow",
                    alpha=self.qdm.outliers.reshape(self.qdm.data_shape).astype(float),
                    vmin=0,
                    vmax=1,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    zorder=2,
                )
            else:
                img.set_data(self.qdm.outliers.reshape(self.qdm.data_shape))
        self.canvas.draw()

    def toggle_outlier_mask(self, onoff="on"):
        for ax, img in self._outlier_masks.items():
            if onoff == "on":
                if img is None:
                    self.add_outlier_mask()
                    img = self._outlier_masks[ax]
                img.set_visible(True)
            if onoff == "off":
                img.set_visible(False)
        self.canvas.draw()

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

    def _click_in_axes(self, event):
        if event.xdata is None or event.ydata is None:
            self.LOG.debug("clicked outside of axes")
            return False
        return True

    def on_release(self, event):
        if event.inaxes in self._is_spectra:
            self.LOG.debug(f"clicked in {event.inaxes}")
            return
        if not self._click_in_axes(event):
            return

        if self.toolbar.mode:
            self._xy_box[1] = [event.xdata, event.ydata]
            self.LOG.debug(f"zoom box: at {self._xy_box}")

    def on_press(self, event):
        if event.inaxes in self._is_spectra:
            self.LOG.debug(f"clicked in {event.inaxes}")
            return

        if not self._click_in_axes(event):
            return

        if self.toolbar.mode:
            self._xy_box = [[event.xdata, event.ydata], []]

        if event.button == MouseButton.LEFT and not self.toolbar.mode:
            self.qdm.bin_factor
            # event is in image coordinates
            xy = [event.xdata, event.ydata]
            x, y = np.round(xy).astype(int)

            self.xselect.valueChanged.disconnect(self.on_xy_value_change)
            self.xselect.setValue(x)
            self.xselect.valueChanged.connect(self.on_xy_value_change)

            self.yselect.valueChanged.disconnect(self.on_xy_value_change)
            self.yselect.setValue(y)
            self.yselect.valueChanged.connect(self.on_xy_value_change)

            self.set_current_idx(x, y)
            self.indexLabel.setText(f"[{self._current_idx}]")
            self.LOG.debug(f"clicked in {event.inaxes} with new index: {self._current_idx}")

            self.on_xy_value_change()

    def on_xy_value_change(self):
        """
        Changes the current ODMR spectra on all odmr axes and the marker on all image axes
        for the current pixel.
        """
        self.set_current_idx(x=self.xselect.value(), y=self.yselect.value())
        self.LOG.debug(f"XY value changed to {self._current_xy} ({self._current_idx})")
        self.indexLabel.setText(f"[{self._current_idx}]")

        if self.canvas.has_img:
            self.caller.update_marker()
        if self.canvas.has_odmr:
            self.caller.update_odmr()
        self.canvas.draw()

    def get_current_odmr(self):
        """
        Returns the current odmr spectra for all polarities and franges.
        The spectra may or may not be corrected for the global fluorescence.

        Returns
        -------
        np.array of shape (n_polarities, n_franges, n_freqs)
            ODMR spectrum of current_idx
        """
        return self.qdm.odmr.data[:, :, self._current_idx]

    def get_current_fit(self):
        parameter = self.qdm.fit.parameter[:, :, self._current_idx]
        model_func = self.qdm.fit.model[0]
        freqs = np.empty((parameter.shape[1], 200))
        models = np.empty((parameter.shape[0], parameter.shape[1], 200))
        for f in np.arange(parameter.shape[1]):
            freqs[f] = np.linspace(self.qdm.odmr.f_ghz[f].min(), self.qdm.odmr.f_ghz[f].max(), 200)
            for p in np.arange(parameter.shape[0]):
                models[p, f, :] = model_func(freqs[f], parameter[p, f])
        return models

    def get_uncorrected_odmr(self):
        """
        Returns the uncorrected odmr spectra for all polarities and franges.

        Returns
        -------
        np.array of shape (n_polarities, n_franges, n_freqs)
            uncorrected ODMR spectrum of current_idx
        """
        # get current pixel data, may be corrected
        current_data = self.qdm.odmr.data[:, :, self._current_idx].copy()

        # get current correction
        if self.qdm.odmr.global_factor > 0:
            current_correct = self.qdm.odmr.get_gf_correction(gf=self.qdm.odmr.global_factor)
            # make uncorrected
            current_data += current_correct
        return current_data

    def get_corrected_odmr(self, slider_value=None):
        """
        Returns the corrected odmr spectra for all polarities and franges.

        Parameters
        ----------
        slider_value : int,float, optional
            Slider value from gf_window, by default None

        Returns
        -------
        np.array of shape (n_polarities, n_franges, n_freqs)
            corrected ODMR spectrum of current_idx
        """
        uncorrected_odmr = self.get_uncorrected_odmr()

        if slider_value is not None:
            new_correct = self.qdm.odmr.get_gf_correction(gf=slider_value / 100)
            corrected = uncorrected_odmr - new_correct
        else:
            corrected = self.get_current_odmr()
            # corrected = np.empty(uncorrected_odmr.shape)
            # corrected[:, :] = np.nan

        return corrected

    def get_pixel_data(self, slider_value=None):
        """GEt the GF-corrected data for the current pixel

        Parameters
        ----------
        slider_value : int, optional
            the slider value of the gf, by default None

        Returns
        -------
        _type_
            _description_
        """

        # get current pixel data, may be corrected
        current_data = self.qdm.odmr.data[:, :, self._current_idx].copy()

        # get current correction
        current_correct = self.qdm.odmr.get_gf_correction(gf=self.qdm.odmr.global_factor)
        # make uncorrected
        uncorrected = current_data + current_correct

        if slider_value is not None:
            new_correct = self.qdm.odmr.get_gf_correction(gf=slider_value / 100)
            corrected = uncorrected - new_correct
        else:
            corrected = np.empty(current_correct.shape)
            corrected[:, :] = np.nan

        return current_data, uncorrected, corrected

    def set_current_idx(self, x=None, y=None, idx=None):
        self.caller.set_current_idx(x=x, y=y, idx=idx)

    def update_marker(self):
        """
        Update the marker position on the image plots.
        """
        x, y = self._current_xy
        self.canvas.update_marker(x, y)

    def update_odmr(self):
        """
        Update the marker position on the image plots.
        """
        self.canvas.update_odmr(freq=self.qdm.odmr.f_ghz, data=self.get_corrected_odmr())
        self.canvas.update_odmr_lims()

    def update_clims(self):
        self.canvas.update_clims(
            use_percentile=self.fix_clim_check_box.isChecked(),
            percentile=self.clims_selector.value(),
        )
        self.canvas.draw()

    def redraw_all_plots(self):
        self.update_data()
        self.update_odmr()
        self.update_marker()

    @property
    def model(self):
        return [None, models.esrsingle, models.esr15n, models.esr14n][self.qdm._diamond_type]

    @property
    def pixel_size(self):
        return self.qdm.pixel_size
