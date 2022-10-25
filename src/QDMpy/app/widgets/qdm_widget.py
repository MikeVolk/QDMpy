import logging

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backend_bases import MouseButton, Event
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbarQT
from numpy.typing import NDArray

from QDMpy._core import models
from QDMpy.app.assets.GuiElements import LabeledDoubleSpinBox
from QDMpy.app.models import Pix
from QDMpy.utils import rms

MUT = "µT"

B111 = "B$_{111}$"


class NavigationToolbar(NavigationToolbarQT):
    def home(self, *args, **kwargs):
        s = "home_event"
        event = Event(s, self)
        event.foo = 100
        self.canvas.callbacks.process(s, event)
        super().home(*args, **kwargs)


class QDMWidget(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    LOG = logging.getLogger(__name__)

    POL = ["+", "-"]
    RANGE = ["<", ">"]

    @property
    def needs_marker_update(self):
        return len(self._is_img) > 0

    @property
    def needs_odmr_update(self):
        return len(self._is_spectra) > 0

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

    def __init__(
        self,
        canvas,
        clim_select=True,
        pixel_select=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.main_toolbar = QToolBar("Toolbar")
        self.main_toolbar.setStyleSheet("QToolBar{spacing:0px;padding:0px;}")
        self.addToolBar(self.main_toolbar)
        self._add_plt_toolbar()
        self.mainVerticalLayout = QVBoxLayout()
        self.toolbarLayout = QHBoxLayout()
        self.main_toolbar.addSeparator()
        if clim_select:
            self._add_cLim_select(self.main_toolbar)
            self.main_toolbar.addSeparator()

        if pixel_select:
            self._add_pixel_select(self.main_toolbar)

        self.mainVerticalLayout.addWidget(self.canvas)
        plt.tight_layout()

    # canvas wrapper methods
    def add_light(self):
        self.canvas.add_light(self.qdm.light, self.qdm.data_shape)

    def add_laser(self):
        self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)

    def update_data(self):
        """
        Update the data in the data canvases.
        """
        if not hasattr(self, "b111_select"):
            self.LOG.error("No b111_select QCombobox found")
            return

        # get a copy of the data
        d = self.qdm.b111[self.b111_select.currentIndex()].copy()

        # apply the men / quad background subtraction
        if hasattr(self, "subtract_median") and self.subtract_median.isChecked():
            self.LOG.debug("Subtracting median")
            d -= np.median(d)

        if hasattr(self, "subtract_quad") and self.subtract_quad.isChecked():
            self.LOG.debug("Subtracting Quad")
            d -= self.quad_background[self.b111_select.currentIndex()]

        self.canvas.add_data(d, self.qdm.data_shape)

        # set the colorbar label
        for ax in self.canvas.data:
            cax = self.canvas.data[ax]["cax"]
            cax.set_ylabel(f"{B111}({self.b111_select.currentText()[:3]}.)")
        self.update_clims()

    def toggle_outlier(self, visible):
        self.canvas.toggle_outlier(visible)

    def update_outlier(self):
        self.canvas.update_outlier(self.qdm.outliers.reshape(*self.qdm.data_shape))

    def add_scalebars(self):
        self.canvas.add_scalebars(self.qdm.pixel_size)

    def add_mean_odmr(self):
        self.canvas.add_mean_odmr(self.qdm.odmr.f_ghz, self.qdm.odmr.mean_odmr)

    def add_odmr(self, mean=False):
        self.canvas.update_odmr(
            self.qdm.odmr.f_ghz,
            self.get_current_odmr(),
            mean=self.qdm.odmr.mean_odmr if mean else None,
        )

    def update_extent(self):
        self.canvas.update_extent(self.qdm.data_shape)
        self.canvas.draw_idle()

    def set_main_window(self):
        """
        Sets the final widget to the layout.
        This is separate so you can add additional things to the toplayout ....
        """
        central_widget = QWidget()
        central_widget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(central_widget)

    def add_bottom_infobar(self):
        bottom_info_layout = QHBoxLayout()
        xy_txt_label = QLabel("dimensions: ")
        xy_label = QLabel()
        xy_unit = QLabel("µm")
        mean_txt_label = QLabel("mean: ")
        mean_label = QLabel("nan")
        mean_unit = QLabel(MUT)
        min_txt_label = QLabel("min: ")
        min_label = QLabel("nan")
        min_unit = QLabel(MUT)
        max_txt_label = QLabel("max: ")
        max_label = QLabel("nan")
        max_unit = QLabel(MUT)
        rms_txt_label = QLabel("RMS: ")
        rms_label = QLabel("nan")
        rms_unit = QLabel(MUT)
        filler = QLabel(" " * 300)
        bottom_info_layout.addWidget(xy_txt_label)
        bottom_info_layout.addWidget(xy_label)
        bottom_info_layout.addWidget(xy_unit)
        bottom_info_layout.addWidget(mean_txt_label)
        bottom_info_layout.addWidget(mean_label)
        bottom_info_layout.addWidget(mean_unit)
        bottom_info_layout.addWidget(min_txt_label)
        bottom_info_layout.addWidget(min_label)
        bottom_info_layout.addWidget(min_unit)
        bottom_info_layout.addWidget(max_txt_label)
        bottom_info_layout.addWidget(max_label)
        bottom_info_layout.addWidget(max_unit)
        bottom_info_layout.addWidget(rms_txt_label)
        bottom_info_layout.addWidget(rms_label)
        bottom_info_layout.addWidget(rms_unit)
        bottom_info_layout.addWidget(filler)
        self.mainVerticalLayout.addLayout(bottom_info_layout)

        self.infobar_labels = {
            "min": min_label,
            "max": max_label,
            "mean": mean_label,
            "rms": rms_label,
            "dimensions": xy_label,
        }
        self.fill_infobar(self.qdm.b111[self.b111_select.currentIndex()], 0, self.data_shape[1], 0, self.data_shape[0])

    def update_bottom_info(self):
        x0, x1 = np.sort([self._xy_box[0][0], self._xy_box[1][0]]).astype(int)
        y0, y1 = np.sort([self._xy_box[0][1], self._xy_box[1][1]]).astype(int)

        if hasattr(self, "infobar_labels") and hasattr(self, "b111_select"):
            d = self.qdm.b111[self.b111_select.currentIndex()][y0:y1, x0:x1]
            self.fill_infobar(d, x0, x1, y0, y1)

    def init_info_bar(self, *args):
        """
        Initialize the info bar with the data from the whole b111 image.

        Args:
            *args: not used
        """
        self.fill_infobar(self.qdm.b111[self.b111_select.currentIndex()], 0, self.data_shape[1], 0, self.data_shape[0])

    def fill_infobar(self, d: NDArray, x0: int, x1: int, y0: int, y1: int) -> None:
        """Fill the infobar with the statistical data in d.

        Args:
            d: data to fill the infobar with
            x0: x0 coordinate
            x1: x1 coordinate
            y0: y0 coordinate
            y1: y1 coordinate
        """
        x_size = (x1 - x0) * self.pixel_size * 1e6
        y_size = (y1 - y0) * self.pixel_size * 1e6
        self.infobar_labels["dimensions"].setText(f"{x_size:.2f} x {y_size:.2f}")
        self.infobar_labels["min"].setText(f"{np.nanmin(d):.2f}")
        self.infobar_labels["max"].setText(f"{np.nanmax(d):.2f}")
        self.infobar_labels["mean"].setText(f"{np.nanmean(d):.2f}")
        self.infobar_labels["rms"].setText(f"{rms(d):.2f}")

    def _add_cLim_select(self, toolbar):
        clim_widget = QWidget()
        clim_selection_layout = QHBoxLayout()
        clim_label, self.clims_selector = LabeledDoubleSpinBox(
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
        self.canvas.mpl_connect("home_event", self.init_info_bar)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMinimumWidth(380)
        self.toolbar.addSeparator()
        self.addToolBar(self.toolbar)

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

    def _add_pixel_select(self, toolbar):
        pixel_box_widget = QWidget()
        coord_box = QHBoxLayout()
        self.xlabel, self.xselect = LabeledDoubleSpinBox(
            "x",
            int(self._current_xy[0]),
            0,
            1,
            0,
            self.qdm.odmr.data_shape[1],
            self.on_xy_value_change,
        )
        self.ylabel, self.yselect = LabeledDoubleSpinBox(
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

    # def toggle_outlier_mask(self, onoff="on"):
    #     for axdict in [self.data, self.laser, self.fluorescence]:
    #         for ax in axdict:
    #             img = axdict[ax]['outlier']
    #             if img is None:
    #                 self.add_outlier()
    #                 img = self._outlier_masks[ax]
    #             else:
    #                 img.set_visible(img.)
    #     self.canvas.draw_idle()

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
            self.update_bottom_info()

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
            x, y = int(event.xdata), int(event.ydata)

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

        self.caller.update_marker()
        self.caller.update_odmr()

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
        model_func = self.qdm.fit.model_func
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
            current_correct = self.qdm.odmr.calc_gf_correction(gf=self.qdm.odmr.global_factor)
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
            new_correct = self.qdm.odmr.calc_gf_correction(gf=slider_value / 100)
            corrected = uncorrected_odmr - new_correct
        else:
            corrected = self.get_current_odmr()

        return corrected

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
        self.set_ylim()

    def set_ylim(self):
        self.canvas.update_odmr_lims(self.qdm.odmr.data)

    def update_clims(self):
        if hasattr(self, "fix_clim_check_box"):
            use_percentile = self.fix_clim_check_box.isChecked()
            percentile = self.clims_selector.value()
        else:
            use_percentile = False
            percentile = 100

        self.canvas.update_clims(
            use_percentile=use_percentile,
            percentile=percentile,
        )
        self.canvas.draw_idle()

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
