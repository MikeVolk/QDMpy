import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar
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

from pyqdm.app.windows.tools import get_label_box
from pyqdm.core import models
from pyqdm.app.models import Pix

matplotlib.rcParams.update(
    {  # 'font.size': 8,
        # 'axes.labelsize': 8,
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    }
)


class PyQdmWindow(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    _pixel_marker = []  # Line2D objects that contain the pixel marker
    _pixel_lines = [[None, None], [None, None]]  # Line2D for selected pixel
    _fit_lines = [[None, None], [None, None]]  # Line2D for fit line

    _need_cLim_update = []

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

    # @property
    # def qdm(self):
    #     return self.caller.qdm

    def __init__(self, canvas, qdm_instance, includes_fits=False, *args, **kwargs):
        self.LOG = logging.getLogger(f"pyqdm.{self.__class__.__name__}")
        self.qdm = qdm_instance
        self._includes_fits = includes_fits
        self.pix = Pix()
        self.data_shape = self.qdm.data_shape
        self.img_shape = self.qdm.light.shape

        super().__init__(*args, **kwargs)
        self.caller = self.parent()

        self.setContentsMargins(0, 0, 0, 0)
        self.canvas = canvas
        self._xy_box = [[0, 0], [0, 0]]

        self.data_ax_img = {}
        self.led_ax_img = {}
        self.laser_ax_img = {}

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)

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

    def set_main_window(self):
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

    def _add_cLim_selector(self, toolbar):
        clim_widget = QWidget()
        clim_selection_layout = QHBoxLayout()
        clim_label, self.clims_selector = get_label_box(
            label="clim",
            value=99,
            decimals=1,
            step=1,
            vmin=1,
            vmax=100,
            callback=self.update_img_plots,
        )
        clim_label_unit = QLabel("[%]")
        self.fix_clim_check_box = QCheckBox("set ")
        self.fix_clim_check_box.setStatusTip("Fix the color scale")
        self.fix_clim_check_box.stateChanged.connect(self.update_img_plots)
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

    def _add_pixel_box(self, toolbar):
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

            self.caller.update_marker()
            self.caller.update_odmr()

    def on_xy_value_change(self):
        self.set_current_idx(x=self.xselect.value(), y=self.yselect.value())
        self.LOG.debug(f"XY value changed to {self._current_xy} ({self._current_idx})")
        self.indexLabel.setText(f"[{self._current_idx}]")

        self.caller.update_marker()
        self.caller.update_odmr()

    def get_pixel_data(self):
        gf_factor = self.gfSlider.value() / 100
        new_correct = self.qdm.odmr.get_gf_correction(gf=gf_factor)
        old_correct = self.qdm.odmr.get_gf_correction(gf=self.qdm.odmr.global_factor)
        pixel_spectra = self.qdm.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        uncorrected = pixel_spectra + old_correct
        corrected = uncorrected - new_correct
        mn = np.min([np.min(pixel_spectra), np.min(corrected), np.min(uncorrected)]) * 0.998
        mx = np.max([np.max(pixel_spectra), np.max(corrected), np.max(uncorrected)]) * 1.002
        return pixel_spectra, uncorrected, corrected, mn, mx

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
        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()
        self.canvas.update_odmr(data=pixel_spectra, uncorrected=uncorrected, corrected=corrected)
        self.canvas.update_odmr_lims()
        self.canvas.draw()

    def need_extend(self):
        return self.fix_clim_check_box.isChecked() and self.clims_selector.value() != 100

    def update_img_plots(self):
        """
        needs to be implemented by classes that inherit from pyqdm_window
        """
        pass

    def redraw_all_plots(self):
        self.update_img_plots()
        self.update_marker()

    @property
    def model(self):
        return [None, models.esrsingle, models.esr15n, models.esr14n][self.qdm._diamond_type]

    @property
    def pixel_size(self):
        return self.qdm.pixel_size
