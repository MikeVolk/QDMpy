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
        return self.canvas._is_spectra

    # @property
    # def qdm(self):
    #     return self.caller.qdm

    def __init__(self, caller, canvas, qdm_instance, includes_fits=False, *args, **kwargs):
        self.LOG = logging.getLogger(f"pyqdm.{self.__class__.__name__}")
        self.caller = caller
        self.qdm = qdm_instance
        self._includes_fits = includes_fits
        super().__init__(*args, **kwargs)
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
            self.qdm.odmr.scan_dimensions[1],
            self.on_xy_value_change,
        )
        self.ylabel, self.yselect = get_label_box(
            "y",
            int(self._current_xy[1]),
            0,
            1,
            0,
            self.qdm.odmr.scan_dimensions[0],
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
                    self.qdm.outliers.reshape(self.qdm.scan_dimensions),
                    cmap="gist_rainbow",
                    alpha=self.qdm.outliers.reshape(self.qdm.scan_dimensions).astype(float),
                    vmin=0,
                    vmax=1,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    zorder=2,
                )
            else:
                img.set_data(self.qdm.outliers.reshape(self.qdm.scan_dimensions))
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

    def set_current_idx(self, x=None, y=None, idx=None):
        self.caller.set_current_idx(x=x, y=y, idx=idx)

    def add_laser_img(self, ax, cax=None):
        im = ax.imshow(
            self.qdm.laser,
            cmap="magma",
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=[
                0,
                self.qdm.odmr.scan_dimensions[1],
                0,
                self.qdm.odmr.scan_dimensions[0],
            ],
        )
        ax.set(
            xlabel="px",
            ylabel="px",
            title="Laser",
        )
        if cax is not None:
            self.cbar = plt.colorbar(im, cax=cax)

    def add_light_img(self, ax):
        ax.imshow(
            self.qdm.light,
            cmap="bone",
            interpolation="none",
            origin="lower",
            aspect="equal",
            extent=[
                0,
                self.qdm.odmr.scan_dimensions[1],
                0,
                self.qdm.odmr.scan_dimensions[0],
            ],
        )
        ax.set(
            xlabel="px",
            ylabel="px",
            title="Light",
            xlim=(0, self.qdm.odmr.scan_dimensions[1]),
            ylim=(0, self.qdm.odmr.scan_dimensions[0]),
        )

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
            self.caller.update_pixel()
            self.update_fit_lines()

    def on_xy_value_change(self):
        self.set_current_idx(x=self.xselect.value(), y=self.yselect.value())
        self.LOG.debug(f"XY value changed to {self._current_xy} ({self._current_idx})")
        self.indexLabel.setText(f"[{self._current_idx}]")

        self.caller.update_marker()
        self.caller.update_pixel()

    def set_current_idx(self, x=None, y=None, idx=None):
        self.caller.set_current_idx(x=x, y=y, idx=idx)

    def init_plots(self):
        """
        needs to be implemented by classes that inherit from pyqdm_window
        """

    def _add_scalebars(self):
        for ax in self.img_axes:
            # Create scale bar
            if ax in self._is_image:
                scalebar = ScaleBar(
                    self.pixelsize,
                    "m",
                    length_fraction=0.25,
                    frameon=True,
                    box_alpha=0.5,
                    location="lower left",
                )
            else:
                scalebar = ScaleBar(
                    self.pixelsize * self.qdm.bin_factor,
                    "m",
                    length_fraction=0.25,
                    frameon=True,
                    box_alpha=0.5,
                    location="lower left",
                )
            ax.add_artist(scalebar)

    def _init_lines(self):
        if self._is_spectra is None:
            return
        # init the lists for marker, pixel and fit lines
        self._pixel_marker = np.array([None for _ in self._is_img])
        self._pixel_lines = np.array(
            [[None for _ in np.arange(self.qdm.odmr.n_pol)] for _ in np.arange(self.qdm.odmr.n_frange)]
        )
        self._fit_lines = np.array(
            [[None for _ in np.arange(self.qdm.odmr.n_pol)] for _ in np.arange(self.qdm.odmr.n_frange)]
        )

    def add_mean_odmr(self):
        for f in np.arange(self.qdm.odmr.n_frange):
            for p in np.arange(self.qdm.odmr.n_pol):
                self._pixel_ax[p][f].plot(
                    self.qdm.odmr.f_ghz[f],
                    self.qdm.odmr.mean_odmr[p, f],
                    marker="",
                    ls="--",
                    alpha=0.5,
                    lw=1,
                    label=f"mean ({self.POL[p]})",
                )

    def update_pixel_lines(self):
        if len(self._pixel_ax) == 0:
            return

        pixel_spectra = self.qdm.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        for f in np.arange(self.qdm.odmr.n_frange):
            for p in np.arange(self.qdm.odmr.n_pol):
                if self._pixel_lines[p][f] is None:
                    (self._pixel_lines[p][f],) = self._pixel_ax[p][f].plot(
                        self.qdm.odmr.f_ghz[f],
                        pixel_spectra[p, f],
                        marker=".",
                        markersize=6,
                        mfc="w",
                        linestyle="" if self._includes_fits else "-",
                    )
                else:
                    self._pixel_lines[p][f].set_data(self.qdm.odmr.f_ghz[f], pixel_spectra[p, f])
                self._pixel_lines[p][f].set_label(f"p({self.POL[p]},{self._current_xy[0]},{self._current_xy[1]})")
                self._pixel_ax[p][f].legend(loc="lower left", fontsize=8, ncol=2)
        self.update_pixel_lims()
        self.canvas.draw()

    def update_fit_lines(self):

        if not self._includes_fits:
            return

        parameter = self.qdm.fit.parameter[:, :, [self._current_idx], :]

        for f in np.arange(self.qdm.odmr.n_frange):
            f_new = np.linspace(min(self.qdm.odmr.f_ghz[f]), max(self.qdm.odmr.f_ghz[f]), 200)
            for p in np.arange(self.qdm.odmr.n_pol):
                m_fit = self.model(parameter=parameter[p, f], x=f_new)
                if self._fit_lines[p][f] is None:
                    (self._fit_lines[p][f],) = self._pixel_ax[p][f].plot(
                        f_new,
                        m_fit[0],
                        marker="",
                        linestyle="-",
                        lw=0.8,
                        color=self._pixel_lines[p][f].get_color(),
                    )
                else:
                    self._fit_lines[p][f].set_ydata(m_fit[0])
                self._fit_lines[p][f].set_label(f"fit ({self.POL[p]},{self._current_xy[0]},{self._current_xy[1]})")
                self._pixel_ax[p][f].legend(loc="lower left", fontsize=8, ncol=3)

        self.update_pixel_lims()
        self.canvas.draw()

    def update_pixel_lims(self):
        self.LOG.debug("updating xy limits for the pixel plots")
        for ax in self._pixel_ax.flatten():
            mn = np.min([np.min(l.get_ydata()) for l in ax.get_lines()])
            mx = np.max([np.max(l.get_ydata()) for l in ax.get_lines()])
            ax.set(ylim=(mn * 0.999, mx * 1.001))

    def update_marker(self):
        """
        Update the marker position on the image plots.
        """
        self.LOG.debug(
            f"Updating marker to ({self._current_xy[0] * self.qdm.bin_factor},"
            f"{self._current_xy[1] * self.qdm.bin_factor})"
        )

        x, y = self._current_xy

        for i, a in enumerate(self._is_img):
            if self._pixel_marker[i] is None:
                (self._pixel_marker[i],) = a.plot(x, y, marker="x", color="chartreuse", zorder=100)
            else:
                self._pixel_marker[i].set_data(self._current_xy[0], self._current_xy[1])

        self.canvas.draw()

    def need_extend(self):
        return self.fix_clim_check_box.isChecked() and self.clims_selector.value() != 100

    def update_img_plots(self):
        """
        needs to be implemented by classes that inherit from pyqdm_window
        """

    def update_pixel(self):
        self.caller.update_pixel()

    def redraw_all_plots(self):
        self.update_img_plots()
        self.update_marker()

    @property
    def model(self):
        return [None, models.esrsingle, models.esr15n, models.esr14n][self.qdm._diamond_type]

    @property
    def pixel_size(self):
        return self.qdm.pixel_size
