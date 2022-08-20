import logging

import matplotlib
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pyqdm.app.canvas import GlobalFluorescenceCanvas
from pyqdm.app.windows.misc import gf_applied_window
from pyqdm.app.windows.pyqdm_plot_window import PyQdmWindow
from pyqdm.app.windows.tools import get_label_box

matplotlib.rcParams.update(
    {  # 'font.size': 8,
        # 'axes.labelsize': 8,
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    }
)


class GlobalFluorescenceWindow(PyQdmWindow):
    def __init__(self, caller, qdm_instance, *args, **kwargs):
        canvas = GlobalFluorescenceCanvas()
        super().__init__(caller=caller, canvas=canvas, qdm_instance=qdm_instance, *args, **kwargs)
        self.setWindowTitle("Global Fluorescence")
        self.gf_label = QLabel(f"Global Fluorescence: {self.qdm.global_factor:.2f}")
        self.gfSlider = QSlider()
        self.gfSlider.setValue(self.caller.gf_select.value())
        self.gfSlider.setRange(0, 100)
        self.gfSlider.setOrientation(Qt.Horizontal)
        # self.gfSlider.valueChanged.connect(self.on_slider_value_changed)
        self.applyButton = QPushButton("Apply")
        # self.applyButton.clicked.connect(self.apply_global_factor)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.gf_label)
        horizontal_layout.addWidget(self.gfSlider)
        horizontal_layout.addWidget(self.applyButton)

        self.mainVerticalLayout.addLayout(horizontal_layout)
        self.set_main_window()

        self.canvas.add_light(self.qdm.light, self.qdm.scan_dimensions)
        self.canvas.add_laser(self.qdm.laser, self.qdm.scan_dimensions)


class GlobalFluorescenceWindowOLD(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    def on_press(self, event):
        if event.inaxes in self.pixel_axes:
            self.LOG.debug(f"clicked in {event.inaxes}")
            return
        if event.xdata is None or event.ydata is None:
            self.LOG.debug("clicked outside of axes")
            return
        if event.button == MouseButton.LEFT and not self.toolbar.mode:
            bin_factor = self.qdm.bin_factor
            xy = [event.xdata / bin_factor, event.ydata / bin_factor]
            x, y = np.round(xy).astype(int)
            self.xselect.valueChanged.disconnect(self.on_xy_value_change)
            self.xselect.setValue(x)
            self.xselect.valueChanged.connect(self.on_xy_value_change)
            self.yselect.valueChanged.disconnect(self.on_xy_value_change)
            self.yselect.setValue(y)
            self.yselect.valueChanged.connect(self.on_xy_value_change)
            self.LOG.debug(f"clicked in {event.inaxes} with new index: {self._current_idx}")

            self._current_idx = self.qdm.odmr.rc2idx([y, x])
            self.update_marker()
            self.update_plots()

    @property
    def _current_xy(self):
        return self.qdm.odmr.idx2rc(self._current_idx)[::-1]

    def __init__(self, main_window, qdm_instance=None, pixelsize=1e-6, *args, **kwargs):
        self.LOG = logging.getLogger(f"pyQDM.{self.__class__.__name__}")
        self.main_window = main_window
        self.qdm = qdm_instance
        self.pixelsize = pixelsize

        super().__init__(*args, **kwargs)
        self.setWindowTitle("Global Fluorescence Estimation")
        super(GlobalFluorescenceWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Global Fluorescence Estimation")

        # Create the maptlotlib FigureCanvas object,
        self.canvas = GlobalFluorescenceCanvas(self, width=12, height=6, dpi=100)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.pixel_axes = [self.canvas.left_mean_odmr_ax, self.canvas.right_meanODMR_ax]
        self.img_axes = [self.canvas.led_ax, self.canvas.laser_ax]
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.pixel_axes = self.canvas.odmr_axes
        self.img_axes = self.canvas.img_axes

        self._current_idx = self.qdm.odmr.get_most_divergent_from_mean()[-1]
        self.LOG.debug(f"setting index of worst pixel to {self._current_xy} ({self._current_idx})")

        vertical_layout = QVBoxLayout()
        horizontal_layout_top = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        horizontal_layout_top.addWidget(self.toolbar)

        label = QLabel("Data pixel:")
        horizontal_layout_top.addWidget(label)
        self.xlabel, self.xselect = get_label_box(
            "x",
            int(self._current_xy[0]),
            0,
            1,
            0,
            self.qdm.odmr.scan_dimensions[0],
            self.on_xy_value_change,
        )
        horizontal_layout_top.addWidget(self.xlabel)
        horizontal_layout_top.addWidget(self.xselect)
        self.ylabel, self.yselect = get_label_box(
            "y",
            int(self._current_xy[1]),
            0,
            1,
            0,
            self.qdm.odmr.scan_dimensions[1],
            self.on_xy_value_change,
        )

        self.xselect.valueChanged.disconnect(self.on_xy_value_change)
        self.xselect.setValue(int(self._current_xy[0]))
        self.xselect.valueChanged.connect(self.on_xy_value_change)

        self.yselect.valueChanged.disconnect(self.on_xy_value_change)
        self.yselect.setValue(int(self._current_xy[1]))
        self.yselect.valueChanged.connect(self.on_xy_value_change)

        self.indexLabel = QLabel(f"({self._current_idx}")

        horizontal_layout_top.addWidget(self.ylabel)
        horizontal_layout_top.addWidget(self.yselect)
        horizontal_layout_top.addWidget(self.indexLabel)

        vertical_layout.addLayout(horizontal_layout_top)
        vertical_layout.addWidget(self.canvas)

        self.gf_label = QLabel(f"Global Fluorescence: {self.qdm.global_factor:.2f}")
        self.gfSlider = QSlider()
        self.gfSlider.setValue(self.main_window.gf_select.value())
        self.gfSlider.setRange(0, 100)
        self.gfSlider.setOrientation(Qt.Horizontal)
        self.gfSlider.valueChanged.connect(self.on_slider_value_changed)
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.apply_global_factor)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.gf_label)
        horizontal_layout.addWidget(self.gfSlider)
        horizontal_layout.addWidget(self.applyButton)

        vertical_layout.addLayout(horizontal_layout)
        main_widget = QWidget()
        main_widget.setLayout(vertical_layout)

        self.setCentralWidget(main_widget)

        self._corrected_lines = [[None, None], [None, None]]
        self._pixel_lines = [[None, None], [None, None]]
        self._uncorrected_lines = [[None, None], [None, None]]
        self._marker_line = [None, None]

        self.init_plots()
        self.resize(1000, 700)
        self.show()

    def init_plots(self):
        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()

        pols = ["+", "-"]

        for f in np.arange(self.qdm.odmr.n_frange):
            for p in np.arange(self.qdm.odmr.n_pol):
                (ul,) = self.pixel_axes[f].plot(
                    self.qdm.odmr.f_ghz[f],
                    uncorrected[p, f],
                    ".--",
                    mfc="w",
                    label=f"{pols[p]} original",
                    lw=0.8,
                )
                self._uncorrected_lines[p][f] = ul

                (cl,) = self.pixel_axes[f].plot(
                    self.qdm.odmr.f_ghz[f],
                    corrected[p, f],
                    ".-",
                    label=f"{pols[p]} corrected",
                    color=ul.get_color(),
                )
                self._corrected_lines[p][f] = cl

                if self.qdm.odrm.global_factor != 0:
                    (pl,) = self.pixel_axes[f].plot(
                        self.qdm.odmr.f_ghz[f],
                        pixel_spectra[p, f],
                        ":",
                        label=f"{pols[p]} current: GF={self.qdm.odrm.global_factor}",
                        color=ul.get_color(),
                        lw=0.8,
                    )
                if self.qdm.odmr._gf_factor != 0:
                    (pl,) = self.pixel_axes[f].plot(
                        self.qdm.odmr.f_ghz[f],
                        pixel_spectra[p, f],
                        ":",
                        label=f"{pols[p]} current: GF={self.qdm.odmr._gf_factor}",
                        color=ul.get_color(),
                        lw=0.8,
                    )
                    self._pixel_lines[p][f] = pl

            if self.qdm.odrm.global_factor != 0:
                h, l = self.pixel_axes[f].get_legend_handles_labels()
            if self.qdm.odmr._gf_factor != 0:
                h, labels = self.pixel_axes[f].get_legend_handles_labels()
                h = np.array(h).reshape((2, -1)).T.flatten()
                l = np.array(l).reshape((2, -1)).T.flatten()
                self.pixel_axes[f].legend(
                    h,
                    l,
                    ncol=3,
                    bbox_to_anchor=(0, 1.01),
                    loc="lower left",
                    borderaxespad=0.0,
                    frameon=False,
                    prop={"family": "DejaVu Sans Mono"},
                )
                labels = np.array(labels).reshape((2, -1)).T.flatten()
                self.pixel_axes[f].legend(
                    h,
                    labels,
                    ncol=3,
                    bbox_to_anchor=(0, 1.01),
                    loc="lower left",
                    borderaxespad=0.0,
                    frameon=False,
                    prop={"family": "DejaVu Sans Mono"},
                )
            else:
                self.pixel_axes[f].legend(
                    ncol=2,
                    bbox_to_anchor=(0, 1.01),
                    loc="lower left",
                    borderaxespad=0.0,
                    frameon=False,
                    prop={"family": "DejaVu Sans Mono"},
                )

            self.pixel_axes[f].set(ylabel="ODMR contrast", xlabel="Frequency [GHz]", ylim=(mn, mx))

        self.img_axes[0].imshow(self.qdm.light, cmap="gray", interpolation="none", origin="lower")
        self.img_axes[1].imshow(self.qdm.laser, cmap="inferno", interpolation="none", origin="lower")
        (self._marker_line[0],) = self.img_axes[0].plot(
            self._current_xy[1] * self.qdm.bin_factor,
            self._current_xy[0] * self.qdm.bin_factor,
            "cx",
            markersize=5,
            zorder=10,
        )
        (self._marker_line[1],) = self.img_axes[1].plot(
            self._current_xy[1] * self.qdm.bin_factor,
            self._current_xy[0] * self.qdm.bin_factor,
            "cx",
            markersize=5,
            zorder=10,
        )
        self.img_axes[0].set(xlabel="x [px]", ylabel="y [px]")
        self.img_axes[1].set(xlabel="x [px]", ylabel="y [px]")
        self._add_scalebars()
        self.canvas.draw()

    def get_pixel_data(self):
        gf_factor = self.gfSlider.value() / 100
        new_correct = self.qdm.odmr.get_gf_correction(gf=gf_factor)
        old_correct = self.qdm.odmr.get_gf_correction(gf=self.qdm.odrm.global_factor)
        pixel_spectra = self.qdm.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        uncorrected = pixel_spectra + old_correct
        corrected = uncorrected - new_correct
        mn = np.min([np.min(pixel_spectra), np.min(corrected), np.min(uncorrected)]) * 0.998
        mx = np.max([np.max(pixel_spectra), np.max(corrected), np.max(uncorrected)]) * 1.002
        return pixel_spectra, uncorrected, corrected, mn, mx

    def _add_scalebars(self):
        for ax in self.img_axes:
            # Create scale bar
            scalebar = ScaleBar(self.pixelsize, "m", length_fraction=0.25, location="lower left")
            ax.add_artist(scalebar)

    def update_pixel_spectrum(self, x, y):
        """
        Update the pixel spectrum plot with the current pixel position.
        Adds a marker to the plot and adds the pixel spectrum to the mean ODMR plot.

        """
        if self.toolbar.mode:
            return

        idx = self.qdm.odmr.rc2idx([y, x])  # get the index of the current pixel
        labels = ["p(<+", "p(<-", "p(>+", "p(>-"]
        # update the pixel spectrum plot
        for l in [
            self.low_pos_pixel,
            self.low_neg_pixel,
            self.high_pos_pixel,
            self.high_neg_pixel,
        ]:
            l.set_data(x, y)

        # update the mean ODMR plot legend
        h, l = self.canvas.lowF_meanODMR_ax.get_legend_handles_labels()
        h.extend([self.low_pos_pixel_line, self.low_neg_pixel_line])
        l.extend([f"{labels[0]},{x},{y})", f"{labels[1]},{x},{y})"])
        self.canvas.lowF_meanODMR_ax.legend(h, l, loc="lower left", fontsize=8)

        h, l = self.canvas.highF_meanODMR_ax.get_legend_handles_labels()
        h.extend([self.high_pos_pixel_line, self.high_neg_pixel_line])
        l.extend([f"{labels[2]},{x},{y})", f"{labels[3]},{x},{y})"])
        self.canvas.highF_meanODMR_ax.legend(h, l, loc="lower left", fontsize=8)

        # add lines to mean ODMR plot
        self.low_pos_pixel_line.set_ydata(self.qdm.odmr.data[0, 0, idx])
        self.low_neg_pixel_line.set_ydata(self.qdm.odmr.data[0, 1, idx])
        self.high_pos_pixel_line.set_ydata(self.qdm.odmr.data[1, 0, idx])
        self.high_neg_pixel_line.set_ydata(self.qdm.odmr.data[1, 1, idx])

        self.canvas.draw()

    def update_plots(self):
        """
        Update the plot with the current index.
        """
        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()

        for p in np.arange(self.qdm.odmr.n_pol):
            for f in np.arange(self.qdm.odmr.n_frange):
                self._uncorrected_lines[p][f].set_ydata(uncorrected[p, f])
                self._corrected_lines[p][f].set_ydata(corrected[p, f])
                if self.qdm.odmr._gf_factor != 0:
                    self._pixel_lines[p][f].set_ydata(pixel_spectra[p, f])
                if self.qdm.odrm.global_factor != 0:
                    self._pixel_lines[p][f].set_ydata(pixel_spectra[p, f])

        for a in self.pixel_axes:
            a.set_ylim(mn, mx)

        self.canvas.draw()

    def update_marker(self):
        """
        Update the marker position on the image plots.
        """
        self.LOG.debug(
            f"Updating marker to ({self._current_xy[0] * self.qdm.bin_factor},{self._current_xy[1] * self.qdm.bin_factor})"
        )

        for i, a in enumerate(self.img_axes):
            self._marker_line[i].set_data(
                self._current_xy[0] * self.qdm.bin_factor,
                self._current_xy[1] * self.qdm.bin_factor,
            )
        self.canvas.draw()

    def on_frange_selector_changed(self, frange):
        self.LOG.info(f"frange changed to {frange}")

    def on_slider_value_changed(self):
        self.gf_label.setText(f"Global Fluorescence: {self.gfSlider.value() / 100:.2f}")
        self.update_plots()

    def on_xy_value_change(self):
        self._current_idx = self.qdm.odmr.rc2idx([int(self.yselect.value()), int(self.xselect.value())])
        self.LOG.debug(f"XY value changed to {self._current_xy} ({self._current_idx})")
        self.indexLabel.setText(f"({self._current_idx})")
        self.update_plots()
        self.update_marker()

    def apply_global_factor(self):
        self.LOG.debug(f"applying global factor {self.gfSlider.value() / 100:.2f}")
        self.qdm.odmr.correct_glob_fluorescence(self.gfSlider.value() / 100)
        gf_applied_window(self.gfSlider.value() / 100)

        self.main_window.gf_select.setValue(self.gfSlider.value() / 100)
        self.close()


# class MplCanvas(FigureCanvas):
#     LOG = logging.getLogger(f'pyQDM.{self.__class__.__name__}')
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
#     LOG = logging.getLogger(f'pyQDM.{self.__class__.__name__}')
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
