import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QWidget

from pyqdm.app.canvas import FluorescenceCanvas

matplotlib.rcParams.update(
    {  # 'font.size': 8,
        # 'axes.labelsize': 8,
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    }
)


class FluorescenceWindow(QMainWindow):
    def on_press(self, event):
        if event.inaxes in self.pixel_axes:
            self.LOG.debug(f"clicked in {event.inaxes}")
            return

        if event.xdata is None or event.ydata is None:
            self.LOG.debug("clicked outside of axes")
            return

        if event.button == MouseButton.LEFT and not self.toolbar.mode:
            bin_factor = self.qdm.bin_factor
            # event is in image coordinates
            if event.inaxes in self._is_data:
                xy = [event.xdata, event.ydata]
            else:
                xy = [event.xdata / bin_factor, event.ydata / bin_factor]

            x, y = np.round(xy).astype(int)

            self.xselect.valueChanged.disconnect(self.onXYValueChange)
            self.xselect.setValue(x)
            self.xselect.valueChanged.connect(self.onXYValueChange)

            self.yselect.valueChanged.disconnect(self.onXYValueChange)
            self.yselect.setValue(y)
            self.yselect.valueChanged.connect(self.onXYValueChange)

            self.set_current_idx(x, y)
            self.LOG.debug(f"clicked in {event.inaxes} with new index: {self._current_idx}")
            self.update_marker()
            self.update_plots()

    def __init__(self, qdm_instance=None, pixelsize=1e-6, *args, **kwargs):
        self.qdm = qdm_instance
        self.pixelsize = pixelsize
        self.LOG = logging.getLogger(f"pyqdm.{self.__class__.__name__}")
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Fluorescence Plots")

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = FluorescenceCanvas(self, width=12, height=6, dpi=100)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.fluo_axes = [
            self.canvas.fluo_lowF_pos_ax,
            self.canvas.fluo_lowF_neg_ax,
            self.canvas.fluo_highF_pos_ax,
            self.canvas.fluo_highF_neg_ax,
        ]

        self.img_data = self.qdm.odmr["r"]
        self.indexSlider = QSlider()
        self.indexSlider.setOrientation(Qt.Horizontal)

        self.indexSlider.setMinimum(0)
        self.indexSlider.setMaximum(self.qdm.odmr.n_freqs - 1)
        self.indexSlider.setSingleStep(1)

        self.indexSlider.valueChanged.connect(self.on_slider_value_changed)

        vertical_layout = QVBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout.addWidget(self.toolbar)
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.indexSlider)

        main_widget = QWidget()
        main_widget.setLayout(vertical_layout)

        self.setCentralWidget(main_widget)
        self.init_plots()
        self.resize(900, 900)
        self.show()

    def init_plots(self):
        self._init_odmr_plots()

        self._init_fluorescence_plots()

    def _init_fluorescence_plots(self):
        # fluorescence plots
        vmin, vmax = np.percentile(self.qdm.odmr.data[:, :, :, 0], [2, 98])
        self.fluo_lowF_pos_img = self.canvas.fluo_lowF_pos_ax.imshow(
            self.img_data[0, 0, :, :, 0],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        self.fluo_lowF_neg_img = self.canvas.fluo_lowF_neg_ax.imshow(
            self.img_data[1, 0, :, :, 0],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        self.fluo_highF_pos_img = self.canvas.fluo_highF_pos_ax.imshow(
            self.img_data[0, 1, :, :, 0],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        self.fluo_highF_neg_img = self.canvas.fluo_highF_neg_ax.imshow(
            self.img_data[1, 1, :, :, 0],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )
        for ax in self.fluo_axes:
            ax.set(xlabel="x [px]", ylabel="y [px]")

        self._add_scalebars()

        # pixel marker
        (self.low_pos_pixel,) = self.canvas.fluo_lowF_pos_ax.plot(
            [np.nan],
            [np.nan],
            "x",
            color=self.low_pos_pixel_line.get_color(),
            markersize=5,
            zorder=100,
        )
        (self.low_neg_pixel,) = self.canvas.fluo_lowF_neg_ax.plot(
            [np.nan],
            [np.nan],
            "x",
            color=self.low_neg_pixel_line.get_color(),
            markersize=5,
            zorder=100,
        )
        (self.high_pos_pixel,) = self.canvas.fluo_highF_pos_ax.plot(
            [np.nan],
            [np.nan],
            "x",
            color=self.high_pos_pixel_line.get_color(),
            markersize=5,
            zorder=100,
        )
        (self.high_neg_pixel,) = self.canvas.fluo_highF_neg_ax.plot(
            [np.nan],
            [np.nan],
            "x",
            color=self.high_neg_pixel_line.get_color(),
            markersize=5,
            zorder=100,
        )
        self.cbar = plt.colorbar(
            self.fluo_lowF_pos_img,
            cax=self.canvas.cbar_ax,
            label="ODMR contrast",
            extend="both",
            orientation="horizontal",
        )

    def _add_scalebars(self):
        for ax in self.fluo_axes:
            # Create scale bar
            scalebar = ScaleBar(self.pixelsize, "m", length_fraction=0.25, location="lower left")
            ax.add_artist(scalebar)

    def _init_odmr_plots(self):
        # mean ODMR spectrum lines
        self.canvas.low_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[0],
            self.qdm.odmr.mean_odmr[0, 0],
            "-",
            label="lowF, pos",
            linewidth=0.8,
        )
        self.canvas.low_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[0],
            self.qdm.odmr.mean_odmr[1, 0],
            "-",
            label="lowF, neg",
            linewidth=0.8,
        )
        self.canvas.high_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[1],
            self.qdm.odmr.mean_odmr[0, 1],
            "-",
            label="highF, pos",
            linewidth=0.8,
        )
        self.canvas.high_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[1],
            self.qdm.odmr.mean_odmr[1, 1],
            "-",
            label="highF, neg",
            linewidth=0.8,
        )
        self.lowF_line = self.canvas.low_f_mean_odmr_ax.axvline(
            self.qdm.odmr.f_ghz[0, 0], color="k", alpha=0.7, zorder=0
        )
        self.highF_line = self.canvas.high_f_mean_odmr_ax.axvline(
            self.qdm.odmr.f_ghz[1, 0], color="k", alpha=0.7, zorder=0
        )
        # single ODMR spectrum lines
        (self.low_pos_pixel_line,) = self.canvas.low_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[0],
            [np.nan for _ in self.qdm.odmr.f_ghz[0]],
            ".-",
            mfc="w",
            label="",
            linewidth=0.8,
            markersize=2,
        )
        (self.low_neg_pixel_line,) = self.canvas.low_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[0],
            [np.nan for _ in self.qdm.odmr.f_ghz[0]],
            ".-",
            mfc="w",
            label="",
            linewidth=0.8,
            markersize=2,
        )
        (self.high_pos_pixel_line,) = self.canvas.high_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[1],
            [np.nan for _ in self.qdm.odmr.f_ghz[1]],
            ".-",
            mfc="w",
            label="",
            linewidth=0.8,
            markersize=2,
        )
        (self.high_neg_pixel_line,) = self.canvas.high_f_mean_odmr_ax.plot(
            self.qdm.odmr.f_ghz[1],
            [np.nan for _ in self.qdm.odmr.f_ghz[1]],
            ".-",
            mfc="w",
            label="",
            linewidth=0.8,
            markersize=2,
        )
        for a in [self.canvas.low_f_mean_odmr_ax, self.canvas.high_f_mean_odmr_ax]:
            a.set_xlabel("Frequency (GHz)", fontsize=8)
            a.set_ylabel("ODMR (a.u.)", fontsize=8)
            a.set_title("mean ODMR", fontsize=10)
            a.legend(ncol=1, loc="lower left", fontsize=8)

    def update_pixel_spectrum(self, x, y):
        """
        Update the pixel spectrum plot with the current pixel position.
        Adds a marker to the plot and adds the pixel spectrum to the mean ODMR plot.

        """

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
        h, l = self.canvas.low_f_mean_odmr_ax.get_legend_handles_labels()
        h.extend([self.low_pos_pixel_line, self.low_neg_pixel_line])
        l.extend([f"{labels[0]},{x},{y})", f"{labels[1]},{x},{y})"])
        self.canvas.low_f_mean_odmr_ax.legend(h, l, loc="lower left", fontsize=8)

        h, l = self.canvas.high_f_mean_odmr_ax.get_legend_handles_labels()
        h.extend([self.high_pos_pixel_line, self.high_neg_pixel_line])
        l.extend([f"{labels[2]},{x},{y})", f"{labels[3]},{x},{y})"])
        self.canvas.high_f_mean_odmr_ax.legend(h, l, loc="lower left", fontsize=8)

        # add lines to mean ODMR plot
        self.low_pos_pixel_line.set_ydata(self.qdm.odmr.data[0, 0, idx])
        self.low_neg_pixel_line.set_ydata(self.qdm.odmr.data[0, 1, idx])
        self.high_pos_pixel_line.set_ydata(self.qdm.odmr.data[1, 0, idx])
        self.high_neg_pixel_line.set_ydata(self.qdm.odmr.data[1, 1, idx])

        self.canvas.draw()

    def update_plot(self, idx):
        vmin, vmax = np.percentile(self.qdm.odmr.data[:, :, :, idx], [2, 98])

        self.fluo_lowF_pos_img.set_data(self.img_data[0, 0, :, :, idx])
        self.fluo_lowF_neg_img.set_data(self.img_data[1, 0, :, :, idx])
        self.fluo_highF_pos_img.set_data(self.img_data[0, 1, :, :, idx])
        self.fluo_highF_neg_img.set_data(self.img_data[1, 1, :, :, idx])

        for img in [
            self.fluo_lowF_pos_img,
            self.fluo_lowF_neg_img,
            self.fluo_highF_pos_img,
            self.fluo_highF_neg_img,
        ]:
            img.set_clim(vmin, vmax)

        self.lowF_line.set_xdata([self.qdm.odmr.f_ghz[0, idx], self.qdm.odmr.f_ghz[0, idx]])
        self.highF_line.set_xdata([self.qdm.odmr.f_ghz[1, idx], self.qdm.odmr.f_ghz[1, idx]])

        # self.cbar.vmin= vmin
        # self.cbar.vmax= vmax
        self.cbar.update_normal(self.fluo_lowF_pos_img)
        # self.canvas.cbar_ax.clear()
        # print(self.canvas.cbar_ax)
        # self.cbar = plt.colorbar(self.fluo_lowF_pos_img,
        #                          cax=self.canvas.cbar_ax, label='ODMR contrast',
        #                          # extend='both',
        #                          orientation='horizontal')
        # self.lowF_neg_marker.set_data(self.QDMObj.odmr.f_GHz[0, idx], self.QDMObj.odmr.mean_odmr[1, 0, idx])
        # self.highF_line.set_data(self.QDMObj.odmr.f_GHz[1, idx], self.QDMObj.odmr.mean_odmr[0, 1, idx])
        # self.highF_neg_marker.set_data(self.QDMObj.odmr.f_GHz[1, idx], self.QDMObj.odmr.mean_odmr[1, 1, idx])

        self.canvas.draw()

    def clear_axes(self):
        for a in self.canvas.fig.axes:
            if a not in [
                self.canvas.low_f_mean_odmr_ax,
                self.canvas.high_f_mean_odmr_ax,
                self.canvas.fluo_lowF_pos_ax,
                self.canvas.fluo_lowF_neg_ax,
                self.canvas.fluo_highF_pos_ax,
                self.canvas.fluo_highF_neg_ax,
            ]:
                a.remove()
        self.canvas.low_f_mean_odmr_ax.clear()
        self.canvas.high_f_mean_odmr_ax.clear()
        self.canvas.fluo_lowF_pos_ax.clear()
        self.canvas.fluo_lowF_neg_ax.clear()
        self.canvas.fluo_highF_pos_ax.clear()
        self.canvas.fluo_highF_neg_ax.clear()

    def on_slider_value_changed(self, value):
        # self.clear_axes()
        self.update_plot(value)


#
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
