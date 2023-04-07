import itertools
import logging

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import QDMpy.plotting as qdmplot

POL = ["+", "-"]
FRANGE = ["<", ">"]


class FrequencyToolCanvas(FigureCanvas):
    LOG = logging.getLogger(__name__)

    def __init__(self, *args, **kwargs):
        self.fig = Figure()
        self.fig.subplots_adjust(top=0.8, bottom=0.13, left=0.1, right=0.9)
        self.ax = self.fig.add_subplot(111)

        self.axb = self.ax.twiny()
        self.ax.set_xlabel("Frequency (GHz)")
        self.axb.set_xlabel("B$_{\mathrm{equivalent}}$ [mT]")
        self.ax.set_ylabel("Contrast (a.u.)")
        self.ax.grid(True)

        super().__init__(self.fig)


class QDMCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    LOG = logging.getLogger(__name__)

    @property
    def data_axes(self):
        return list(self.data.keys())

    @property
    def img_axes(self):
        return (
            list(self.light.keys())
            + list(self.laser.keys())
            + list(self.data.keys())
            + list(self.fluorescence.keys())
        )

    @property
    def odmr_axes(self):
        return list(self.odmr.keys())

    @property
    def has_odmr(self):
        return len(self.odmr_axes) != 0

    @property
    def has_img(self):
        return len(self.img_axes) != 0

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.img_dict = {
            "data": None,
            "outlier": None,
            "overlay": None,
            "marker": None,
            "pol": [],
            "frange": [],
            "cax": False,
            "cax_locator": None,
        }
        self.odmr_dict = {
            "data": [[None, None], [None, None]],
            "pol": [],
            "frange": [],
            "mean": [[None, None], [None, None]],
            "fit": [[None, None], [None, None]],
            "corrected": [[None, None], [None, None]],
            "uncorrected": [[None, None], [None, None]],
        }
        self.light = {}  # dict of ax : led img
        self.laser = {}  # laser is a dictionary of dictionaries
        self.fluorescence = {}  # fluorescence is a dictionary of dictionaries
        self.data = {}
        self.odmr = {}  # dictionary of ax : odmr data lines

    def _add_cax(self, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        original_locator = cax.get_axes_locator()
        return cax, original_locator

    def set_img(self, cbar_label=r"B$_{111}$ [$\mu$T]"):
        for axdict in [self.data, self.laser, self.light, self.fluorescence]:
            for a in axdict:
                a.set(xlabel="px", ylabel="px")
                if axdict == self.data and isinstance(self.data[a]["cax"], Axes):
                    self.data[a]["cax"].set_ylabel(cbar_label)
                elif isinstance(axdict[a]["cax"], Axes):
                    axdict[a]["cax"].set_ylabel("intensity [a.u.]")

    def set_odmr(self):
        for a in self.odmr:
            a.set(xlabel="GHz", ylabel="contrast [a.u.]")

    def add_light(self, light, data_dimensions):
        for ax in self.light.keys():
            self.LOG.debug(f"Adding Light image to axis {ax}")
            self.light[ax]["data"] = qdmplot.plot_light_img(
                ax=ax,
                data=light,
                img=self.light[ax]["data"],
                data_dimensions=data_dimensions,
            )

    def add_laser(self, laser, data_dimensions):
        for ax in self.laser:
            self.LOG.debug(f"Adding laser to axis {ax}")
            self.laser[ax]["data"] = qdmplot.plot_laser_img(
                ax,
                laser,
                self.laser[ax]["data"],
                data_dimensions=data_dimensions,
            )

    def add_data(self, data, data_dimensions, p=None, f=None, **plt_props):
        for ax in self.data:
            if p is not None and p not in self.data[ax]["pol"]:
                continue
            if f is not None and f not in self.data[ax]["frange"]:
                continue

            self.data[ax]["data"] = qdmplot.plot_data(
                ax=ax,
                data=data,
                img=self.data[ax]["data"],
                data_dimensions=data_dimensions,
                aspect="equal",
                origin="lower",
                **plt_props,
            )

    def add_outlier_masks(self, outlier):
        for axdict in [self.light, self.laser, self.data]:
            for ax in axdict:
                if axdict[ax]["outlier"] is None:
                    self.LOG.debug(f"Adding outlier mask to axis {ax}")
                    axdict[ax]["outlier"] = ax.imshow(
                        outlier,
                        cmap="gist_rainbow",
                        alpha=outlier.astype(float),
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        origin="lower",
                        aspect="equal",
                        zorder=2,
                    )

    def add_scalebars(self, pixelsize):
        for axdict in [self.light, self.laser, self.data]:
            # Create scale bar
            scalebar = ScaleBar(
                pixelsize,
                "m",
                length_fraction=0.25,
                location="lower left",
                box_alpha=0.5,
                frameon=True,
            )
            for ax in axdict:
                ax.add_artist(scalebar)

    def add_cax(self, ax, axdict, save=True):
        cax, original_locator = self._add_cax(ax)
        if save:
            axdict[ax]["cax"] = cax
            axdict[ax]["cax_locator"] = original_locator
        else:
            [s.set_visible(False) for s in cax.spines.values()]
            cax.set_xticks([])
            cax.set_yticks([])

    def update_marker(self, x, y):
        for axdict in [self.light, self.laser, self.data, self.fluorescence]:
            for ax in axdict:
                self.LOG.debug(f"Updating marker to ({x},{y}) axis {ax}")
                axdict[ax]["marker"] = qdmplot.update_marker(
                    ax,
                    x + 0.5,
                    y + 0.5,
                    line=axdict[ax]["marker"],
                    marker="X",
                    c="m",
                    mfc="w",
                    zorder=100,
                )

    def update_outlier(self, outlier):
        for axdict in [self.laser, self.data, self.fluorescence]:
            for ax in axdict:
                if axdict[ax]["outlier"] == False:
                    continue
                self.LOG.debug(f"Updating outlier mask to axis {ax}")
                axdict[ax]["outlier"] = qdmplot.plot_outlier(
                    ax,
                    outlier,
                    img=axdict[ax]["outlier"],
                )
        self.draw()

    def toggle_outlier(self, visible):
        for axdict in [self.laser, self.data, self.fluorescence]:
            for ax in axdict:
                axdict[ax]["outlier"].set_visible(visible)
        self.draw_idle()

    def add_mean_odmr(self, freq, mean):
        for ax in self.odmr:
            for p, f in itertools.product(self.odmr[ax]["pol"], self.odmr[ax]["frange"]):
                self.LOG.debug(f"Adding mean odmr to axis {ax}")

    def update_odmr(self, freq, data=None, fit=None, corrected=None, uncorrected=None, mean=None):
        for ax in self.odmr:
            self.LOG.debug(f"Updating ODMR lines in axis {ax}")
            for p, f in itertools.product(self.odmr[ax]["pol"], self.odmr[ax]["frange"]):
                if data is not None:
                    self.update_odmr_line(ax, data, freq, p, f)
                if fit is not None:
                    self.update_fit_line(ax, freq, fit, p, f)
                if corrected is not None:
                    self.update_corrected_line(ax, freq, corrected, p, f)
                if uncorrected is not None:
                    self.update_uncorrected_line(ax, freq, uncorrected, p, f)
                if mean is not None:
                    self.update_mean_line(ax, freq, mean, p, f)

    def update_mean_line(self, ax, freq, mean, p, f):
        self.odmr[ax]["mean"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=mean[p, f],
            line=self.odmr[ax]["mean"][p][f],
            ls="--",
            zorder=0,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=0.8,
        )

    def update_uncorrected_line(self, ax, freq, uncorrected, p, f):
        self.odmr[ax]["uncorrected"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=uncorrected[p, f],
            line=self.odmr[ax]["uncorrected"][p][f],
            ls="-.",
            zorder=4,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=0.7,
        )

    def update_corrected_line(self, ax, freq, corrected, p, f):
        self.odmr[ax]["corrected"][p][f] = qdmplot.update_line(
            ax,
            freq[f],
            y=corrected[p, f],
            line=self.odmr[ax]["corrected"][p][f],
            ls="-",
            zorder=3,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=1,
        )

    def update_fit_line(self, ax, freq, fit, p, f):
        self.odmr[ax]["fit"][p][f] = qdmplot.update_line(
            ax,
            np.linspace(freq[f].min(), freq[f].max(), 200),
            fit[p, f],
            line=self.odmr[ax]["fit"][p][f],
            zorder=2,
            color=self.odmr[ax]["data"][p][f].get_color(),
            lw=1,
        )

    def update_odmr_line(self, ax, data, freq, p, f):
        self.odmr[ax]["data"][p][f] = qdmplot.update_line(
            ax=ax,
            x=freq[f],
            y=data[p, f],
            line=self.odmr[ax]["data"][p][f],
            marker=".",
            ls="",
            zorder=1,
            mfc="w",
        )

    def update_extent(self, new_shape):
        for axdict in [self.laser, self.light, self.data, self.fluorescence]:
            for ax in axdict:
                im = axdict[ax]["data"]
                im.set(extent=[0, new_shape[1], 0, new_shape[0]])

    def add_fluorescence(self, fluorescence):
        for ax in self.fluorescence:
            for p, f in itertools.product(
                self.fluorescence[ax]["pol"], self.fluorescence[ax]["frange"]
            ):
                if (
                    p not in self.fluorescence[ax]["pol"]
                    or f not in self.fluorescence[ax]["frange"]
                ):
                    continue
                self.LOG.debug(f"Adding fluorescence of {POL[p]}, {FRANGE[f]} to axis {ax}")
                self.fluorescence[ax]["data"] = qdmplot.plot_fluorescence(
                    ax, fluorescence[p][f], img=self.fluorescence[ax]["data"]
                )

    def update_data(self, ax, data, p, f, data_shape):
        self.odmr[ax]["data"] = qdmplot.plot_data(
            ax=ax,
            data=data,
            zorder=0,
            mfc="w",
            extent=[0, data_shape[1], 0, data_shape[0]],
        )

    def update_odmr_lims(self, data):
        self.LOG.debug("setting xy limits for the pixel plots")
        for ax in self.odmr:
            mn, mx = np.nanmin(data), np.nanmax(data)
            ax.set(ylim=(mn * 0.999, mx * 1.001))

    def update_clims(self, use_percentile, percentile):
        self.LOG.debug(
            f"updating clims for all images with {percentile} percentile ({use_percentile})."
        )

        for axdict in [self.data, self.laser, self.fluorescence]:
            for ax in axdict:
                if axdict[ax]["data"] is None:
                    continue
                vmin, vmax = qdmplot.get_vmin_vmax(axdict[ax]["data"], percentile, use_percentile)
                norm = qdmplot.get_color_norm(vmin=vmin, vmax=vmax)
                axdict[ax]["data"].set(norm=norm)
                if axdict[ax]["cax"]:
                    qdmplot.update_cbar(
                        img=axdict[ax]["data"],
                        cax=axdict[ax]["cax"],
                        vmin=vmin,
                        vmax=vmax,
                        original_cax_locator=axdict[ax]["cax_locator"],
                    )


class StatCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

        spec = self.fig.add_gridspec(ncols=3, nrows=3)
        self.fig.subplots_adjust(
            top=0.952, bottom=0.076, left=0.06, right=0.959, hspace=0.35, wspace=0.594
        )
        self.outlier_ax = self.fig.add_subplot(spec[0:2, :])
        self.data = {self.outlier_ax: self.img_dict.copy()}
        self.add_cax(self.outlier_ax, self.data)

        self.chi_ax = self.fig.add_subplot(spec[2, 0])
        self.chi_ax.set_title("$\chi^2$")
        self.width_ax = self.fig.add_subplot(spec[2, 1])
        self.width_ax.set_title("Width [MHz]")
        self.contrast_ax = self.fig.add_subplot(spec[2, 2])
        self.contrast_ax.set_title("Mean Contrast [%]")


class GlobalFluorescenceCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        self.fig.subplots_adjust(
            top=0.9, bottom=0.09, left=0.075, right=0.925, hspace=0.28, wspace=0.899
        )

        spec = self.fig.add_gridspec(ncols=6, nrows=2)

        self.left_mean_odmr_ax = self.fig.add_subplot(spec[0, :3])
        self.right_mean_odmr_ax = self.fig.add_subplot(spec[0, 3:6])

        self.light_ax = self.fig.add_subplot(spec[1, :3])
        self.laser_ax = self.fig.add_subplot(spec[1, 3:])

        self.light_ax.get_shared_x_axes().join(self.light_ax, self.laser_ax)
        self.light_ax.get_shared_y_axes().join(self.light_ax, self.laser_ax)

        # setup the dictionaries for the data
        self.light = {self.light_ax: self.img_dict.copy()}
        self.add_cax(self.light_ax, self.light, save=False)
        self.laser = {self.laser_ax: self.img_dict.copy()}
        self.add_cax(self.laser_ax, self.laser)

        self.odmr = {
            self.left_mean_odmr_ax: self.odmr_dict.copy(),
            self.right_mean_odmr_ax: self.odmr_dict.copy(),
        }
        self.odmr[self.left_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.left_mean_odmr_ax]["frange"] = [0]
        self.odmr[self.right_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.right_mean_odmr_ax]["frange"] = [1]

        self.set_img()
        self.set_odmr()


class FitCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        spec = self.fig.add_gridspec(ncols=6, nrows=3)
        self.fig.subplots_adjust(
            top=0.952, bottom=0.076, left=0.06, right=0.959, hspace=0.35, wspace=0.594
        )
        self.data_ax = self.fig.add_subplot(spec[0:2, :4])
        self.data = {self.data_ax: self.img_dict.copy()}
        self.add_cax(self.data_ax, self.data)

        self.light_ax = self.fig.add_subplot(spec[0, 4:])
        self.light = {self.light_ax: self.img_dict.copy()}
        self.add_cax(self.light_ax, self.light_ax, save=False)

        self.laser_ax = self.fig.add_subplot(spec[1, 4:])
        self.laser = {self.laser_ax: self.img_dict.copy()}
        self.add_cax(self.laser_ax, self.laser)

        self.data_ax.get_shared_x_axes().join(self.data_ax, self.light_ax, self.laser_ax)
        self.data_ax.get_shared_y_axes().join(self.data_ax, self.light_ax, self.laser_ax)

        self.left_odmr_ax = self.fig.add_subplot(spec[2, :3])
        self.right_odmr_ax = self.fig.add_subplot(spec[2, 3:])

        self.odmr = {
            self.left_odmr_ax: self.odmr_dict.copy(),
            self.right_odmr_ax: self.odmr_dict.copy(),
        }

        # setup the dictionaries for the data
        self.odmr[self.left_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.left_odmr_ax]["frange"] = [0]
        self.odmr[self.right_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.right_odmr_ax]["frange"] = [1]

        self.set_img()
        self.set_odmr()


class SimpleCanvas(QDMCanvas):
    def __init__(self, dtype, width=5, height=4, dpi=100):
        super().__init__(width=width, height=height, dpi=dpi)
        self.fig.subplots_adjust(
            top=0.97, bottom=0.082, left=0.106, right=0.979, hspace=0.2, wspace=0.2
        )

        if "light" in dtype.lower():
            self.light_ax = self.fig.add_subplot(111)
            self.light = {self.light_ax: self.img_dict.copy()}
        elif "laser" in dtype.lower():
            self.laser_ax = self.fig.add_subplot(111)
            self.laser = {self.laser_ax: self.img_dict.copy()}
            self.add_cax(self.laser_ax, self.laser, save=True)
        else:
            raise ValueError(f"dtype {dtype} not recognized")


class FluoImgCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        widths = [1, 1]
        heights = [1, 1, 1, 0.1]
        gs = self.fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)
        self.fig.subplots_adjust(
            top=0.981, bottom=0.019, left=0.097, right=0.93, hspace=0.47, wspace=0.406
        )
        self.low_f_mean_odmr_ax = self.fig.add_subplot(gs[0, 0])
        self.high_f_mean_odmr_ax = self.fig.add_subplot(gs[0, 1])

        self.fluo_lowF_pos_ax = self.fig.add_subplot(gs[1, 0])
        self.fluo_highF_pos_ax = self.fig.add_subplot(gs[1, 1])
        self.fluo_lowF_neg_ax = self.fig.add_subplot(gs[2, 0])
        self.fluo_highF_neg_ax = self.fig.add_subplot(gs[2, 1])

        # setup the dictionaries for the fluorescence data
        self.fluorescence = {
            self.fluo_lowF_pos_ax: self.img_dict.copy(),
            self.fluo_highF_pos_ax: self.img_dict.copy(),
            self.fluo_lowF_neg_ax: self.img_dict.copy(),
            self.fluo_highF_neg_ax: self.img_dict.copy(),
        }
        self.fluorescence[self.fluo_lowF_pos_ax]["pol"] = [0]
        self.fluorescence[self.fluo_highF_pos_ax]["pol"] = [0]
        self.fluorescence[self.fluo_lowF_neg_ax]["pol"] = [1]
        self.fluorescence[self.fluo_highF_neg_ax]["pol"] = [1]

        self.fluorescence[self.fluo_lowF_pos_ax]["frange"] = [0]
        self.fluorescence[self.fluo_highF_pos_ax]["frange"] = [1]
        self.fluorescence[self.fluo_lowF_neg_ax]["frange"] = [0]
        self.fluorescence[self.fluo_highF_neg_ax]["frange"] = [1]

        self.add_cax(self.fluo_lowF_pos_ax, self.fluorescence)
        self.add_cax(self.fluo_highF_pos_ax, self.fluorescence)
        self.add_cax(self.fluo_lowF_neg_ax, self.fluorescence)
        self.add_cax(self.fluo_highF_neg_ax, self.fluorescence)

        self.odmr = {
            self.low_f_mean_odmr_ax: self.odmr_dict.copy(),
            self.high_f_mean_odmr_ax: self.odmr_dict.copy(),
        }
        self.odmr[self.low_f_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.low_f_mean_odmr_ax]["frange"] = [0]
        self.odmr[self.high_f_mean_odmr_ax]["pol"] = [0, 1]
        self.odmr[self.high_f_mean_odmr_ax]["frange"] = [1]

        self.set_img()
        self.set_odmr()


class QualityCanvas(QDMCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

        self.fig.subplots_adjust(
            top=0.94, bottom=0.029, left=0.061, right=0.921, hspace=0.0, wspace=0.346
        )

        gs = self.fig.add_gridspec(ncols=2, nrows=2)
        self.low_f_neg = self.fig.add_subplot(gs[0, 0])
        self.low_f_pos = self.fig.add_subplot(gs[1, 0])
        self.high_f_neg = self.fig.add_subplot(gs[0, 1])
        self.high_f_pos = self.fig.add_subplot(gs[1, 1])

        self.low_f_neg.get_shared_x_axes().join(
            self.low_f_neg, self.low_f_pos, self.high_f_neg, self.high_f_pos
        )
        self.low_f_neg.get_shared_y_axes().join(
            self.low_f_neg, self.low_f_pos, self.high_f_neg, self.high_f_pos
        )

        self.data[self.low_f_neg] = self.img_dict.copy()
        self.data[self.low_f_pos] = self.img_dict.copy()
        self.data[self.high_f_neg] = self.img_dict.copy()
        self.data[self.high_f_pos] = self.img_dict.copy()

        # setup the dictionaries for the data
        self.data[self.low_f_neg]["pol"] = [0]
        self.data[self.low_f_neg]["frange"] = [0]
        self.data[self.low_f_pos]["pol"] = [1]
        self.data[self.low_f_pos]["frange"] = [0]
        self.data[self.high_f_neg]["pol"] = [0]
        self.data[self.high_f_neg]["frange"] = [1]
        self.data[self.high_f_pos]["pol"] = [1]
        self.data[self.high_f_pos]["frange"] = [1]

        for ax in [self.low_f_neg, self.low_f_pos, self.high_f_neg, self.high_f_pos]:
            self.add_cax(ax, self.data)


if __name__ == "__main__":
    c = GlobalFluorescenceCanvas()
    print("c:", c.data_axes)
