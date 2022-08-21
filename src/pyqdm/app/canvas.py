import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyqdm.plotting as qdmplot
from matplotlib_scalebar.scalebar import ScaleBar


class PyQdmCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    @property
    def data_axes(self):
        return list(self.data.keys())

    @property
    def img_axes(self):
        return (
                list(self.light.keys())
                + list(self.laser.keys())
                + list(self.data.keys())
                + list(self.outlier.keys())
                + list(self.overlay.keys())
        )

    @property
    def odmr_axes(self):
        return list(self.odmr.keys())

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.LOG = logging.getLogger(f"pyQDM.{self.__class__.__name__}")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PyQdmCanvas, self).__init__(self.fig)
        self.img_dict = {'data': None,
                         'outlier': None,
                         'overlay': None,
                         'marker': None,
                         'cax': False,
                         'cax_locator': None}
        self.odmr_dict = {'data': [[None, None], [None, None]],
                          'pol': [],
                          'frange': [],
                          'mean': [[None, None], [None, None]],
                          'fit': [[None, None], [None, None]],
                          'corrected': [[None, None], [None, None]],
                          'uncorrected': [[None, None], [None, None]]}
        self.light = {}  # dict of ax : led img
        self.laser = {}  # laser is a dictionary of dictionaries
        self.data = {}
        self.odmr = {}  # dictionary of ax : odmr data lines

    def add_cax(self, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        original_locator = cax.get_axes_locator()
        return cax, original_locator

    def set_img(self):
        for a in self.light:
            a.set(xlabel="px", ylabel="px", title="Light", aspect="equal", origin="lower")

    def add_light(self, light, data_dimensions):
        for ax in self.light.keys():
            self.LOG.debug(f"Adding Light image to axis {ax}")
            self.light[ax]['data'] = qdmplot.plot_light_img(
                ax=ax, data=light, img=self.light[ax]['data'],
                extent=[0, data_dimensions[1], 0, data_dimensions[0]],
            )

    def add_laser(self, laser, data_dimensions):
        for ax in self.laser:
            self.LOG.debug(f"Adding laser to axis {ax}")
            self.laser[ax]['data'] = qdmplot.plot_laser_img(
                ax, laser, self.laser[ax]['data'],
                extent=[0, data_dimensions[1], 0, data_dimensions[0]],
            )
            if self.laser[ax]['cax']:
                plt.colorbar(self.laser[ax]['data'], cax=self.laser[ax]['cax'])

    def add_outlier_masks(self, outlier):
        for axdict in [self.light, self.laser, self.data]:
            for ax in axdict:
                if axdict[ax]['outlier'] is None:
                    self.LOG.debug(f"Adding outlier mask to axis {ax}")
                    axdict[ax]['outlier'] = ax.imshow(
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

    def add_scalebars(self, pixelsize   ):
        for axdict in [self.light, self.laser, self.data]:
            # Create scale bar
            scalebar = ScaleBar(pixelsize, "m", length_fraction=0.25, location="lower left")
            for ax in axdict:
                ax.add_artist(scalebar)

    def update_marker(self, x, y):
        for axdict in [self.light, self.laser, self.data]:
            for ax in axdict:
                if axdict[ax]['marker'] is None:
                    self.LOG.debug(f"Adding marker to axis {ax}")
                    axdict[ax]['marker'], = ax.plot(x, y, marker="x", c="c", zorder=100)
                else:
                    self.LOG.debug(f"Updating marker to axis {ax}")
                    axdict[ax]['marker'].set_data(x, y)
        self.draw()

    def update_outlier_masks(self, outlier):
        for ax, img in self.outlier.items():
            self.LOG.debug(f"Updating outlier mask to axis {ax}")
            img.set_data(outlier)

    def add_odmr(self, freq, data=None, fit=None, corrected=None, uncorrected=None):
        for ax in self.odmr:
            for p in self.odmr[ax]['pol']:
                for f in self.odmr[ax]['frange']:
                    self.LOG.debug(f"Adding odmr to axis {ax}")
                    if data is not None:
                        pl, = ax.plot(freq[f], data[p, f], '.',
                                      zorder=1, mfc='w', ms=5)
                        self.odmr[ax]['data'][p][f] = pl

                    if fit is not None:
                        fl, = ax.plot(freq[f], fit[p, f], '-',
                                      zorder=2, color=pl.get_color(), lw=1)
                        self.odmr[ax]['fit'][p][f] = fl

                    if corrected is not None:
                        cl, = ax.plot(freq[f], corrected[p, f], '-',
                                      zorder=3, color=pl.get_color(), lw=1)
                        self.odmr[ax]['corrected'][p][f] = cl

                    if uncorrected is not None:
                        if np.all(uncorrected[p, f] == data[p, f]):
                            continue
                        ul, = ax.plot(freq[f], uncorrected[p, f], '-.',
                                      zorder=4, color=pl.get_color(), lw=0.7)
                        self.odmr[ax]['uncorrected'][p][f] = ul

    def update_odmr(self, data=None, fit=None, corrected=None, uncorrected=None):
        for ax in self.odmr:
            for p in self.odmr[ax]['pol']:
                for f in self.odmr[ax]['frange']:
                    self.LOG.debug(f"Updating odmr in axis {ax}")
                    if data is not None and self.odmr[ax]['data'][p][f] is not None:
                        self.odmr[ax]['data'][p][f].set_ydata(data[p, f])
                    if fit is not None and self.odmr[ax]['fit'][p][f] is not None:
                        self.odmr[ax]['fit'][p][f].set_ydata(fit[p, f])
                    if corrected is not None and self.odmr[ax]['corrected'][p][f] is not None:
                        self.odmr[ax]['corrected'][p][f].set_ydata(corrected[p, f])
                    if uncorrected is not None and self.odmr[ax]['uncorrected'][p][f] is not None:
                        self.odmr[ax]['uncorrected'][p][f].set_ydata(uncorrected[p, f])
        self.draw()

    def update_odmr_lims(self):
        self.LOG.debug("updating xy limits for the pixel plots")
        for ax in self.odmr:
            mn = np.min([np.min(l.get_ydata()) for l in ax.get_lines()])
            mx = np.max([np.max(l.get_ydata()) for l in ax.get_lines()])
            for p in self.odmr[ax]['pol']:
                for f in self.odmr[ax]['frange']:
                    ax.set(ylim=(mn * 0.999, mx * 1.001))
        self.draw()


class GlobalFluorescenceCanvas(PyQdmCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(GlobalFluorescenceCanvas, self).__init__(parent, width, height, dpi)

        self.fig.subplots_adjust(top=0.9, bottom=0.09, left=0.075, right=0.925, hspace=0.28, wspace=0.899)

        spec = self.fig.add_gridspec(ncols=6, nrows=2)

        self.left_mean_odmr_ax = self.fig.add_subplot(spec[0, :3])
        self.right_mean_odmr_ax = self.fig.add_subplot(spec[0, 3:6])

        self.light_ax = self.fig.add_subplot(spec[1, :3])
        self.laser_ax = self.fig.add_subplot(spec[1, 3:])

        self.light_ax.get_shared_x_axes().join(self.light_ax, self.laser_ax)
        self.light_ax.get_shared_y_axes().join(self.light_ax, self.laser_ax)

        # setup the dictionaries for the data
        self.light = {self.light_ax: self.img_dict.copy()}
        self.laser = {self.laser_ax: self.img_dict.copy()}
        cax, original_locator = self.add_cax(self.laser_ax)
        self.laser[self.laser_ax]['cax'] = cax
        self.laser[self.laser_ax]['cax_locator'] = original_locator

        self.odmr = {self.left_mean_odmr_ax: self.odmr_dict.copy(),
                     self.right_mean_odmr_ax: self.odmr_dict.copy()}
        self.odmr[self.left_mean_odmr_ax]['pol'] = [0, 1]
        self.odmr[self.left_mean_odmr_ax]['frange'] = [0]
        self.odmr[self.right_mean_odmr_ax]['pol'] = [0, 1]
        self.odmr[self.right_mean_odmr_ax]['frange'] = [1]


class FittingPropertyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.966, bottom=0.06, left=0.056, right=0.985, hspace=0.325, wspace=1.0)

        self.fig = fig

        spec = fig.add_gridspec(ncols=6, nrows=3)

        self.main_ax = fig.add_subplot(spec[0:2, :4])
        divider = make_axes_locatable(self.main_ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        self.original_cax_locator = self.cax._axes_locator

        self.led_ax = fig.add_subplot(spec[0, 4:])
        self.led_ax.set_title("reflected light")
        self.led_ax.set_xlabel("px")
        self.led_ax.set_ylabel("px")

        self.laser_ax = fig.add_subplot(spec[1, 4:])
        self.laser_ax.set_title("laser")
        self.laser_ax.set_xlabel("px")
        self.laser_ax.set_ylabel("px")

        self.main_ax.get_shared_x_axes().join(self.main_ax, self.led_ax, self.laser_ax)
        self.main_ax.get_shared_y_axes().join(self.main_ax, self.led_ax, self.laser_ax)

        self.left_ODMR_ax = fig.add_subplot(spec[2, :3])
        self.left_ODMR_ax.set_title("low freq. ODMR")
        self.left_ODMR_ax.set_xlabel("frequency [GHz]")
        self.left_ODMR_ax.set_ylabel("contrast [a.u.]")

        self.right_ODMR_ax = fig.add_subplot(spec[2, 3:])
        self.right_ODMR_ax.set_title("high freq. ODMR")
        self.right_ODMR_ax.set_xlabel("frequency [GHz]")
        self.right_ODMR_ax.set_ylabel("contrast [a.u.]")

        self._is_spectra = [self.left_ODMR_ax, self.right_ODMR_ax]
        self._is_data = [self.main_ax]
        self._is_img = [self.main_ax, self.led_ax, self.laser_ax]
        super().__init__(fig)


class GlobalFluorescenceCanvasOLD(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.9, bottom=0.09, left=0.075, right=0.925, hspace=0.28, wspace=0.899)

        self.fig = fig
        widths = [1, 1, 1, 1, 1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=6, nrows=2, width_ratios=widths, height_ratios=heights)

        self.left_mean_odmr_ax = fig.add_subplot(spec[0, :3])
        self.right_meanODMR_ax = fig.add_subplot(spec[0, 3:6])

        self.led_ax = fig.add_subplot(spec[1, :3])
        self.laser_ax = fig.add_subplot(spec[1, 3:])

        self.led_ax.get_shared_x_axes().join(self.led_ax, self.laser_ax)
        self.led_ax.get_shared_y_axes().join(self.led_ax, self.laser_ax)
        super().__init__(fig)


class FluorescenceCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.963, bottom=0.064, left=0.099, right=0.982, hspace=0.355, wspace=0.253)

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1, 1, 0.1]
        spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)

        self.low_f_mean_odmr_ax = fig.add_subplot(gs[0, 0])
        self.high_f_mean_odmr_ax = fig.add_subplot(gs[0, 1])

        self.fluo_lowF_pos_ax = fig.add_subplot(gs[1, 0])
        self.fluo_highF_pos_ax = fig.add_subplot(gs[1, 1])
        self.fluo_lowF_neg_ax = fig.add_subplot(gs[2, 0])
        self.fluo_highF_neg_ax = fig.add_subplot(gs[2, 1])
        self.cbar_ax = fig.add_subplot(gs[3, :])

        self.fluo_lowF_pos_ax.get_shared_x_axes().join(
            self.fluo_lowF_pos_ax,
            *[self.fluo_lowF_pos_ax, self.fluo_lowF_neg_ax, self.fluo_highF_neg_ax, self.fluo_highF_pos_ax],
        )
        self.fluo_lowF_pos_ax.get_shared_y_axes().join(
            self.fluo_lowF_pos_ax,
            *[self.fluo_lowF_pos_ax, self.fluo_lowF_neg_ax, self.fluo_highF_neg_ax, self.fluo_highF_pos_ax],
        )

        super().__init__(fig)


class SimpleCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, cax=True):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.97, bottom=0.055, left=0.075, right=0.925, hspace=0.28, wspace=0.23)

        self.fig = fig
        self.ax = fig.add_subplot(111)

        if cax:
            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes("right", size="5%", pad=0.05)
            self.original_cax_locator = self.cax._axes_locator
        else:
            self.cax = None
            self.original_cax_locator = None

        self._is_img = [self.ax]
        self._is_spectra = []
        self._is_data = []

        super().__init__(fig)


class QualityCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=False)
        fig.subplots_adjust(top=0.94, bottom=0.054, left=0.038, right=0.957, hspace=0.15, wspace=0.167)

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)

        self.left_top_ax = fig.add_subplot(spec[0, 0])
        self.right_top_ax = fig.add_subplot(spec[0, 1])
        self.left_bottom_ax = fig.add_subplot(spec[1, 0])
        self.right_bottom_ax = fig.add_subplot(spec[1, 1])

        self.left_top_ax.get_shared_x_axes().join(
            self.left_top_ax, self.left_bottom_ax, self.right_top_ax, self.right_bottom_ax
        )
        self.left_top_ax.get_shared_y_axes().join(
            self.left_top_ax, self.left_bottom_ax, self.right_top_ax, self.right_bottom_ax
        )

        self.ax = np.array([[self.left_top_ax, self.right_top_ax], [self.left_bottom_ax, self.right_bottom_ax]])
        self._is_img = self.ax.flatten()
        self._is_spectra = []

        for a in self.ax.flatten():
            a.set(xlabel="px", ylabel="px")

        self.caxes = np.array([[None, None], [None, None]])
        self.original_cax_locator = np.array([[None, None], [None, None]])

        for p, f in itertools.product(range(2), range(2)):
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(self.ax[p][f])
            self.caxes[p][f] = divider.append_axes("right", size="5%", pad=0.05)
            self.original_cax_locator[p][f] = self.caxes[p][f]._axes_locator

        super().__init__(fig)


if __name__ == "__main__":
    c = GlobalFluorescenceCanvas()
    print("c:", c.data_axes)
