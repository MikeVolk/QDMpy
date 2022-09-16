import itertools

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


class FittingPropertyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.974, bottom=0.051, left=0.051, right=0.992, hspace=0.245, wspace=0.716)

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


class GlobalFluorescenceCanvas(FigureCanvas):
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
        gs = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights)

        self.low_f_mean_odmr_ax = fig.add_subplot(gs[0, 0])
        self.high_f_mean_odmr_ax = fig.add_subplot(gs[0, 1])

        self.fluo_lowF_pos_ax = fig.add_subplot(gs[1, 0])
        self.fluo_highF_pos_ax = fig.add_subplot(gs[1, 1])
        self.fluo_lowF_neg_ax = fig.add_subplot(gs[2, 0])
        self.fluo_highF_neg_ax = fig.add_subplot(gs[2, 1])
        self.cbar_ax = fig.add_subplot(gs[3, :])

        self.fluo_lowF_pos_ax.get_shared_x_axes().join(
            self.fluo_lowF_pos_ax,
            *[
                self.fluo_lowF_pos_ax,
                self.fluo_lowF_neg_ax,
                self.fluo_highF_neg_ax,
                self.fluo_highF_pos_ax,
            ],
        )
        self.fluo_lowF_pos_ax.get_shared_y_axes().join(
            self.fluo_lowF_pos_ax,
            *[
                self.fluo_lowF_pos_ax,
                self.fluo_lowF_neg_ax,
                self.fluo_highF_neg_ax,
                self.fluo_highF_pos_ax,
            ],
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
        fig.subplots_adjust(top=0.943, bottom=0.028, left=0.05, right=0.929, hspace=0.0, wspace=0.289)

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)

        self.left_top_ax = fig.add_subplot(spec[0, 0])
        self.right_top_ax = fig.add_subplot(spec[0, 1])
        self.left_bottom_ax = fig.add_subplot(spec[1, 0])
        self.right_bottom_ax = fig.add_subplot(spec[1, 1])

        self.left_top_ax.get_shared_x_axes().join(
            self.left_top_ax,
            self.left_bottom_ax,
            self.right_top_ax,
            self.right_bottom_ax,
        )
        self.left_top_ax.get_shared_y_axes().join(
            self.left_top_ax,
            self.left_bottom_ax,
            self.right_top_ax,
            self.right_bottom_ax,
        )

        self.ax = np.array(
            [
                [self.left_top_ax, self.right_top_ax],
                [self.left_bottom_ax, self.right_bottom_ax],
            ]
        )
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
