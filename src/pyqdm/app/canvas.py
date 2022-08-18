import itertools
import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PyQdmCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    @property
    def data_axes(self):
        return list(self.data.keys())

    @property
    def img_axes(self):
        return list(self.led.keys()) + list(self.laser.keys()) + \
               list(self.data.keys()) + list(self.outlier.keys()) + \
               list(self.overlay.keys())

    @property
    def odmr_axes(self):
        return list(self.odmr.keys())

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.LOG = logging.getLogger(f'pyQDM.{self.__class__.__name__}')
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PyQdmCanvas, self).__init__(self.fig)
        self.led = {}  # dict of ax : led img
        self.laser = {}  # laser is a dictionary of dictionaries
        self.data = {}  # data is a dictionary of dictionaries
        self.outlier = {}  # outlier is a dictionary of ax: outlier data
        self.overlay = {}  # dictionary of ax: overlay img objects
        self.cbar = {}  # dictionary of ax: colorbar object

        self.odmr = {}  # dictionary of ax : odmr data lines
        self.corrected = {}  # dictionary of ax : corrected data lines
        self.uncorrected = {}  # dictionary of ax : uncorrected data lines

        self.fit = {}  # dictionary of ax : odmr fit lines
        self.markers = {}  # dictionary of Line2D objects containing the markers for all axes in _is_img

    @staticmethod
    def add_cax(ax):
        divider = make_axes_locatable(ax)
        return divider.append_axes("right", size="5%", pad=0.05)

    def set_led(self):
        for a in self.led:
            a.set(xlabel='px', ylabel='px', title='Light', aspect='equal', origin='lower')

    def set_laser(self):
        for a in self.laser:
            a.set(xlabel='px', ylabel='px', title='Laser', aspect='equal', origin='lower')

    def set_data(self):
        for a in self.data:
            a.set(xlabel='px', ylabel='px', title='Data', aspect='equal', origin='lower')

    def add_light(self, light, data_dimensions):
        for ax, img in self.led.items():
            self.LOG.debug(f'Adding LED to axis {ax}')
            if img is None:
                self.led[ax] = ax.imshow(light, cmap='bone',
                                         interpolation='none', origin='lower', aspect='equal',
                                         extent=[0, data_dimensions[1], 0, data_dimensions[0]],
                                         zorder=0)

    def add_laser(self, laser, data_dimensions):
        for ax, img in self.laser.items():
            self.LOG.debug(f'Adding laser to axis {ax}')
            if img is None:
                self.laser[ax] = ax.imshow(laser, cmap='magma',
                                           interpolation='none', origin='lower', aspect='equal',
                                           extent=[0, data_dimensions[1], 0, data_dimensions[0]],
                                           zorder=0)

    def add_outlier_masks(self, outlier):
        for ax, img in self.outlier.items():
            self.LOG.debug(f'Adding outlier mask to axis {ax}')
            if img is None:
                self.outlier[ax] = ax.imshow(outlier,
                                             cmap='gist_rainbow',
                                             alpha=outlier.astype(float),
                                             vmin=0, vmax=1, interpolation='none', origin='lower', aspect='equal',
                                             zorder=2)

    def update_outlier_masks(self, outlier):
        for ax, img in self.outlier.items():
            self.LOG.debug(f'Updating outlier mask to axis {ax}')
            img.set_data(outlier)


class GlobalFluorescenceCanvas(PyQdmCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(GlobalFluorescenceCanvas, self).__init__(parent, width, height, dpi)

        self.fig.subplots_adjust(top=0.9, bottom=0.09, left=0.075,
                                 right=0.925, hspace=0.28, wspace=0.899)

        spec = self.fig.add_gridspec(ncols=6, nrows=2)

        self.left_mean_odmr_ax = self.fig.add_subplot(spec[0, :3])
        self.right_mean_odmr_ax = self.fig.add_subplot(spec[0, 3:6])

        self.led_ax = self.fig.add_subplot(spec[1, :3])
        self.laser_ax = self.fig.add_subplot(spec[1, 3:])

        self.led_ax.get_shared_x_axes().join(self.led_ax, self.laser_ax)
        self.led_ax.get_shared_y_axes().join(self.led_ax, self.laser_ax)

        self.led = {self.led_ax: None}
        self.laser = {self.laser_ax: None}
        self.odmr = {self.left_mean_odmr_ax: None,
                     self.right_mean_odmr_ax: None}
        self.cbar = {self.laser_ax: self.add_cax(self.laser_ax)}


class FittingPropertyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(
            top=0.966,
            bottom=0.06,
            left=0.056,
            right=0.985,
            hspace=0.325,
            wspace=1.0
        )

        self.fig = fig

        spec = fig.add_gridspec(ncols=6, nrows=3)

        self.main_ax = fig.add_subplot(spec[0:2, :4])
        divider = make_axes_locatable(self.main_ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        self.original_cax_locator = self.cax._axes_locator

        self.led_ax = fig.add_subplot(spec[0, 4:])
        self.led_ax.set_title('reflected light')
        self.led_ax.set_xlabel('px')
        self.led_ax.set_ylabel('px')

        self.laser_ax = fig.add_subplot(spec[1, 4:])
        self.laser_ax.set_title('laser')
        self.laser_ax.set_xlabel('px')
        self.laser_ax.set_ylabel('px')

        self.main_ax.get_shared_x_axes().join(self.main_ax, self.led_ax, self.laser_ax)
        self.main_ax.get_shared_y_axes().join(self.main_ax, self.led_ax, self.laser_ax)

        self.left_ODMR_ax = fig.add_subplot(spec[2, :3])
        self.left_ODMR_ax.set_title('low freq. ODMR')
        self.left_ODMR_ax.set_xlabel('frequency [GHz]')
        self.left_ODMR_ax.set_ylabel('contrast [a.u.]')

        self.right_ODMR_ax = fig.add_subplot(spec[2, 3:])
        self.right_ODMR_ax.set_title('high freq. ODMR')
        self.right_ODMR_ax.set_xlabel('frequency [GHz]')
        self.right_ODMR_ax.set_ylabel('contrast [a.u.]')

        self._is_spectra = [self.left_ODMR_ax, self.right_ODMR_ax]
        self._is_data = [self.main_ax]
        self._is_img = [self.main_ax, self.led_ax, self.laser_ax]
        super(FittingPropertyCanvas, self).__init__(fig)


class GlobalFluorescenceCanvasOLD(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.9, bottom=0.09, left=0.075,
                            right=0.925, hspace=0.28, wspace=0.899)

        self.fig = fig
        widths = [1, 1, 1, 1, 1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=6, nrows=2, width_ratios=widths,
                                height_ratios=heights)

        self.left_meanODMR_ax = fig.add_subplot(spec[0, :3])
        self.right_meanODMR_ax = fig.add_subplot(spec[0, 3:6])

        self.led_ax = fig.add_subplot(spec[1, :3])
        self.laser_ax = fig.add_subplot(spec[1, 3:])

        self.led_ax.get_shared_x_axes().join(self.led_ax, self.laser_ax)
        self.led_ax.get_shared_y_axes().join(self.led_ax, self.laser_ax)
        super(GlobalFluorescenceCanvas, self).__init__(fig)


class FluorescenceCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(
            top=0.963,
            bottom=0.064,
            left=0.099,
            right=0.982,
            hspace=0.355,
            wspace=0.253
        )

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1, 1, 0.1]
        spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                                height_ratios=heights)

        self.lowF_meanODMR_ax = fig.add_subplot(spec[0, 0])
        self.highF_meanODMR_ax = fig.add_subplot(spec[0, 1])

        self.fluo_lowF_pos_ax = fig.add_subplot(spec[1, 0])
        self.fluo_highF_pos_ax = fig.add_subplot(spec[1, 1])
        self.fluo_lowF_neg_ax = fig.add_subplot(spec[2, 0])
        self.fluo_highF_neg_ax = fig.add_subplot(spec[2, 1])
        self.cbar_ax = fig.add_subplot(spec[3, :])

        self.fluo_lowF_pos_ax.get_shared_x_axes().join(self.fluo_lowF_pos_ax,
                                                       *[self.fluo_lowF_pos_ax, self.fluo_lowF_neg_ax,
                                                         self.fluo_highF_neg_ax, self.fluo_highF_pos_ax])
        self.fluo_lowF_pos_ax.get_shared_y_axes().join(self.fluo_lowF_pos_ax,
                                                       *[self.fluo_lowF_pos_ax, self.fluo_lowF_neg_ax,
                                                         self.fluo_highF_neg_ax, self.fluo_highF_pos_ax])

        super(FluorescenceCanvas, self).__init__(fig)


class SimpleCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, cax=True):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(top=0.97, bottom=0.055, left=0.075,
                            right=0.925, hspace=0.28, wspace=0.23)

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

        super(SimpleCanvas, self).__init__(fig)


class QualityCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=False)
        fig.subplots_adjust(
            top=0.94,
            bottom=0.054,
            left=0.038,
            right=0.957,
            hspace=0.15,
            wspace=0.167
        )

        self.fig = fig
        widths = [1, 1]
        heights = [1, 1]
        spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                                height_ratios=heights)

        self.left_top_ax = fig.add_subplot(spec[0, 0])
        self.right_top_ax = fig.add_subplot(spec[0, 1])
        self.left_bottom_ax = fig.add_subplot(spec[1, 0])
        self.right_bottom_ax = fig.add_subplot(spec[1, 1])

        self.left_top_ax.get_shared_x_axes().join(self.left_top_ax, self.left_bottom_ax, self.right_top_ax,
                                                  self.right_bottom_ax)
        self.left_top_ax.get_shared_y_axes().join(self.left_top_ax, self.left_bottom_ax, self.right_top_ax,
                                                  self.right_bottom_ax)

        self.ax = np.array([[self.left_top_ax, self.right_top_ax], [
            self.left_bottom_ax, self.right_bottom_ax]])
        self._is_img = self.ax.flatten()
        self._is_spectra = []

        for a in self.ax.flatten():
            a.set(xlabel='px', ylabel='px')

        self.caxes = np.array([[None, None], [None, None]])
        self.original_cax_locator = np.array([[None, None], [None, None]])

        for p, f in itertools.product(range(2), range(2)):
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(self.ax[p][f])
            self.caxes[p][f] = divider.append_axes(
                "right", size="5%", pad=0.05)
            self.original_cax_locator[p][f] = self.caxes[p][f]._axes_locator

        super(QualityCanvas, self).__init__(fig)


if __name__ == '__main__':
    c = GlobalFluorescenceCanvas()
    print('c:', c.data_axes)
