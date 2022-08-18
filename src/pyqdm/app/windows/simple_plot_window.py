import logging

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout
)
from pyqdm_plot_window import pyqdmWindow
from canvas import SimpleCanvas
from matplotlib import colors
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar


class SimplePlotWindow(pyqdmWindow):
    def __init__(self, title, cmap, cbar, caller, *args, **kwargs):
        canvas = SimpleCanvas(self, width=12, height=12, dpi=100, cax=cbar)
        super(SimplePlotWindow, self).__init__(caller, canvas, *args, **kwargs)

        self.cmap = cmap
        self.title = title
        self.cbar = None
        self.ax = self.canvas.ax
        self.cax = self.canvas.cax

        self.setWindowTitle(f'{title}')

        self._includes_fits = False
        self._pixel_ax = []
        self._init_lines()

        mainWidget = QWidget()
        mainWidget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(mainWidget)
        self.resize(1080, 720)
        self.update_marker()

    def update_img_plots(self):
        img = self.ax.images[0]
        d = img.get_array()
        vmin, vmax = np.min(d), np.max(d)

        if self.fixClimCheckBox.isChecked():
            vmin, vmax = np.percentile(d, [100 - self.cLimSelector.value(), self.cLimSelector.value()])

        img.set(norm=colors.Normalize(vmin=vmin, vmax=vmax))

        if self.cbar is not None:
            self.canvas.cax.clear()
            self.canvas.cax.set_axes_locator(self.canvas.original_cax_locator)

            im_ratio = d.shape[0] / d.shape[1]
            self.cbar = plt.colorbar(img, cax=self.canvas.cax,  # fraction=0.047 * im_ratio, pad=0.01,
                                     extend='both' if self.fixClimCheckBox.isChecked() else 'neither',
                                     label='B$_{111}$ [$\mu$T]')

        self.canvas.draw()


class SimplePlotWindowOLD(QMainWindow):
    LOG = logging.getLogger('pyqdm' + __name__)

    def on_press(self, event):
        if event.button == MouseButton.LEFT:
            x, y = np.round([event.xdata, event.ydata]).astype(int)
            self.LOG.debug(f'clicked on: x, y = {x}, {y}')

    def __init__(self, caller, plot_data=None, title='', pixelsize=1e-6, cmap='bone', cbar=None,
                 *args, **kwargs):
        super(SimplePlotWindow, self).__init__(*args, **kwargs)
        self.LOG.debug('__init__')
        self.caller = caller
        self.plot_data = plot_data
        self.pixelsize = pixelsize
        self.cmap = cmap
        self.title = title
        self.cbar = cbar

        self.setWindowTitle(f'{title}')
        self.canvas = QualityCanvas(self, width=12, height=6, dpi=100)
        cid = self.canvas.mpl_connect('button_press_event', self.on_press)

        self.toolbar = NavigationToolbar(self.canvas, self)

        verticalLayout = QVBoxLayout()
        verticalLayout.addWidget(self.toolbar)
        verticalLayout.addWidget(self.canvas)

        mainWidget = QWidget()
        mainWidget.setLayout(verticalLayout)
        self.setCentralWidget(mainWidget)
        self.resize(1080, 720)
        self.init_plot()
        self.show()

    def init_plot(self) -> None:
        self.canvas.ax.clear()
        vmin, vmax = np.percentile(self.plot_data, [2, 98])
        im = self.canvas.ax.imshow(self.plot_data, cmap=self.cmap, aspect='equal', vmin=vmin, vmax=vmax)

        # Create scale bar
        scalebar = ScaleBar(self.caller.pixelsize, "m", length_fraction=0.25,
                            location="lower left")
        self.canvas.ax.add_artist(scalebar)

        self.canvas.ax.set_xlabel('px')
        self.canvas.ax.set_ylabel('px')
        self.canvas.ax.set_title(self.title, fontsize=12)
        self.canvas.ax.grid(False)

        if self.cbar is not None or self.cbar is not False:
            self.cbar.set_axes_locator(self.canvas.original_cax_locator)
            self.cbar = plt.colorbar(im, cax=self.canvas.cax, extend='both', pad=0.01, label='counts')

        self.canvas.draw()
        self.LOG.debug('update')
