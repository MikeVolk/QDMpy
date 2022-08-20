import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PySide6.QtWidgets import QWidget

from pyqdm.app.canvas import SimpleCanvas
from pyqdm.app.windows.pyqdm_plot_window import PyQdmWindow


class SimplePlotWindow(PyQdmWindow):
    def __init__(self, title, cmap, cbar, caller, *args, **kwargs):
        canvas = SimpleCanvas(self, width=12, height=12, dpi=100, cax=cbar)
        super().__init__(caller, canvas, *args, **kwargs)

        self.cmap = cmap
        self.title = title
        self.cbar = None
        self.ax = self.canvas.ax
        self.cax = self.canvas.cbar

        self.setWindowTitle(f"{title}")

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

        if self.fix_clim_check_box.isChecked():
            vmin, vmax = np.percentile(d, [100 - self.clims_selector.value(), self.clims_selector.value()])

        img.set(norm=colors.Normalize(vmin=vmin, vmax=vmax))

        if self.cbar is not None:
            self.canvas.cbar.clear()
            self.canvas.cbar.set_axes_locator(self.canvas.original_cax_locator)

            self.cbar = plt.colorbar(
                img,
                cax=self.canvas.cax,
                extend="both" if self.fix_clim_check_box.isChecked() else "neither",
                label=r"B$_{111}$ [$\mu$T]",
            )

        self.canvas.draw()
