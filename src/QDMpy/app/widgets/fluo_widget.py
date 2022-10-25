import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib_scalebar.scalebar import ScaleBar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from QDMpy.app.canvas import FluoImgCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget

matplotlib.rcParams.update(
    {  # 'font.size': 8,
        # 'axes.labelsize': 8,
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    }
)
SCALEBAR_LOC = "lower left"


class FluoWidget(QDMWidget):
    def __init__(self, *args, **kwargs) -> None:
        canvas = FluoImgCanvas()
        super().__init__(canvas=canvas, *args, **kwargs)

        # add the lines in the ODMR plot
        self.lowF_line = None
        self.highF_line = None

        self.setWindowTitle("Global Fluorescence")

        slider_widget = QWidget()
        h_layout = QHBoxLayout()
        self.index_label = QLabel(f"Freq. index ({0:d}): ")
        self.index_slider = QSlider()
        self.index_slider.setOrientation(Qt.Horizontal)
        self.index_slider.setMinimum(0)
        self.index_slider.setMaximum(self.qdm.odmr.n_freqs - 1)
        self.index_slider.setSingleStep(1)
        self.index_slider.valueChanged.connect(self.update_fluorescence)
        self.freq_label = QLabel()

        h_layout.addWidget(self.index_label)
        h_layout.addWidget(self.index_slider)
        h_layout.addWidget(self.freq_label)
        slider_widget.setLayout(h_layout)
        slider_widget.setContentsMargins(0, 0, 0, 0)
        slider_widget.setMaximumHeight(30)
        self.mainVerticalLayout.addWidget(slider_widget)
        self.set_main_window()

        self.add_odmr(mean=True)
        self.update_fluorescence(0)
        self.add_scalebars()
        self.update_marker()
        self.update_clims()
        self.canvas.draw()
        self.resize(700, 700)

    def update_fluorescence(self, value):
        # self.clear_axes()
        self.canvas.add_fluorescence(
            fluorescence=self.qdm.odmr.data[:, :, :, self.index_slider.value()].reshape(
                self.qdm.odmr.n_pol, self.qdm.odmr.n_frange, *self.qdm.odmr.data_shape
            )
        )

        self.lowF_line = self.update_line(
            self.canvas.low_f_mean_odmr_ax, 0, self.lowF_line
        )
        self.highF_line = self.update_line(
            self.canvas.high_f_mean_odmr_ax, 1, self.highF_line
        )

        self.index_label.setText(f"Freq. index ({value}): ")
        self.freq_label.setText(
            f"| {self.qdm.odmr.f_ghz[0,value]:5.4f}, {self.qdm.odmr.f_ghz[1,value]:5.4f} [GHz]"
        )

        self.update_clims()
        self.canvas.draw()

    def update_line(self, ax, frange, line=None):
        if line is None:
            line = ax.axvline(
                self.qdm.odmr.f_ghz[frange, self.index_slider.value()],
                color="k",
                alpha=0.5,
                zorder=0,
                ls=":",
            )
        else:
            line.set_xdata(self.qdm.odmr.f_ghz[frange, self.index_slider.value()])
        return line
