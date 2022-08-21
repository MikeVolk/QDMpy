import logging

import matplotlib
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
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
    def __init__(self, qdm_instance, *args, **kwargs):
        canvas = GlobalFluorescenceCanvas()
        super().__init__(canvas=canvas, qdm_instance=qdm_instance, *args, **kwargs)
        self.setWindowTitle("Global Fluorescence")
        self.gf_label = QLabel(f"Global Fluorescence: {self.qdm.global_factor:.2f}")
        self.gfSlider = QSlider()
        self.gfSlider.setValue(self.caller.gf_select.value())
        self.gfSlider.setRange(0, 100)
        self.gfSlider.setOrientation(Qt.Horizontal)
        self.gfSlider.valueChanged.connect(self.on_slider_value_changed)
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.apply_global_factor)

        self.canvas.add_light(self.qdm.light, self.qdm.data_shape)
        self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)
        self.canvas.add_scalebars(self.qdm.pixel_size)
        self.update_marker()

        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()
        self.canvas.add_odmr(self.qdm.odmr.f_ghz, data=pixel_spectra,
                             uncorrected=uncorrected, corrected=corrected)
        self.canvas.update_odmr_lims()

        # finish main layout
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.gf_label)
        horizontal_layout.addWidget(self.gfSlider)
        horizontal_layout.addWidget(self.applyButton)
        self.mainVerticalLayout.addLayout(horizontal_layout)
        self.set_main_window()

    def get_pixel_data(self):
        gf_factor = self.gfSlider.value() / 100
        new_correct = self.qdm.odmr.get_gf_correction(gf=gf_factor)
        old_correct = self.qdm.odmr.get_gf_correction(gf=self.qdm.odmr.global_factor)
        pixel_spectra = self.qdm.odmr.data[:, :, self._current_idx].copy()  # possibly already corrected
        uncorrected = pixel_spectra + old_correct
        corrected = uncorrected - new_correct
        mn = np.min([np.min(pixel_spectra), np.min(corrected), np.min(uncorrected)]) * 0.998
        mx = np.max([np.max(pixel_spectra), np.max(corrected), np.max(uncorrected)]) * 1.002
        return pixel_spectra, uncorrected, corrected, mn, mx

    def on_slider_value_changed(self):
        print(self.canvas.odmr)
        self.gf_label.setText(f"Global Fluorescence: {self.gfSlider.value() / 100:.2f}")
        pixel_spectra, uncorrected, corrected, mn, mx = self.get_pixel_data()
        self.canvas.update_odmr(data=uncorrected, corrected=corrected)
        self.canvas.update_odmr_lims()

    def apply_global_factor(self):
        self.LOG.debug(f"applying global factor {self.gfSlider.value() / 100:.2f}")
        self.qdm.odmr.correct_glob_fluorescence(self.gfSlider.value() / 100)
        gf_applied_window(self.gfSlider.value() / 100)
        self.caller.gf_select.setValue(self.gfSlider.value() / 100)
        self.close()
