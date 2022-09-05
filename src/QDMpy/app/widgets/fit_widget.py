import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from QDMpy.app.canvas import FitCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget
from QDMpy.app.widgets.quality_widget import QualityWidget
from QDMpy.app.widgets.stats_widget import StatisticsWidget
from QDMpy.utils import polyfit2d


class FitWidget(QDMWidget):
    def __init__(self, *args, **kwargs):
        canvas = FitCanvas(self, width=12, height=12, dpi=100)

        super().__init__(canvas=canvas, *args, **kwargs)

        self._add_b111_select_box(self.main_toolbar)
        self._add_subtract_box(self.main_toolbar)
        self._add_quality_button(self.main_toolbar)
        self._add_statistsics_button(self.main_toolbar)

        self.update_data()
        self.add_light()
        self.add_laser()
        self.add_odmr()

        self.quad_background = [None, None]  # for BG of ferro/induced

        self.set_main_window()
        self.update_clims()
        self.add_scalebars()
        self.canvas.draw_idle()

    def _calculate_quad_background(self):
        for i in range(2):
            if self.quad_background[i] is None or any(
                self.quad_background[i].shape != self.qdm.odmr.data_shape
            ):
                self.LOG.debug(
                    f'Calculating quad background for {["remanent", "induced"][i]} component'
                )
                x = np.arange(self.qdm.odmr.data_shape[0])
                y = np.arange(self.qdm.odmr.data_shape[1])
                kx = ky = 2
                solution = polyfit2d(x, y, self.qdm.b111[i], kx=kx, ky=ky)
                self.quad_background[i] = np.polynomial.polynomial.polygrid2d(
                    x, y, solution[0].reshape((kx + 1, ky + 1))
                )

    def add_odmr(self, mean=False):
        self.canvas.update_odmr(
            self.qdm.odmr.f_ghz,
            data=self.get_current_odmr(),
            fit=self.get_current_fit(),
            mean=self.qdm.odmr.mean_odmr if mean else None,
        )

    def update_odmr(self):
        """
        Update the marker position on the image plots.
        """
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_corrected_odmr(),
            fit=self.get_current_fit(),
        )
        self.set_ylim()
    def on_quality_clicked(self):
        if self.quality_widget is None:
            self.LOG.debug("Creating Quality Widget")
            self.quality_widget = QualityWidget(parent=self.caller)
            self.caller.quality_widget = self.quality_widget
        if self.quality_widget.isVisible():
            self.quality_widget.hide()
        else:
            self.quality_widget.show()

    def on_stat_clicked(self):
        if self.stat_widget is None:
            self.LOG.debug("Creating Statistics Widget")
            self.stat_widget = StatisticsWidget(parent=self.caller)
            self.caller.stat_widget = self.stat_widget
        if self.stat_widget.isVisible():
            self.stat_widget.hide()
        else:
            self.stat_widget.show()
    def on_subtract_median_clicked(self):
        if self.subtract_median.isChecked() and self.subtract_quad.isChecked():
            self.subtract_quad.setChecked(False)
        self.update_data()
        self.canvas.draw_idle()

    def on_subtract_quad_clicked(self):
        if self.subtract_median.isChecked() and self.subtract_quad.isChecked():
            self.subtract_median.setChecked(False)
        self._calculate_quad_background()
        self.update_data()
        self.canvas.draw_idle()

    def _add_quality_button(self, toolbar):
        self.quality_widget = None
        self.qualityButton = QPushButton("Quality")
        self.qualityButton.clicked.connect(self.on_quality_clicked)
        toolbar.addWidget(self.qualityButton)

    def _add_statistsics_button(self, toolbar):
        self.stat_widget = None
        self.stat_button = QPushButton("Statistics")
        self.stat_button.clicked.connect(self.on_stat_clicked)
        toolbar.addWidget(self.stat_button)

    def _add_subtract_box(self, toolbar):
        subtract_widget = QWidget()
        bg_check_box = QHBoxLayout()
        subtract_label = QLabel("Subtract: ")
        self.subtract_median = QCheckBox("median")
        self.subtract_median.stateChanged.connect(self.on_subtract_median_clicked)
        self.subtract_quad = QCheckBox("quadratic")
        self.subtract_quad.stateChanged.connect(self.on_subtract_quad_clicked)
        bg_check_box.addWidget(subtract_label)
        bg_check_box.addWidget(self.subtract_median)
        bg_check_box.addWidget(self.subtract_quad)
        subtract_widget.setLayout(bg_check_box)
        toolbar.addWidget(subtract_widget)
