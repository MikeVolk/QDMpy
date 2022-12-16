import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event, MouseButton
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from numpy.typing import NDArray
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QScreen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import matplotlib.transforms as transforms
from pypole.convert import dim2xyz, xyz2dim

from QDMpy._core import models
from QDMpy._core.convert import b2shift, freq2b
from QDMpy._core.models import esr14n, esr15n
from QDMpy.app.assets.GuiElements import LabeledDoubleSpinBox
from QDMpy.app.canvas import FrequencyToolCanvas
from QDMpy.app.models import Pix
from QDMpy.utils import rms

ZFS = 2.87

MUT = "µT"
B111 = "B$_{111}$"
FRES = "B[111] [µT]"


class FrequencySelectWidget(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    LOG = logging.getLogger(__name__)

    def get_label_slider(self, value, min, max, callback, decimals=0, step=1):
        hlayout = QHBoxLayout()
        label, spinbox = LabeledDoubleSpinBox(f"{MUT}", value, decimals, step, min, max, callback)
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setRange(min, max)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        spinbox.valueChanged.connect(callback)

        slider.valueChanged.connect(lambda x: spinbox.setValue(slider.value()))
        spinbox.valueChanged.connect(lambda x: slider.setValue(x))
        hlayout.addWidget(spinbox)
        hlayout.addWidget(label)
        hlayout.addWidget(slider)
        return hlayout, slider

    def __init__(
        self,
        clim_select=True,
        pixel_select=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.caller = self.parent()

        self.setContentsMargins(0, 0, 0, 0)
        self.setGeometry(0, 0, 800, 600)

        self.canvas = FrequencyToolCanvas()

        # layout
        self.mainVerticalLayout = QVBoxLayout()
        vlayout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.add_bias_field_setter(top_layout)

        separator = QLabel("   |   ")
        range_label, self.range_box = LabeledDoubleSpinBox(
            f"Measurement range [{MUT}]", 500, 0, 10, 0, 10000, self.update_plot
        )
        top_layout.addWidget(separator)
        top_layout.addWidget(range_label)
        top_layout.addWidget(self.range_box)

        vlayout.addLayout(top_layout)

        # nv settings box
        box = QGroupBox("nv-settings")
        self.nv_settings_box = QGridLayout()
        # HEADER ROW
        self.nv_settings_box.addWidget(QLabel(f"{FRES}"), 0, 1)
        self.nv_settings_box.addWidget(QLabel("width [MHz]"), 0, 2)
        self.nv_settings_box.addWidget(QLabel("contrast [%]"), 0, 3)

        self.b111 = [
            self.get_label_slider(0, -1000, 1000, self.update_plot, step=10),
            self.get_label_slider(0, -1000, 1000, self.update_plot, step=10),
            self.get_label_slider(0, -1000, 1000, self.update_plot, step=10),
            self.get_label_slider(0, -1000, 1000, self.update_plot, step=10),
        ]
        for i, b111 in enumerate(self.b111):
            self.nv_settings_box.addLayout(b111[0], i + 1, 1)

        self.widths = [
            LabeledDoubleSpinBox("W$_1$", 1, 1, 0.1, 0, 1000, self.update_plot),
            LabeledDoubleSpinBox("W$_2$", 1, 1, 0.1, 0, 1000, self.update_plot),
            LabeledDoubleSpinBox("W$_3$", 1, 1, 0.1, 0, 1000, self.update_plot),
            LabeledDoubleSpinBox("W$_4$", 1, 1, 0.1, 0, 1000, self.update_plot),
        ]  # in MHz

        for i, width in enumerate(self.widths):
            self.nv_settings_box.addWidget(width[1], i + 1, 2)

        # add contrast Spinboxes to the group box
        self.contrasts = [
            LabeledDoubleSpinBox("C$_1$", 1, 1, 0.1, 0, 100, self.update_plot),
            LabeledDoubleSpinBox("C$_2$", 1, 1, 0.1, 0, 100, self.update_plot),
            LabeledDoubleSpinBox("C$_3$", 1, 1, 0.1, 0, 100, self.update_plot),
            LabeledDoubleSpinBox("C$_4$", 1, 1, 0.1, 0, 100, self.update_plot),
        ]

        for i, contrast in enumerate(self.contrasts):
            self.nv_settings_box.addWidget(contrast[1], i + 1, 3)

        vlayout.addLayout(self.nv_settings_box)
        box.setLayout(vlayout)

        self.mainVerticalLayout.addWidget(box)
        self.mainVerticalLayout.addWidget(self.canvas)

        f = np.linspace(ZFS - 0.1, ZFS + 0.1, 1000)
        self.frequencies = [f, f]

        self.nv_lines = {1: [None, None], 2: [None, None], 3: [None, None], 4: [None, None]}
        self.nv_settings = {
            1: [[ZFS, 0.002, 0.02, 0.02, 0], [ZFS, 0.002, 0.02, 0.02, 0]],
            2: [[ZFS, 0.002, 0.02, 0.02, 0], [ZFS, 0.002, 0.02, 0.02, 0]],
            3: [[ZFS, 0.002, 0.02, 0.02, 0], [ZFS, 0.002, 0.02, 0.02, 0]],
            4: [[ZFS, 0.002, 0.02, 0.02, 0], [ZFS, 0.002, 0.02, 0.02, 0]],
        }
        self.nv_data = {1: [None, None], 2: [None, None], 3: [None, None], 4: [None, None]}
        self.nv_text = {1: [None, None], 2: [None, None], 3: [None, None], 4: [None, None]}
        self.nv_directions = {1: [1, 1, 1], 2: [1, -1, -1], 3: [-1, 1, -1], 4: [-1, -1, 1]}
        self.nv_span = [None, None]
        self.nv_span_text = [None, None]

        self.nv_sum = None
        self.nv_sum_line = None

        self.nv_labels = {1: "NV(1)", 2: "NV(2)", 3: "NV(3)", 4: "NV(4)"}

        for i, label in self.nv_labels.items():
            self.nv_settings_box.addWidget(QLabel(label), i, 0)

        self.nv_colors = {1: "C00", 2: "C01", 3: "C02", 4: "C03"}

        # update parameter for bias field
        self.update_parameter()
        self.update_plot()
        self.update_xlim()
        plt.tight_layout()
        self.set_main_window()

    def add_bias_field_setter(self, top_layout):
        bias_field_label, self.bias_field_box = LabeledDoubleSpinBox(
            f"Strength [{MUT}]:", 900, 0, 10, 0, 10000, self.update_plot
        )
        bias_dec_label, self.bias_dec_box = LabeledDoubleSpinBox(f"Dec: ", 0, 1, 1, 0, 360, self.update_plot)
        bias_inc_label, self.bias_inc_box = LabeledDoubleSpinBox(f"Inc: ", 35.3, 1, 1, -90, 90, self.update_plot)
        top_layout.addWidget(QLabel("Bias field: "))
        top_layout.addWidget(bias_dec_label)
        top_layout.addWidget(self.bias_dec_box)
        top_layout.addWidget(bias_inc_label)
        top_layout.addWidget(self.bias_inc_box)
        top_layout.addWidget(bias_field_label)
        top_layout.addWidget(self.bias_field_box)

    def set_main_window(self):
        """
        Sets the final widget to the layout.
        This is separate so you can add additional things to the toplayout ....
        """
        central_widget = QWidget()
        central_widget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(central_widget)

    def set_bias_field(self, value):
        self.caller.bias_field = value
        self.update_plot()

    def update_xlim(self):
        self.LOG.debug("Updating X limits")
        xmin = np.min(self.frequencies[0])
        xmax = np.max(self.frequencies[0])
        self.canvas.ax.set_xlim(xmin, xmax)
        self.canvas.axb.set_xlim(
            freq2b(xmin, in_unit="GHz", out_unit="milliT"), freq2b(xmax, in_unit="GHz", out_unit="milliT")
        )

    def update_ylim(self):
        self.LOG.debug("Updating Y limits")
        mn = np.min(self.nv_sum_line.get_ydata())
        mx = np.max(self.nv_sum_line.get_ydata())
        self.canvas.ax.set_ylim(mn * 0.999, mx * 1.001)

    def update_plot(self):
        self.LOG.debug("Updating plot")
        self.LOG.debug("-" * 80)
        self.update_parameter()
        self.update_measure_range()
        self.update_data()
        self.update_nv_lines()
        self.update_sum_line()
        self.update_nv_text()
        self.update_range_text()
        self.update_ylim()
        self.canvas.draw()

    def update_nv_lines(self):
        self.LOG.debug("Updating NV lines")
        for nv in self.nv_lines:
            for i, line in enumerate(self.nv_lines[nv]):
                if line is None:
                    l = self.canvas.ax.plot(
                        self.frequencies[i],
                        self.nv_data[nv][i],
                        label=self.nv_labels[nv],
                        color=self.nv_colors[nv],
                        alpha=0.3,
                        ls="--",
                        lw=1,
                    )
                    self.nv_lines[nv][i] = l[0]
                else:
                    line.set_data(self.frequencies[i], self.nv_data[nv][i])

    def update_sum_line(self):
        self.LOG.debug("Updating sum line")
        if self.nv_sum_line is None:
            l = self.canvas.ax.plot(self.frequencies[0], self.nv_sum, label="Sum", color="k", lw=1)
            self.nv_sum_line = l[0]
        else:
            self.nv_sum_line.set_data(self.frequencies[0], self.nv_sum)

    def update_data(self):
        self.LOG.debug(
            f"Updating data to:\n {self.nv_settings[1]} \n {self.nv_settings[2]} \n {self.nv_settings[3]} \n {self.nv_settings[4]}"
        )
        self.nv_data = {
            nv: [esr15n(self.frequencies[i], self.nv_settings[nv][i])[0] for i in range(2)] for nv in self.nv_settings
        }
        self.nv_sum = np.sum([self.nv_data[nv][i] for nv in self.nv_data for i in range(2)], axis=0)
        self.nv_sum -= 7

    def update_parameter(self):
        self.LOG.debug("updating parameters")
        b_bias = dim2xyz([self.bias_dec_box.value(), self.bias_inc_box.value(), self.bias_field_box.value()])

        for nv in self.nv_settings:
            if nv == 1:
                bias = b2shift(self.bias_field_box.value(), in_unit="microT")
            else:
                bias = 0

            for i, line in enumerate(self.nv_settings[nv]):
                self.nv_settings[nv][i][0] = ZFS + [1, -1][i] * (
                    b2shift(self.b111[nv - 1][1].value(), in_unit="microT") + bias
                )
                self.nv_settings[nv][i][1] = self.widths[nv - 1][1].value() / 1000  # to GHz
                self.nv_settings[nv][i][2] = self.contrasts[nv - 1][1].value() / 100
                self.nv_settings[nv][i][3] = self.contrasts[nv - 1][1].value() / 100

    def update_nv_text(self):
        self.LOG.debug("updating text")
        ax = self.canvas.ax
        trans = transforms.blended_transform_factory(self.canvas.ax.transData, self.canvas.ax.transAxes)

        for nv in self.nv_settings:
            for i, line in enumerate(self.nv_settings[nv]):
                if self.nv_text[nv][i] is None:
                    txt = ax.text(
                        self.nv_settings[nv][i][0],
                        1.2,
                        f"NV$_{nv}$\n |",
                        va="center",
                        ha="center",
                        transform=trans,
                        bbox=dict(facecolor="w", alpha=0.5, edgecolor="none", pad=0),
                        color=self.nv_colors[nv],
                    )
                    self.nv_text[nv][i] = txt
                else:
                    self.nv_text[nv][i].set_x(self.nv_settings[nv][i][0])

    def projection(self, bias_field, source_field):
        """
        Project the bias field and source field on each nv axis
        """
        fields = np.zeros(len(self.nv_directions))
        for i, nv in enumerate(self.nv_directions):
            f =  np.dot(bias_field + source_field, nv) / np.linalg.norm(nv)
            fields[i] = np.linalg.norm(f)
        return fields
    def update_measure_range(self):
        bias_shift = b2shift(self.bias_field_box.value(), in_unit="microT")
        b_shift = b2shift(self.range_box.value(), in_unit="microT")

        for i, span in enumerate(self.nv_span):
            if span is None:
                self.nv_span[i] = self.canvas.ax.axvspan(
                    ZFS - b_shift + bias_shift * [1, -1][i],
                    ZFS + b_shift + bias_shift * [1, -1][i],
                    color="g",
                    alpha=0.1,
                    label="NV range",
                )
            else:
                xy = [
                    [ZFS - b_shift + bias_shift * [1, -1][i], 0],
                    [ZFS - b_shift + bias_shift * [1, -1][i], 1],
                    [ZFS + b_shift + bias_shift * [1, -1][i], 1],
                    [ZFS + b_shift + bias_shift * [1, -1][i], 0],
                ]
                span.set_xy(xy)

    def update_range_text(self):
        bias_shift = b2shift(self.bias_field_box.value(), in_unit="microT")
        b_shift = b2shift(self.range_box.value(), in_unit="microT")

        for i, txt in enumerate(self.nv_span_text):
            if txt is not None:
                txt.remove()

            trans = transforms.blended_transform_factory(self.canvas.ax.transData, self.canvas.ax.transAxes)
            self.nv_span_text[i] = self.canvas.ax.text(
                ZFS + bias_shift * [1, -1][i],
                0.1,
                f"{ZFS + bias_shift* [1, -1][i] - b_shift:.4}-{ZFS + bias_shift* [1, -1][i] + b_shift:.4} GHz",
                va="center",
                ha="center",
                transform=trans,
                bbox=dict(facecolor="w", alpha=0.5, edgecolor="none", pad=0),
                color="k",
            )


def main(**kwargs):
    app = QApplication()
    screen = app.primaryScreen()
    mainwindow = FrequencySelectWidget()
    mainwindow.show()

    center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
    geo = mainwindow.frameGeometry()
    geo.moveCenter(center)
    mainwindow.move(geo.topLeft())

    app.exec()


if __name__ == "__main__":
    main()
