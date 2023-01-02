import logging
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from pypole.convert import dim2xyz
from PySide6.QtCore import Qt
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from QDMpy._core.convert import b2shift, freq2b, project
from QDMpy._core.models import esr15n
from QDMpy.app.canvas import FrequencyToolCanvas
from QDMpy._core.convert import b2shift, freq2b, project
from QDMpy import utils

ZFS = 2.87

MUT = "µT"
B111 = "B$_{111}$"
FRES = "B[111] [µT]"


class FrequencySelectWidget(QMainWindow):
    """
    Window for checking the global fluorescence correction.
    """

    LOG = logging.getLogger(f"QDMpy.{__name__}")
    LOG.setLevel(logging.DEBUG)

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

        top_layout = QVBoxLayout()
        top_grid = QGridLayout()
        top_layout.addLayout(top_grid)

        self.add_measurement_range_selector(top_grid, 1)
        self.add_visibility_boxes(top_grid, 4)
        self.add_row_header(top_grid)
        self.add_bias_field_setter(top_grid, 1)
        self.add_source_field_setter(top_grid, 4)
        self.add_spacer(top_grid, 8)
        self.add_nv_setter(top_grid, 9)

        self.mainVerticalLayout.addLayout(top_layout)
        self.mainVerticalLayout.addWidget(self.canvas)

        f = np.linspace(ZFS - 0.1, ZFS + 0.1, 1000)
        self.frequencies = [f, f]

        self.nvs = {
            0: {
                "line": None,
                "field": 0.0,
                "resonance": [ZFS, ZFS],
                "width": 0.002,
                "contrast": 0.02,
                "color": "C00",
                "label": "NV1",
                "data": [None, None],
                "text": [None, None],
                "direction": [1, -1, -1],
            },
            1: {
                "line": None,
                "field": 0.0,
                "resonance": [ZFS, ZFS],
                "width": 0.002,
                "contrast": 0.02,
                "color": "C01",
                "label": "NV2",
                "data": [None, None],
                "text": [None, None],
                "direction": [-1, -1, 1],
            },
            2: {
                "line": None,
                "field": 0.0,
                "resonance": [ZFS, ZFS],
                "width": 0.002,
                "contrast": 0.02,
                "color": "C02",
                "label": "NV3",
                "data": [None, None],
                "text": [None, None],
                "direction": [1, 1, 1],
            },
            3: {
                "line": None,
                "field": 0.0,
                "resonance": [ZFS, ZFS],
                "width": 0.002,
                "contrast": 0.02,
                "color": "C03",
                "label": "NV4",
                "data": [None, None],
                "text": [None, None],
                "direction": [-1, 1, -1],
            },
        }

        self.nv_span = [None, None]
        self.danger_range = 0.998
        self.danger_span = [None, None, None]
        self.nv_span_text = [None, None]

        self.nv_sum = None
        self.nv_sum_line = None

        # update parameter for bias field
        self.update_field_shift()
        self.update_widths()
        self.update_contrasts()
        self.update_field_shift_labels()
        self.update_plot()
        self.update_xlim()
        self.canvas.ax.legend()
        plt.tight_layout()
        self.set_main_window()

    def update_nv_dict(self, nv, key, value):
        self.LOG.debug(f"update nv {nv}: {key} to {value}")
        self.nvs[nv][key] = value

    def add_spacer(self, layout, col):
        spacer = QLabel("")
        spacer.setFixedWidth(50)
        spacer.setAlignment(Qt.AlignCenter)
        layout.addWidget(spacer, 0, col)

    def add_nv_setter(self, layout, col):
        # row header
        for i in range(4):
            label = QLabel(f"NV{i + 1}: ")
            label.setFixedWidth(50)
            label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
            layout.addWidget(label, i + 1, col)

        # column header
        for i, key in enumerate(["f1 [MHz]", "f2 [MHz]", f"B [{MUT}]", "width [MHz]", "c [%]", "show"]):
            label = QLabel(key)
            label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
            layout.addWidget(label, 0, col + i + 1)

        # spinboxes
        self.b111 = [
            [QLabel("0"), QLabel("0"), QLabel("0")],
            [QLabel("0"), QLabel("0"), QLabel("0")],
            [QLabel("0"), QLabel("0"), QLabel("0")],
            [QLabel("0"), QLabel("0"), QLabel("0")],
        ]
        for i, b111 in enumerate(self.b111):
            for j, b in enumerate(b111):
                b.setFixedWidth(75)
                b.setAlignment(Qt.AlignCenter | Qt.AlignRight)
                b.setContentsMargins(0, 0, 10, 0)
                layout.addWidget(b, i + 1, col + j + 1)

        self.width_boxes = [
            DoubleSpinBox(1, 1, 0.1, 0, 1000, [self.update_widths, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 1000, [self.update_widths, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 1000, [self.update_widths, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 1000, [self.update_widths, self.update_nv_data, self.update_plot]),
        ]  # in MHz

        for i, width in enumerate(self.width_boxes):
            layout.addWidget(width, i + 1, col + 4)

        # add contrast Spinboxes to the group box
        self.contrast_boxes = [
            DoubleSpinBox(1, 1, 0.1, 0, 100, [self.update_contrasts, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 100, [self.update_contrasts, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 100, [self.update_contrasts, self.update_nv_data, self.update_plot]),
            DoubleSpinBox(1, 1, 0.1, 0, 100, [self.update_contrasts, self.update_nv_data, self.update_plot]),
        ]

        for i, contrast in enumerate(self.contrast_boxes):
            layout.addWidget(contrast, i + 1, col + 5)

        # add contrast Show to the group box
        self.visibility_boxes = [
            CheckBox("", self.update_visibility, False),
            CheckBox("", self.update_visibility, False),
            CheckBox("", self.update_visibility, False),
            CheckBox("", self.update_visibility, False),
        ]

        for i, box in enumerate(self.visibility_boxes):
            layout.addWidget(box, i + 1, col + 6)

    def add_measurement_range_selector(self, layout, col):
        # add the measurement range selector
        range_layout, self.range_box = BoxSlider(500, 0, 1, 0, 5000, self.update_plot)
        layout.addLayout(range_layout, 0, col + 1)

    def add_visibility_boxes(self, layout, col):
        # add the measurement range selector
        self.range_visibility = CheckBox(
            "", [self.update_range_text_visibilities, self.update_range_visibility], False
        )
        self.measure_range_visibility = CheckBox("NV1", self.update_range_visibility, False)
        self.danger_range_visibility = CheckBox("NV2-NV4", self.update_range_visibility, True)

        layout.addWidget(self.range_visibility, 0, col+1)
        layout.addWidget(self.measure_range_visibility, 0, col + 2)
        layout.addWidget(self.danger_range_visibility, 0, col + 3)

    def add_row_header(self, layout):
        range_label = QLabel(f"Range [{MUT}]")
        range_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        dec_label = QLabel("Dec [°]")
        dec_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        inc_label = QLabel("Inc [°]")
        inc_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        field_label = QLabel(f"Field [{MUT}]")
        field_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        # row header
        layout.addWidget(range_label, 0, 0)
        layout.addWidget(dec_label, 2, 0)
        layout.addWidget(inc_label, 3, 0)
        layout.addWidget(field_label, 4, 0)

    def add_source_field_setter(self, layout, col):
        """add a box to set the source field to the top layout

        Parameters
        ----------
        layout : QHBoxLayout
            layout to add the box to

        """
        # column header
        label = QLabel("Source field: ")

        source_field_layout, self.source_field_slider = BoxSlider(
            0, 0, 10, -5000, 5000, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )
        source_dec_layout, self.source_dec_slider = BoxSlider(
            0, 2, 1, 0, 360, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )
        source_inc_layout, self.source_inc_slider = BoxSlider(
            0, 2, 1, -90, 90, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )

        # widgets
        for i, box in enumerate([label, source_dec_layout, source_inc_layout, source_field_layout]):
            if i == 0:
                layout.addWidget(box, i + 1, col + 1, 1, 3, alignment=Qt.AlignLeft)
            else:
                layout.addLayout(box, i + 1, col + 1, 1, 3, alignment=Qt.AlignLeft)

    def add_bias_field_setter(self, layout, col):
        # column header
        label = QLabel("Bias field:")

        bias_field_layout, self.bias_field_slider = BoxSlider(
            900, 0, 10, 0, 10000, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )
        bias_dec_layout, self.bias_dec_slider = BoxSlider(
            45, 2, 1, 0, 360, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )
        bias_inc_layout, self.bias_inc_slider = BoxSlider(
            35.3, 2, 1, -90, 90, [self.update_field_shift, self.update_field_shift_labels, self.update_plot]
        )

        # widgets
        for i, box in enumerate([label, bias_dec_layout, bias_inc_layout, bias_field_layout]):
            if i == 0:
                layout.addWidget(box, i + 1, col + 1, 1, 3, alignment=Qt.AlignLeft)
            else:
                layout.addLayout(box, i + 1, col + 1, 1, 3, alignment=Qt.AlignLeft)

    def set_main_window(self):
        """
        Sets the final widget to the layout.
        This is separate so you can add additional things to the toplayout ....
        """
        central_widget = QWidget()
        central_widget.setLayout(self.mainVerticalLayout)
        self.setCentralWidget(central_widget)

    def update_field_shift(self):
        b_bias = dim2xyz(
            np.array(
                [
                    self.bias_dec_slider.value(),
                    self.bias_inc_slider.value(),
                    self.bias_field_slider.value(),
                ]
            )
        )
        b_source = dim2xyz(
            np.array(
                [
                    self.source_dec_slider.value(),
                    self.source_inc_slider.value(),
                    self.source_field_slider.value(),
                ]
            )
        )
        field = b_bias + b_source

        for nv in self.nvs:
            projected_field = project(field, self.nvs[nv]["direction"])[0]
            self.update_nv_dict(nv, "field", projected_field)
            self.update_nv_dict(
                nv,
                "resonance",
                [
                    (ZFS - b2shift(projected_field, in_unit="microT", out_unit="GHz")),
                    (ZFS + b2shift(projected_field, in_unit="microT", out_unit="GHz")),
                ],
            )

    def update_field_shift_labels(self):
        self.LOG.debug("updating field shift labels")
        for nv in self.nvs:
            self.b111[nv][0].setText(f"{self.nvs[nv]['resonance'][0] * 1000:.0f}")
            self.b111[nv][1].setText(f"{self.nvs[nv]['resonance'][1] * 1000:.0f}")
            self.b111[nv][2].setText(f"{self.nvs[nv]['field']:.2f}")

    def update_widths(self):
        for nv in self.nvs:
            self.update_nv_dict(nv, "width", self.width_boxes[nv].value() / 1000)

    def update_contrasts(self):
        for nv in self.nvs:
            self.update_nv_dict(nv, "contrast", self.contrast_boxes[nv].value() / 100)

    def update_visibility(self):
        for nv in self.nvs:
            line = self.nvs[nv]["line"]
            self.LOG.debug(f"updating visibility of {line}")
            line.set_visible(self.visibility_boxes[nv].isChecked())
        self.canvas.draw_idle()

    def update_range_text_visibilities(self):
        for txt in self.nv_span_text:
            if txt is not None:
                txt.set_visible(self.range_visibility.isChecked())

    def update_range_visibility(self):
        for i, span in enumerate(self.nv_span):
            if span is None:
                continue
            self.LOG.debug(f"updating visibility of {span}")
            span.set_visible(self.range_visibility.isChecked())

        self.danger_span[1].set_visible(self.danger_range_visibility.isChecked())
        self.danger_span[2].set_visible(self.measure_range_visibility.isChecked())

        self.canvas.draw_idle()

    def update_nv_lines(self):
        self.LOG.debug("Updating NV lines")
        for nv in self.nvs:
            line = self.nvs[nv]["line"]
            data = np.min(self.nvs[nv]["data"], axis=0)
            if line is None:
                self.LOG.debug(f"line is None setting initial line")
                l = self.canvas.ax.plot(
                    self.frequencies[0],
                    data,
                    label=self.nvs[nv]["label"].replace(" ", ""),
                    color=self.nvs[nv]["color"],
                    alpha=0.9,
                    ls="--",
                    lw=0.5,
                    visible=self.visibility_boxes[nv].isChecked(),
                )
                self.nvs[nv]["line"] = l[0]
            else:
                line.set_data(self.frequencies[0], data)

    def update_nv_data(self):
        self.LOG.debug("Updating NV data")
        # iterate over all NVs
        for nv in self.nvs:
            # iterate over field ranges
            for i in range(2):
                nv_parameter = np.array(
                    [
                        self.nvs[nv]["resonance"][i],
                        self.nvs[nv]["width"],
                        self.nvs[nv]["contrast"],
                        self.nvs[nv]["contrast"],
                        0,
                    ]
                )

                data = esr15n(
                    self.frequencies[i],
                    nv_parameter,
                )

                self.nvs[nv]["data"][i] = data

        self.nv_sum = np.sum([self.nvs[nv]["data"][i] for nv in self.nvs for i in range(2)], axis=0)
        self.nv_sum -= np.max(self.nv_sum)
        self.nv_sum += 1

    def update_plot_nv_text(self):
        self.LOG.debug("updating text")
        ax = self.canvas.ax
        trans = transforms.blended_transform_factory(self.canvas.ax.transData, self.canvas.ax.transAxes)

        for nv in self.nvs:
            for i in range(2):
                if self.nvs[nv]["text"][i] is None:
                    txt = ax.text(
                        self.nvs[nv]["resonance"][i],
                        1.2,
                        f"NV$_{nv+1}$\n |",
                        va="center",
                        ha="center",
                        transform=trans,
                        bbox=dict(facecolor="w", alpha=0.5, edgecolor="none", pad=0),
                        color=self.nvs[nv]["color"],
                    )
                    self.nvs[nv]["text"][i] = txt
                else:
                    self.nvs[nv]["text"][i].set_x(self.nvs[nv]["resonance"][i])

    def update_plot_measure_range(self):
        bias_shift = b2shift(self.bias_field_slider.value(), in_unit="microT")
        b_shift = b2shift(self.range_box.value(), in_unit="microT")

        for i, span in enumerate(self.nv_span):
            if span is None:
                self.nv_span[i] = self.canvas.ax.axvspan(
                    ZFS - b_shift + bias_shift * [1, -1][i],
                    ZFS + b_shift + bias_shift * [1, -1][i],
                    color="g",
                    alpha=0.1,
                    label="range" if i == 0 else None,
                    visible=self.range_visibility.isChecked(),
                )
            else:
                xy = [
                    [ZFS - b_shift + bias_shift * [1, -1][i], 0],
                    [ZFS - b_shift + bias_shift * [1, -1][i], 1],
                    [ZFS + b_shift + bias_shift * [1, -1][i], 1],
                    [ZFS + b_shift + bias_shift * [1, -1][i], 0],
                ]
                span.set_xy(xy)

    def update_danger_range(self):
        all_models = self.monte_carlo_models()
        nv0_model = self.monte_carlo_models(nv_indices=[0])
        min_all_models = np.min(all_models, 0)
        min_nv0_model = np.min(nv0_model, 0)

        for i, span in enumerate(self.danger_span):
            if span is not None:
                self.danger_span[i].remove()
                self.danger_span[i] = None

        if self.danger_span[0] is None:

            self.danger_span[1] = self.canvas.ax.fill_between(
                self.frequencies[0],
                np.ones(len(self.frequencies[0])),
                min_all_models,
                color="r",
                alpha=0.05,
                linewidth=0,
                label="MIN(NV$_{2-4}$) $\\forall$ dec/inc",
                visible=self.danger_range_visibility.isChecked(),
            )
            self.danger_span[2] = self.canvas.ax.fill_between(
                self.frequencies[0],
                np.ones(len(self.frequencies[0])),
                min_nv0_model,
                color="g",
                alpha=0.1,
                linewidth=0,
                label="MIN(NV$_0$) $\\forall$ dec/inc",
                visible=self.measure_range_visibility.isChecked(),
            )

    def update_plot_range_text(self):
        bias_shift = b2shift(self.bias_field_slider.value(), in_unit="microT")
        b_shift = b2shift(self.range_box.value(), in_unit="microT")

        for i, txt in enumerate(self.nv_span_text):
            if txt is not None:
                txt.remove()

            trans = transforms.blended_transform_factory(self.canvas.ax.transData, self.canvas.ax.transAxes)
            self.nv_span_text[i] = self.canvas.ax.text(
                ZFS + bias_shift * [1, -1][i],
                0.1,
                f"{ZFS + bias_shift * [1, -1][i] - b_shift:.4}-{ZFS + bias_shift * [1, -1][i] + b_shift:.4} GHz",
                va="center",
                ha="center",
                transform=trans,
                color="k",
                visible=self.range_visibility.isChecked(),
            )

    def update_plot_sum_line(self):
        self.LOG.debug("Updating sum line")
        if self.nv_sum_line is None:
            l = self.canvas.ax.plot(self.frequencies[0], self.nv_sum, label="Spectrum", color="0.1", lw=0.9)
            self.nv_sum_line = l[0]
        else:
            self.nv_sum_line.set_data(self.frequencies[0], self.nv_sum)

    def update_xlim(self):
        self.LOG.debug("Updating X limits")
        xmin = np.min(self.frequencies[0])
        xmax = np.max(self.frequencies[0])
        self.canvas.ax.set_xlim(xmin, xmax)
        self.canvas.axb.set_xlim(
            freq2b(xmin, in_unit="GHz", out_unit="milliT"),
            freq2b(xmax, in_unit="GHz", out_unit="milliT"),
        )

    def update_ylim(self):
        mn = np.min(self.nv_sum_line.get_ydata())
        mx = np.max(self.nv_sum_line.get_ydata())
        self.LOG.debug("Updating Y limits to %s, %s", mn, mx)
        self.canvas.ax.set_ylim(mn * 0.997, mx * 1.001)

    def update_plot(self):
        self.LOG.debug("Updating plot")
        self.LOG.debug("-" * 80)
        self.update_plot_measure_range()
        self.update_danger_range()
        self.update_nv_data()
        self.update_nv_lines()
        self.update_plot_sum_line()
        self.update_plot_nv_text()
        self.update_plot_range_text()
        self.update_ylim()
        self.canvas.draw()

    def monte_carlo_models(self, nv_indices=[1, 2, 3]):
        """Calculates the models for all dec/inc combinations of the current applied field.

        This function is meant to calculate the "safe" field margins for the current applied field.

        Returns:
            ndarray: The calculated models for all dec/inc combinations for NV axes > 0 of the current applied field.
        """
        self.LOG.debug("Monte Carlo models")

        bias_xyz = dim2xyz(
            np.array([self.bias_dec_slider.value(), self.bias_inc_slider.value(), self.bias_field_slider.value()])
        )

        all_models = np.concatenate(
            [
                utils.monte_carlo_models(
                    freqs=self.frequencies[0],
                    bias_xyz=bias_xyz,
                    b_source=self.source_field_slider.value(),
                    nv_direction=self.nvs[nv]["direction"],
                    width=self.width_boxes[nv].value() / 1000,
                    contrast=self.contrast_boxes[nv].value() / 100,
                )
                for nv in nv_indices
            ]
        )

        return all_models
        # # generate the dec/inc combinations at current source field
        # source_dim = utils.generate_possible_dim(b_source=self.source_field_slider.value(), n=10)
        # source_xyz = dim2xyz(source_dim)
        #
        # bias_xyz = dim2xyz(
        #     np.array([self.bias_dec_slider.value(), self.bias_inc_slider.value(), self.bias_field_slider.value()])
        # )
        #
        # total_field_xyz = source_xyz + bias_xyz
        #
        # all_models = []
        #
        # for b in total_field_xyz:
        #     all_nv_model = []
        #
        #     # iterate over all NV axes
        #     for nv in self.nvs:
        #         # skip NV axes that are not in the list of NV axes to be considered
        #         if nv not in nv_indices:
        #             continue
        #
        #         # calculate the model for the current NV axis
        #         # for both field ranges (i.e. +/- field shift)
        #         for i in range(2):
        #             # determine projected field on NV axis
        #             projected_field = project(b, self.nvs[nv]["direction"])
        #             parameters = np.array(
        #                 [
        #                     ZFS + [1, -1][i] * b2shift(projected_field, in_unit="microT", out_unit="GHz"),
        #                     self.width_boxes[nv].value() / 1000,
        #                     self.contrast_boxes[nv].value() / 100,
        #                     self.contrast_boxes[nv].value() / 100,
        #                     0,
        #                 ]
        #             )
        #             nv_model = esr15n(self.frequencies[i], parameters)
        #             all_nv_model.append(nv_model)
        #
        #     all_nv_model = np.sum(all_nv_model, axis=0)
        #     all_nv_model -= np.max(all_nv_model)
        #     all_nv_model += 1
        #
        #     all_models.append(all_nv_model)
        #
        # return np.reshape(all_models, (-1, len(self.frequencies[0])))


def BoxSlider(value, decimals, step, vmin, vmax, callback, suffix=None):
    callback = np.atleast_1d(callback)

    hlayout = QHBoxLayout()

    spinbox = QDoubleSpinBox()
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setMinimum(vmin)
    spinbox.setMaximum(vmax)
    spinbox.setValue(value)
    spinbox.setKeyboardTracking(False)
    spinbox.setFixedWidth(70)
    spinbox.setSuffix(suffix)

    slider = QSlider()
    slider.setOrientation(Qt.Horizontal)
    slider.setRange(vmin, vmax)
    slider.setValue(value)
    slider.setFixedWidth(200)

    for cb in callback:
        slider.valueChanged.connect(cb)
        spinbox.valueChanged.connect(cb)

    slider.valueChanged.connect(lambda x: spinbox.setValue(slider.value()))
    spinbox.valueChanged.connect(lambda x: slider.setValue(x))
    hlayout.addWidget(spinbox, Qt.AlignLeft)
    hlayout.addWidget(slider, alignment=Qt.AlignLeft)
    hlayout.setContentsMargins(0, 0, 0, 0)
    return hlayout, slider


def CheckBox(text, callback, checked=False):
    callback = np.atleast_1d(callback)

    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)

    for cb in callback:
        checkbox.stateChanged.connect(cb)

    return checkbox


def DoubleSpinBox(value, decimals, step, vmin, vmax, callback):
    """
    Returns a label and a spin box.

    :param label: str
        label text
    :param value: float, int
        initial value
    :param decimals: int
        number of decimals
    :param step: float, int
        step size
    :param vmin: float, int
        minimum value
    :param vmax: float, int
        maximum value
    :param callback:
        callback function

    :return: QLabel, QDoubleSpinBox
    """
    selector = QDoubleSpinBox()
    selector.setDecimals(decimals)
    selector.setSingleStep(step)
    selector.setMinimum(vmin)
    selector.setMaximum(vmax)
    selector.setValue(value)
    selector.setKeyboardTracking(False)

    callback = np.atleast_1d(callback)
    for cb in callback:
        selector.valueChanged.connect(cb)
    return selector


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

from QDMpy import utils
