#!/usr/bin/env python
"""
This file contains the QDMpy mainwindow for the gui.
QDMpy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
QDMpy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with QDMpy. If not, see <https://www.gnu.org/licenses/>.
Copyright (c) the QDMpy Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/mikevolk/QDMpy>
"""

import contextlib
import logging
import sys
import time
from pathlib import Path

import matplotlib

from src.QDMpy import test_data_location

matplotlib.use("Agg")

from QDMpy.app.widgets.fit_widget import FitWidget
from QDMpy.utils import millify

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QScreen
from PySide6.QtWidgets import (
    QApplication,
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
    QStatusBar,
    QToolBar,
    QWidget,
)

import QDMpy
from QDMpy.app.widgets.fluo_widget import FluoWidget
from QDMpy.app.widgets.global_widget import GlobalWidget
from QDMpy.app.widgets.misc import PandasWidget, gf_applied_window
from QDMpy.app.widgets.simple_widget import SimpleWidget
from QDMpy.app.widgets.warning_windows import PyGPUfitNotInstalledDialog
from QDMpy._core.fit import CONSTRAINT_TYPES
from QDMpy._core.qdm import QDM
from QDMpy.exceptions import CantImportError

colors = {
    "Bright Gray": "#EAEFF9",
    "Pale Cerulean": "#A1C4D8",
    "Blue-Gray": "#5C9DC0",
    "X11 Gray": "#BEBEBE",
    "Taupe Gray": "#878787",
    "near-white": "#F8F8F8",
    "near-black": "#1E1E1E",
}
plt.style.use("fast")
matplotlib.rcParams.update({"font.size": 8, "axes.labelsize": 8, "grid.linestyle": "-", "grid.alpha": 0.5})


class QDMpyApp(QMainWindow):
    _visible_if_qdm_present = []
    _visible_if_fitted = []

    def __init__(self, **kwargs):

        self.LOG = logging.getLogger(f"QDMpy.{self.__class__.__name__}")
        super().__init__()
        self.debug = kwargs.pop("debug", False)
        self.outlier_pd = pd.DataFrame(columns=["idx", "x", "y"])

        if not QDMpy.PYGPUFIT_PRESENT:
            self.pygpufit_not_available_dialog()

        screen = kwargs.pop("screen", None)

        if screen is None:
            self.screen_size = [1920, 1080]
        else:
            self.screen_size = screen.size().width(), screen.size().height()

        self.screen_ratio = self.screen_size[1] / self.screen_size[0]
        self.fluorescence_window = None
        self.laser_window = None
        self.light_window = None
        self.gf_window = None
        self.main_content_figure = None
        self.quality_window = None

        self.qdm = None
        self.fitconstraints_widget = None
        self.fitconstraints = {}

        self.infotable_widget = None
        self._info_elements = None

        self._current_idx = None

        self.setWindowTitle("QDMpy")
        self.resize(
            int(0.6 * self.screen_size[0]),
            int(0.8 * self.screen_size[0] * self.screen_ratio),
        )

        self.get_menu()
        self.get_toolbar()
        self.get_statusbar()

        self.init_main_content()
        self.get_infotable_widget()

        self._change_tool_visibility()

        self._data_windows = []

        if self.debug:
            self.debug_call()

    # TOOLBAR
    def get_toolbar(self):
        """
        Create the toolbar and add the actions
        """

        # create the main toolbar
        main_toolbar = QToolBar("Toolbar")
        main_toolbar.setIconSize(QSize(12, 12))
        self.addToolBar(main_toolbar)

        self._add_info_button_toolbar(main_toolbar)
        self._add_pixelsize_toolbar(main_toolbar)

        plotting_toolbar = QToolBar("Plotting")
        self.addToolBar(plotting_toolbar)

        self._add_led_plot_toolbar(plotting_toolbar)
        self._add_laser_plot_toolbar(plotting_toolbar)
        self._add_fluorescence_plot_toolbar(plotting_toolbar)

        self.edit_toolbar = QToolBar("Edit")
        self.addToolBar(self.edit_toolbar)
        self._add_bin_toolbar(self.edit_toolbar)
        self._add_gf_toolbar(self.edit_toolbar)
        self._add_fit_toolbar(self.edit_toolbar)
        self._add_quickstart_button(self.edit_toolbar)

    def _add_info_button_toolbar(self, toolbar):
        """
        Add the info button to toolbar
        """
        self.infoButton = QAction("I", self)
        self.infoButton.triggered.connect(self.on_info_button_press)
        self.infoButton.setShortcut(QKeySequence("Ctrl+I"))
        self._visible_if_qdm_present.append(self.infoButton)
        self.infoButton.setEnabled(False)
        icon = QIcon("assets/icons/info.png")
        self.infoButton.setIcon(icon)
        self.infoButton.setStatusTip("Show info")
        toolbar.addAction(self.infoButton)

    def _add_quickstart_button(self, toolbar):
        """
        Add the quickstart button to toolbar
        """
        toolbar.addSeparator()
        self.quickstartButton = QPushButton("Q")
        self.quickstartButton.setStatusTip("Open quickstart")
        self.quickstartButton.clicked.connect(self.on_quick_start_button_press)
        self.quickstartButton.setFixedSize(25, 25)
        self.quickstartButton.setEnabled(True)
        toolbar.addWidget(self.quickstartButton)

    def _add_fluorescence_plot_toolbar(self, toolbar):
        """
        Add the fluorescence plot button to toolbar
        """
        self.fluorescencePlotButton = QAction("Fluo.", self)
        self.fluorescencePlotButton.setStatusTip("Open fluorescence plots")
        self.fluorescencePlotButton.triggered.connect(self.on_fluorescence_button_press)
        self.fluorescencePlotButton.setShortcut(QKeySequence("Ctrl+F"))
        self._visible_if_qdm_present.append(self.fluorescencePlotButton)
        toolbar.addAction(self.fluorescencePlotButton)

    def _add_laser_plot_toolbar(self, toolbar):
        """
        Add the laser plot button to toolbar
        """
        self.laserPlotButton = QAction("Laser", self)
        self.laserPlotButton.setStatusTip("Open Laser Scan")
        self.laserPlotButton.triggered.connect(self.on_laser_button_press)
        self.laserPlotButton.setShortcut(QKeySequence("Ctrl+L"))
        self.laserPlotButton.setEnabled(False)
        self._visible_if_qdm_present.append(self.laserPlotButton)
        toolbar.addAction(self.laserPlotButton)

    def _add_led_plot_toolbar(self, toolbar):
        """
        Add the LED plot button to toolbar
        """
        self.ledPlotButton = QAction("Light", self)
        self.ledPlotButton.setStatusTip("Open Reflected Light Plot")
        self.ledPlotButton.triggered.connect(self.on_led_button_press)
        self.ledPlotButton.setShortcut(QKeySequence("Ctrl+R"))
        self.ledPlotButton.setEnabled(False)
        self._visible_if_qdm_present.append(self.ledPlotButton)
        toolbar.addAction(self.ledPlotButton)

    def _add_fit_toolbar(self, toolbar):
        toolbar.addSeparator()

        self.fit_button = QPushButton("Fit")
        self.fit_button.setStatusTip("Fit the data")
        self.fit_button.clicked.connect(self.on_fit_button_press)
        self.fit_button.setFixedSize(50, 25)

        self._visible_if_qdm_present.append(self.fit_button)
        toolbar.addWidget(self.fit_button)
        self.fit_constraints_button = QPushButton("Constraints")
        self.fit_constraints_button.setStatusTip("Edit the fit constraints")
        self.fit_constraints_button.clicked.connect(self.on_set_fitconstraints_button_press)
        self._visible_if_qdm_present.append(self.fit_constraints_button)
        toolbar.addWidget(self.fit_constraints_button)

    def _add_pixelsize_toolbar(self, toolbar):
        pixel_widget = QWidget()
        pixel_size_box = QHBoxLayout()
        pixel_size_label, self.pixel_size_select = self.get_label_box(
            label="Pixel Size [µm]:",
            value=1,
            decimals=2,
            step=1,
            vmin=1,
            vmax=99,
            callback=self.on_pixel_size_changed,
        )
        pixel_size_box.addWidget(pixel_size_label)
        pixel_size_box.addWidget(self.pixel_size_select)
        pixel_widget.setLayout(pixel_size_box)
        toolbar.addWidget(pixel_widget)

    def _add_bin_toolbar(self, toolbar):
        bin_widget = QWidget()
        bin_box = QHBoxLayout()
        bin_factor_label, self.binfactor_select = self.get_label_box(
            label="Bin Factor:",
            value=1,
            decimals=0,
            step=1,
            vmin=1,
            vmax=32,
            callback=self.on_bin_factor_changed,
        )
        bin_box.addWidget(bin_factor_label)
        bin_box.addWidget(self.binfactor_select)
        self.bin_button = QPushButton("Bin")
        self.bin_button.setStatusTip("Bin the data")
        self.bin_button.setFixedSize(50, 25)
        self.bin_button.clicked.connect(self.on_bin_button_press)
        self._visible_if_qdm_present.append(self.bin_button)
        bin_box.addWidget(self.bin_button)
        bin_widget.setLayout(bin_box)
        toolbar.addWidget(bin_widget)

    def _add_gf_toolbar(self, toolbar):
        global_widget = QWidget()
        global_box = QHBoxLayout()
        gf_label, self.gf_select = self.get_label_box("Global Fluorescence", 0, 1, 0.1, 0, 1, None)
        self.gf_detect_button = QPushButton("detect")
        self.gf_detect_button.setStatusTip("Detect global fluoresence")
        self.gf_detect_button.clicked.connect(self.on_gf_detect_button_press)
        self.gf_detect_button.setFixedSize(50, 25)

        self.gf_apply_button = QPushButton("apply")
        self.gf_apply_button.setFixedSize(50, 25)
        self.gf_apply_button.setStatusTip("Apply global fluoresence")
        self.gf_apply_button.clicked.connect(self.on_gf_apply_button_press)
        global_box.addWidget(gf_label)
        global_box.addWidget(self.gf_select)
        global_box.addWidget(self.gf_apply_button)
        global_box.addWidget(self.gf_detect_button)
        self._visible_if_qdm_present.extend([self.gf_detect_button, self.gf_apply_button])
        global_widget.setLayout(global_box)
        toolbar.addWidget(global_widget)

    # MENU BAR

    def get_menu(self):
        """
        Create the menu and add the actions
        """
        self.setStatusBar(QStatusBar(self))
        menu = self.menuBar()

        # About_QDMpy
        about_QDMpy_button = QAction("&About QDMpy", self)
        about_QDMpy_button.setStatusTip("This is your button")
        about_QDMpy_button.triggered.connect(self.on_about_QDMpy_button_press)

        # file menu
        file_menu = menu.addMenu("&File")
        file_submenu_import = file_menu.addMenu("import")
        # import QDM button
        import_qdmio_button = QAction("QDMio", self)
        import_qdmio_button.setStatusTip("import QDMio like files")
        import_qdmio_button.triggered.connect(self.on_qdmio_button_press)
        file_submenu_import.addAction(import_qdmio_button)
        import_qdmio_button.setShortcut(QKeySequence("Ctrl+I"))

        self._export_menu(file_menu)

        file_menu.addSeparator()
        file_menu.addAction(about_QDMpy_button)
        close_action = file_menu.addAction("&Quit", self.close)
        close_action.setShortcut(QKeySequence("Ctrl+Q"))

        # edit menu
        edit_menu = menu.addMenu("&Edit")
        set_fit_constraints_button = QAction("Set Fit Constraints", self)
        set_fit_constraints_button.setStatusTip("Set Fit Constraints")
        set_fit_constraints_button.triggered.connect(self.on_set_fitconstraints_button_press)
        edit_menu.addAction(set_fit_constraints_button)

        # view menu
        view_menu = menu.addMenu("&View")
        self._add_led_plot_toolbar(view_menu)
        self._add_laser_plot_toolbar(view_menu)
        self._add_fluorescence_plot_toolbar(view_menu)

        outlier_menu = menu.addMenu("outliers")
        detect_outlier_button = QAction("detect outliers", self)
        detect_outlier_button.setStatusTip("detect outliers")
        detect_outlier_button.triggered.connect(self.on_detect_outlier_button_press)
        outlier_menu.addAction(detect_outlier_button)

        self.mark_outlier_button = QAction("mark outliers", self, checkable=True)
        self.mark_outlier_button.setStatusTip("mark outliers on plots")
        self.mark_outlier_button.triggered.connect(self.mark_outlier)
        outlier_menu.addAction(self.mark_outlier_button)

        outlier_list_button = QAction("outliers list", self)
        outlier_list_button.setStatusTip("outliers list")
        outlier_list_button.triggered.connect(self.on_show_outlier_list_button_press)
        outlier_menu.addAction(outlier_list_button)
        # file_submenu.addAction(import_qdmio_button)

    def _export_menu(self, file_menu):
        file_submenu_export = file_menu.addMenu("export")
        # export QDM button
        export_qdmpy_button = QAction("QDMpy", self)
        export_qdmpy_button.setStatusTip("export to QDMio file")
        export_qdmpy_button.triggered.connect(self.export_qdmpy)
        export_qdmpy_button.setEnabled(False)  # disable until a QDMpy file is fitted
        file_submenu_export.addAction(export_qdmpy_button)
        self._visible_if_fitted.append(export_qdmpy_button)

        export_qdmio_button = QAction("QDMio", self)
        export_qdmio_button.setStatusTip("export to QDMio file")
        export_qdmio_button.triggered.connect(self.export_qdmio)
        export_qdmio_button.setEnabled(False)  # disable until a QDMpy file is fitted
        file_submenu_export.addAction(export_qdmio_button)
        self._visible_if_fitted.append(export_qdmio_button)

    # fit parameters

    def on_fitconstraints_button_press(self):
        """
        Create the widget for the fit constraints
        :return:
        """

        if self.fitconstraints_widget is None:
            self.get_fitconstraints_widget()
            self._fill_fitconstraints_widget()

        if self.fitconstraints_widget.isVisible():
            self.fitconstraints_widget.hide()
        else:
            self.fitconstraints_widget.show()

    def get_fitconstraints_widget(self):
        self.fitconstraints_widget = QGroupBox("Fit Constraints")
        self.fitconstraints_gridlayout = QGridLayout()
        self.fitconstraints_widget.setTitle("Fit Constraints")

        for row, (text, (vmin, vmax, constraint, unit)) in enumerate(self.qdm.fit.constraints.items()):
            self.fitconstraints[text] = {
                "label": QLabel(text),
                "vmin": QLineEdit(f"{vmin}"),
                "vmax": QLineEdit(f"{vmax}"),
                "unit": QLabel(f"{unit}"),
                "box": QComboBox(),
            }
            self.fitconstraints[text]["vmin"].returnPressed.connect(self.on_fitconstraints_widget_item_changed)
            self.fitconstraints[text]["vmax"].returnPressed.connect(self.on_fitconstraints_widget_item_changed)

            self.fitconstraints[text]["box"].addItems(["FREE", "LOWER", "UPPER", "LOWER_UPPER"])
            self.fitconstraints[text]["box"].setCurrentText(constraint)
            self.fitconstraints[text]["box"].currentIndexChanged.connect(self.on_fitconstraints_widget_item_changed)

            # add them to the layout
            for col, item in enumerate(self.fitconstraints[text].values()):
                self.fitconstraints_gridlayout.addWidget(item, row, col)
        self.fitconstraints_widget.setLayout(self.fitconstraints_gridlayout)

    def _fill_fitconstraints_widget(self):
        for text, (vmin, vmax, constraint, unit) in self.qdm.fit.constraints.items():
            self.fitconstraints[text]["box"].setCurrentText(constraint)
            self.fitconstraints[text]["vmin"].setText(str(vmin))
            self.fitconstraints[text]["vmax"].setText(str(vmax))
            self.fitconstraints[text]["unit"].setText(str(unit))

            self._set_constraint_visibility(
                constraint,
                self.fitconstraints[text]["vmin"],
                self.fitconstraints[text]["vmax"],
            )

    @staticmethod
    def _set_constraint_visibility(tpe: str, vmax_item: QLabel, vmin_item: QLabel):
        """
        Set the visibility of the min and max values depending on the constraint type

        Args:
            tpe: constraint type
            vmax_item: max value item
            vmin_item: min value item

        Returns:

        """
        if tpe in ["LOWER", "FREE"]:
            vmin_item.setEnabled(False)
        else:
            vmin_item.setEnabled(True)
        if tpe in ["UPPER", "FREE"]:
            vmax_item.setEnabled(False)
        else:
            vmax_item.setEnabled(True)

    def on_fitconstraints_widget_item_changed(self):
        for k, v in self.fitconstraints.items():
            self.qdm.set_constraints(k, float(v["vmin"].text()), float(v["vmax"].text()), v["box"].currentText())
        self._fill_fitconstraints_widget()

    ### INFO TABLE ###
    def get_infotable_widget(self):
        self.infotable_widget = QGroupBox(
            "Measurement Infos",
        )
        self.info_grid = QGridLayout()
        self.infotable_widget.setLayout(self.info_grid)

    def _fill_info_table(self):

        entries = [
            "Image dimensions",
            "Data dimensions",
            "Field directions",
            "Frequency ranges",
            "# pixels",
            "# frequencies",
            "bin factor",
        ]

        # fill index column
        if self._info_elements is None:
            self._info_elements = {}
            for i, e in enumerate(entries):
                self._info_elements[e] = [QLabel(e), QLabel()]
                self._info_elements[e][0].setStyleSheet("font-weight: bold")

                self.info_grid.addWidget(self._info_elements[e][0], i, 0)
                self.info_grid.addWidget(self._info_elements[e][1], i, 1)

        # stop if qdm is not loaded, yet
        if self.qdm is None:
            return

        # fill the info column
        infos = self.get_infos()

        self.LOG.debug("updating info table")
        if infos is not None:
            for i, e in enumerate(entries):
                self._info_elements[e][1].setText(str(infos[i]))

    def on_info_button_press(self):
        if self.infotable_widget is None:
            self.get_infotable_widget()
            self._fill_info_table()
        elif self.infotable_widget.isVisible():
            self.infotable_widget.hide()
        else:
            self.infotable_widget.show()

    # status bar # todo add statusbar updates
    def get_statusbar(self):
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Welcome to QDMpy")

    # loading dialog
    @staticmethod
    def get_loading_progress_dialog(title, text):
        loading_progress_dialog = QProgressDialog()
        loading_progress_dialog.setLabelText(text)
        loading_progress_dialog.setMinimum(0)
        loading_progress_dialog.setMaximum(0)
        loading_progress_dialog.setWindowTitle(title)
        loading_progress_dialog.setAutoReset(True)
        loading_progress_dialog.setAutoClose(True)
        loading_progress_dialog.setFixedSize(250, 100)
        loading_progress_dialog.setWindowModality(Qt.WindowModal)
        loading_progress_dialog.show()
        loading_progress_dialog.setValue(0)
        time.sleep(0.1)
        # loading_progress_dialog.setCancelButton()
        return loading_progress_dialog

    def pygpufit_not_available_dialog(self):
        """
        Show a warning dialog if pyGPUfit is not installed
        """
        if not self.debug:
            dlg = PyGPUfitNotInstalledDialog()
            if dlg.exec():
                logging.debug("pyGPUfit not installed continuing without it")
            else:
                logging.debug("pyGPUfit not installed closing")
                sys.exit()
        self.statusBar().showMessage("pygpufit is not available")

    def get_label_box(self, label, value, decimals, step, vmin, vmax, callback):
        label = QLabel(label)
        selector = QDoubleSpinBox(self)
        selector.setValue(value)
        selector.setDecimals(decimals)
        selector.setSingleStep(step)
        selector.setMinimum(vmin)
        selector.setMaximum(vmax)
        selector.setKeyboardTracking(False)
        if callback is not None:
            selector.valueChanged.connect(callback)
        selector.setEnabled(False)
        selector.setFixedWidth(50)
        self._visible_if_qdm_present.append(selector)
        return label, selector

    # MAIN WINDOW
    def init_main_content(self):
        self.main_content_layout = QHBoxLayout()
        self.main_label = QLabel("No QDM data loaded, yet.\nNo fitting possible.")
        self.main_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_content_layout.addWidget(self.main_label)
        self.main_content_figure = None

        widget = QWidget()
        widget.setLayout(self.main_content_layout)
        self.setCentralWidget(widget)

    def update_main_content(self):
        self.LOG.debug("Updating main content")
        self._change_tool_visibility()

        if self.qdm is not None:
            self._fill_info_table()

        if self.qdm is None:
            self.LOG.debug("No QDM data loaded, yet.\nNo fitting possible.")
            self.init_main_content()
        elif not self.qdm.fit.fitted:
            self.LOG.debug("QDM data loaded, fitting possible.")
            self.main_label.setText("No fit calculated yet.")
        else:
            self.LOG.debug("QDM data fitted, plotting results.")
            if self.main_content_figure is None:
                self._replace_label_with_fit_window()
            else:
                self.main_content_figure.redraw_all_plots()

    def _replace_label_with_fit_window(self):
        self.main_content_layout.removeWidget(self.main_label)
        self.main_content_figure = FitWidget(parent=self)
        self._data_windows.append(self.main_content_figure)
        self.main_content_layout.addWidget(self.main_content_figure)
        self.setCentralWidget(self.main_content_figure)
        self.main_content_figure.show()

    @property
    def widgets(self):
        return self.findChildren(QMainWindow)

    @property
    def _need_marker_update(self):
        return [w for w in self.widgets if w.needs_marker_update]

    @property
    def _need_odmr_update(self):
        return [w for w in self.widgets if w.needs_odmr_update]

    def update_marker(self):
        self.LOG.debug(f"update_marker in {self._need_marker_update}")
        for window in self._need_marker_update:
            window.update_marker()
            window.canvas.draw_idle()

    def update_odmr(self):
        self.LOG.debug(f"Updating pixel in {self._need_odmr_update}")
        for window in self._need_odmr_update:
            window.update_odmr()
            window.canvas.draw_idle()

    # DATA RELATED
    def set_current_idx(self, x=None, y=None, idx=None):
        if x is not None and y is not None:
            self._current_idx = self.qdm.odmr.rc2idx([y, x])
        elif idx is not None:
            self._current_idx = idx

    @property
    def _current_xy(self):
        """
        Returns the current xy-coordinates in data coordinates.
        :return:
        """
        return self.qdm.odmr.idx2rc(self._current_idx)[::-1]

    # BUTTON ACTION
    # toolbar / menu
    def on_fluorescence_button_press(self):
        if self.fluorescence_window is None:
            self.fluorescence_window = FluoWidget(parent=self)
        if self.fluorescence_window.isVisible():
            self.fluorescence_window.hide()
        else:
            self.fluorescence_window.show()

    def on_laser_button_press(self):
        if self.laser_window is None:
            self.laser_window = SimpleWidget(dtype="laser", parent=self)
            self.laser_window.show()
        elif self.laser_window.isVisible():
            self.laser_window.hide()
        else:
            self.laser_window.show()

    def on_led_button_press(self):
        if self.light_window is None:
            self.light_window = SimpleWidget(dtype="light", clim_select=False, parent=self)
            self.light_window.show()
        elif self.light_window.isVisible():
            self.light_window.hide()
        else:
            self.light_window.show()

    def on_qdmio_button_press(self):
        work_directory = QFileDialog.getExistingDirectory(self, "Open path", "./")
        self.import_file(work_directory, dialect="QDMio")

    def on_detect_outlier_button_press(self):
        if self.qdm.fitted:
            self.LOG.debug("Detecting outliers")
            self.qdm.detect_outliers("width", method="IsolationForest")

            self.outlier_pd = pd.DataFrame(columns=["idx", "x", "y"])
            self.outlier_pd["x"] = self.qdm.outliers_xy[:, 0]
            self.outlier_pd["y"] = self.qdm.outliers_xy[:, 1]
            self.outlier_pd["idx"] = self.qdm.outliers_idx
            self.statusBar().showMessage(f"{self.qdm.outliers_idx.shape[0]:8d} Outliers detected")
        else:
            self.LOG.warning("No fit calculated yet, no outliers detected.")
            self.statusBar().showMessage("No fit calculated yet.")

    def mark_outlier(self):
        for w in self._data_windows:
            if self.mark_outlier_button.isChecked():
                self.LOG.debug("Marking outliers")
                w.update_outlier()
            else:
                self.LOG.debug("Removing outlier markers")
                w.toggle_outlier(False)

    def on_show_outlier_list_button_press(self):
        self.LOG.debug("Showing outliers list")

        self.outlierListWidget = PandasWidget(self, self.outlier_pd)

        self.outlierListWidget.show()

    def on_about_QDMpy_button_press(self):
        about_message_box = QMessageBox.about(
            self,
            "About QDMpy",
            "QDMpy is written by Mike Volk during his hollidays in 2022...\nWhat a dork... - Chrissi -",
        )

    def on_quick_start_button_press(self):
        if self.qdm.bin_factor == 1:
            self.binfactor_select.setValue(4)
            self.on_bin_button_press()
        self.gf_select.setValue(0.2)
        self.on_gf_apply_button_press()
        self.on_fit_button_press()
        self.on_detect_outlier_button_press()

    # main content
    def on_pixel_size_changed(self):
        self.LOG.debug(f"Pixel Size changed to {self.pixel_size_select.value()}")

    def on_bin_factor_changed(self):
        self.LOG.debug(f"Bin Factor changed to {self.binfactor_select.value()}")

    def on_bin_button_press(self):
        self.LOG.debug("Bin Button clicked")
        self.qdm.bin_data(self.binfactor_select.value())
        if self.qdm.fitted:
            self.qdm.fit_odmr()
        self._current_idx = self.qdm.odmr.get_most_divergent_from_mean()[-1]
        self.update_main_content()
        self._fill_fitconstraints_widget()

    def on_fit_button_press(self):
        if not QDMpy.PYGPUFIT_PRESENT:
            self.LOG.warning("PyGpuFit not present")
            return

        self.qdm.fit_odmr()
        self.update_main_content()
        self.update_marker()
        self.update_odmr()
        self._fill_info_table()
        self._fill_fitconstraints_widget()

    def on_gf_detect_button_press(self):
        if self.gf_window is None:
            self.gf_window = GlobalWidget(parent=self, qdm_instance=self.qdm)
        if self.gf_window.isVisible():
            self.gf_window.hide()
        else:
            self.gf_window.global_slider.setValue(self.gf_select.value() * 100)
            self.gf_window.update_odmr()
            self.gf_window.show()

    def on_gf_apply_button_press(self):
        self.LOG.debug("GF Apply Button clicked")
        self.qdm.correct_glob_fluorescence(self.gf_select.value())
        if not self.debug:
            gf_applied_window(self.gf_select.value())

    def on_set_fitconstraints_button_press(self):
        self.on_fitconstraints_button_press()

    # additional functions
    def set_pixel_size_from_qdm_obj(self):
        if self.qdm is not None:
            with contextlib.suppress(AttributeError):
                self.pixel_size_select.setValue(self.qdm.pixel_size)

    def get_infos(self):
        return [
            f"{self.qdm.odmr.img_shape[0]}x{self.qdm.odmr.img_shape[1]}",
            f"{self.qdm.odmr.data_shape[0]}x{self.qdm.odmr.data_shape[1]}",
            self.qdm.odmr.n_pol,
            self.qdm.odmr.n_frange,
            millify(self.qdm.odmr.data[0, 0, :, 0].size),
            self.qdm.odmr.n_freqs,
            self.qdm.bin_factor,
        ]

    def _change_tool_visibility(self):
        for action in self._visible_if_qdm_present:
            action.setEnabled(self.qdm is not None)
        for action in self._visible_if_fitted:
            if self.qdm is not None:
                action.setEnabled(self.qdm.fitted)

    def import_file(self, work_directory, dialect="QDMio"):
        loading_progress_dialog = self.get_loading_progress_dialog(
            "Importing QDMio like files", "Importing QDMio like files, please wait."
        )
        self.work_directory = Path(work_directory)
        self.init_main_content()

        # time.sleep(1)

        if dialect == "QDMio":
            self.qdm = self.import_qdmio(work_directory)

        if self.qdm is not None:
            self._init_with_new_qdm_obj()
        else:
            QMessageBox.warning(self, "Import Error", "Could not import file.")
        loading_progress_dialog.close()

    def _init_with_new_qdm_obj(self):
        """
        Initializes the app for a new QDMobj.
        """
        self._change_tool_visibility()
        self._fill_info_table()
        # in data coordinates
        self.set_current_idx(idx=self.qdm.odmr.get_most_divergent_from_mean()[-1])
        self.file_imported()
        self.update_main_content()

    # IMPORT FUNCTIONS
    def import_qdmio(self, work_directory):
        self.statusBar().showMessage(f"Importing QDMio like files from {work_directory}")

        self.work_directory = Path(work_directory)
        try:
            qdm_obj = QDM.from_qdmio(self.work_directory)
            self.statusBar().showMessage(f"Successfully imported QDMio like files from {self.work_directory}")

            return qdm_obj
        except CantImportError:
            self.statusBar().showMessage(f"Cant import QDMio like files from {self.work_directory}")

            return

    def file_imported(self):
        self.statusBar().showMessage(f"Successfully imported QDM files from {self.work_directory}")

        self._change_tool_visibility()
        self._fill_info_table()
        self.set_pixel_size_from_qdm_obj()
        self.binfactor_select.setValue(self.qdm.bin_factor)
        self.main_label.setText("No fits calculated yet.")

    # EXPORT FUNCTIONS

    def export_qdmio(self):
        work_directory = QFileDialog.getExistingDirectory(
            self, "Open path", str(self.work_directory), QFileDialog.ShowDirsOnly
        )

        self.LOG.debug(f"Exporting QDMio like files to {work_directory}")
        self.qdm.export_qdmio(work_directory)

    def export_qdmpy(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save File As", str(self.work_directory), "QDMpy B111 Files (*.b111)"
        )
        filename = Path(filename)
        self.LOG.debug(f"Exporting QDMpy like files to {filename}")
        self.qdm.export_qdmpy(filename.with_suffix(".b111"))

    def debug_call(self):
        self.import_file(test_data_location())
        # self.on_quick_start_button_press()
        #
        # if not sys.platform == "darwin":
        #     self.on_fit_button_press()


def main(**kwargs):
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    mainwindow = QDMpyApp(screen=screen, **kwargs)
    mainwindow.show()

    center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
    geo = mainwindow.frameGeometry()
    geo.moveCenter(center)
    mainwindow.move(geo.topLeft())

    app.exec()


if __name__ == "__main__":
    main(debug=True)
