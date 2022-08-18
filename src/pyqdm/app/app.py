# -*- coding: utf-8 -*-
"""
This file contains the pyqdm mainwindow for the gui.
pyqdm is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
pyqdm is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with pyqdm. If not, see <http://www.gnu.org/licenses/>.
Copyright (c) the pyqdm Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/mikevolk/pyqdm>
"""


import contextlib
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import time

import pandas as pd

matplotlib.use('Agg')

import pyqdm
from PySide6.QtCore import Qt, QSize, QAbstractTableModel
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QHeaderView, QWidget, QGroupBox, QDoubleSpinBox, QPushButton, QHBoxLayout,
    QAbstractItemView,
    QLabel, QToolBar, QStatusBar, QTableWidget, QTableWidgetItem, QSizePolicy, QComboBox, QLineEdit, QTableView,
    QMessageBox, QFileDialog, QProgressDialog, QVBoxLayout, QGridLayout)
from pyqdm.core.qdm import QDM
from pyqdm.exceptions import CantImportError

from pyqdm.app.windows.warning_windows import pyGPUfitNotInstalledDialog
from pyqdm.app.windows.misc import GFAppliedWindow
from pyqdm.app.windows.simple_plot_window import SimplePlotWindow
from pyqdm.app.windows.fluorescence_window import FluorescenceWindow
from pyqdm.app.windows.global_fluorescence_window import GlobalFluorescenceWindow

from pyqdm.app.windows import fit_window

colors = {'Bright Gray': '#EAEFF9', 'Pale Cerulean': '#A1C4D8', 'Blue-Gray': '#5C9DC0',
          'X11 Gray': '#BEBEBE', 'Taupe Gray': '#878787', "near-white": '#F8F8F8', "near-black": '#1E1E1E'}
plt.style.use('fast')
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'grid.linestyle': '-',
    'grid.alpha': 0.5})


class PyQDMMainWindow(QMainWindow):
    _visible_if_QDMObj_present = []

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
        self.infoButton.triggered.connect(self.onToolBarinfoButtonClick)
        self.infoButton.setShortcut(QKeySequence("Ctrl+I"))
        self._visible_if_QDMObj_present.append(self.infoButton)
        self.infoButton.setEnabled(False)
        icon = QIcon('assets/icons/info.png')
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
        self.quickstartButton.clicked.connect(self.onQuickStartButtonClick)
        self.quickstartButton.setFixedSize(25, 25)
        self.quickstartButton.setEnabled(True)
        toolbar.addWidget(self.quickstartButton)

    def _add_fluorescence_plot_toolbar(self, toolbar):
        """
        Add the fluorescence plot button to toolbar
        """
        self.fluorescencePlotButton = QAction("Fluo.", self)
        self.fluorescencePlotButton.setStatusTip("Open fluorescence plots")
        self.fluorescencePlotButton.triggered.connect(
            self.onToolBarFluorescenceButtonClick)
        self.fluorescencePlotButton.setShortcut(QKeySequence("Ctrl+F"))
        self._visible_if_QDMObj_present.append(self.fluorescencePlotButton)
        toolbar.addAction(self.fluorescencePlotButton)

    def _add_laser_plot_toolbar(self, toolbar):
        """
        Add the laser plot button to toolbar
        """
        self.laserPlotButton = QAction("Laser", self)
        self.laserPlotButton.setStatusTip("Open Laser Scan")
        self.laserPlotButton.triggered.connect(self.onToolBarLaserButtonClick)
        self.laserPlotButton.setShortcut(QKeySequence("Ctrl+L"))
        self.laserPlotButton.setEnabled(False)
        self._visible_if_QDMObj_present.append(self.laserPlotButton)
        toolbar.addAction(self.laserPlotButton)

    def _add_led_plot_toolbar(self, toolbar):
        """
        Add the LED plot button to toolbar
        """
        self.ledPlotButton = QAction("Light", self)
        self.ledPlotButton.setStatusTip("Open Reflected Light Plot")
        self.ledPlotButton.triggered.connect(self.onToolBarLEDButtonClick)
        self.ledPlotButton.setShortcut(QKeySequence("Ctrl+R"))
        self.ledPlotButton.setEnabled(False)
        self._visible_if_QDMObj_present.append(self.ledPlotButton)
        toolbar.addAction(self.ledPlotButton)

    def _add_fit_toolbar(self, toolbar):
        toolbar.addSeparator()

        self.fit_button = QPushButton("Fit")
        self.fit_button.setStatusTip("Fit the data")
        self.fit_button.clicked.connect(self.onFitButtonClick)
        self.fit_button.setFixedSize(50, 25)

        self._visible_if_QDMObj_present.append(self.fit_button)
        toolbar.addWidget(self.fit_button)
        self.fit_constraints_button = QPushButton("Constraints")
        self.fit_constraints_button.setStatusTip("Edit the fit constraints")
        self.fit_constraints_button.clicked.connect(
            self.onSetFitConstraintsButtonClick)
        self._visible_if_QDMObj_present.append(self.fit_constraints_button)
        toolbar.addWidget(self.fit_constraints_button)

    def _add_pixelsize_toolbar(self, toolbar):
        pixel_widget = QWidget()
        pixel_size_box = QHBoxLayout()
        pixel_size_label, self.pixel_size_select = self.get_label_box(label="Pixel Size [µm]:",
                                                                      value=1, decimals=2,
                                                                      step=1, min=1, max=99,
                                                                      callback=self.onPixelSizeChanged)
        pixel_size_box.addWidget(pixel_size_label)
        pixel_size_box.addWidget(self.pixel_size_select)
        pixel_widget.setLayout(pixel_size_box)
        toolbar.addWidget(pixel_widget)

    def _add_bin_toolbar(self, toolbar):
        bin_widget = QWidget()
        bin_box = QHBoxLayout()
        bin_factor_label, self.binfactor_select = self.get_label_box(label="Bin Factor:",
                                                                     value=1, decimals=0,
                                                                     step=1, min=1, max=32,
                                                                     callback=self.onBinFactorChanged)
        bin_box.addWidget(bin_factor_label)
        bin_box.addWidget(self.binfactor_select)
        self.bin_button = QPushButton("Bin")
        self.bin_button.setStatusTip("Bin the data")
        self.bin_button.setFixedSize(50, 25)
        self.bin_button.clicked.connect(self.onBinButtonClick)
        self._visible_if_QDMObj_present.append(self.bin_button)
        bin_box.addWidget(self.bin_button)
        bin_widget.setLayout(bin_box)
        toolbar.addWidget(bin_widget)

    def _add_gf_toolbar(self, toolbar):
        globalWidget = QWidget()
        globalBox = QHBoxLayout()
        gf_label, self.gf_select = self.get_label_box(
            "Global Fluorescence", 0, 1, 0.1, 0, 1, self.onGFChanged)
        self.gf_detect_button = QPushButton("detect")
        self.gf_detect_button.setStatusTip("Detect global fluoresence")
        self.gf_detect_button.clicked.connect(self.onGFDetectButtonClick)
        self.gf_detect_button.setFixedSize(50, 25)

        self.gf_apply_button = QPushButton("apply")
        self.gf_apply_button.setFixedSize(50, 25)
        self.gf_apply_button.setStatusTip("Apply global fluoresence")
        self.gf_apply_button.clicked.connect(self.onGFApplyButtonClick)
        globalBox.addWidget(gf_label)
        globalBox.addWidget(self.gf_select)
        globalBox.addWidget(self.gf_apply_button)
        globalBox.addWidget(self.gf_detect_button)
        self._visible_if_QDMObj_present.extend(
            [self.gf_detect_button, self.gf_apply_button])
        globalWidget.setLayout(globalBox)
        toolbar.addWidget(globalWidget)

    # MENU BAR

    def get_menu(self):
        """
        Create the menu and add the actions
        """
        self.setStatusBar(QStatusBar(self))
        menu = self.menuBar()

        # About_pyqdm
        about_pyqdm_button = QAction("&About pyqdm", self)
        about_pyqdm_button.setStatusTip("This is your button")
        about_pyqdm_button.triggered.connect(self.onAboutpyqdmButtonClick)

        # import QDM button
        import_qdmio_button = QAction("QDMio", self)
        import_qdmio_button.setStatusTip("import QDMio like files")
        import_qdmio_button.triggered.connect(self.onImportQDMioButtonClick)

        # file menu
        file_menu = menu.addMenu("&File")
        file_submenu_import = file_menu.addMenu("import")
        file_submenu_import.addAction(import_qdmio_button)
        import_qdmio_button.setShortcut(QKeySequence("Ctrl+I"))

        file_submenu_export = file_menu.addMenu("export")

        file_menu.addSeparator()
        file_menu.addAction(about_pyqdm_button)
        close_action = file_menu.addAction("&Quit", self.close)
        close_action.setShortcut(QKeySequence("Ctrl+Q"))

        # edit menu
        edit_menu = menu.addMenu("&Edit")
        setFitConstraintsButton = QAction("Set Fit Constraints", self)
        setFitConstraintsButton.setStatusTip("Set Fit Constraints")
        setFitConstraintsButton.triggered.connect(
            self.onSetFitConstraintsButtonClick)
        edit_menu.addAction(setFitConstraintsButton)

        # view menu
        view_menu = menu.addMenu("&View")
        self._add_led_plot_toolbar(view_menu)
        self._add_laser_plot_toolbar(view_menu)
        self._add_fluorescence_plot_toolbar(view_menu)

        outlierMenu = menu.addMenu("outliers")
        detectOUtlierButton = QAction("detect outliers", self)
        detectOUtlierButton.setStatusTip("detect outliers")
        detectOUtlierButton.triggered.connect(self.onDetectOutlierButtonClick)
        outlierMenu.addAction(detectOUtlierButton)

        markOutlierButton = QAction("mark outliers", self, checkable=True)
        markOutlierButton.setStatusTip("mark outliers on plots")
        markOutlierButton.triggered.connect(self.onMarkOutlierButtonClick)
        outlierMenu.addAction(markOutlierButton)

        outlierListButton = QAction("outliers list", self)
        outlierListButton.setStatusTip("outliers list")
        outlierListButton.triggered.connect(self.onOutlierListButtonClick)
        outlierMenu.addAction(outlierListButton)
        # file_submenu.addAction(import_qdmio_button)

    # fit parameters

    def onFitConstraintsButtonPress(self):
        """
        Create the widget for the fit constraints
        :return:
        """

        if self.fitConstraintsWidget is None:
            self.get_fitConstraintsWidget()
            self.fillFitWidget()

        if self.fitConstraintsWidget.isVisible():
            self.fitConstraintsWidget.hide()
        else:
            self.fitConstraintsWidget.show()

    def get_fitConstraintsWidget(self):
        self.fitConstraintsWidget = QGroupBox("Fit Constraints")
        self.fitGridLayout = QGridLayout()
        self.fitConstraintsWidget.setTitle("Fit Constraints")
        self.fitConstraintsWidget.setLayout(self.fitGridLayout)

    def fillFitWidget(self):
        self.get_fitConstraintsWidget()
        for i, (k, v) in enumerate(self.QDMObj.constraints.items()):
            self.set_FitWidget_line(i, k, v[0], v[1], v[3], v[2])

    def set_FitWidget_line(self, row, text, vmin, vmax, unit, constraint):
        self.fitConstraints[text] = [
            QLabel(text),
            QLineEdit(str(vmin)),
            QLineEdit(str(vmax)),
            QLabel(unit),
            QComboBox(),
        ]
        self.fitConstraints[text][-1].addItems(
            ['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'])
        self.fitConstraints[text][-1].setCurrentIndex(
            self.QDMObj._CONSTRAINT_TYPES[constraint])

        self.fitConstraints[text][1].returnPressed.connect(
            self.onFitWidgetItemChanged)
        self.fitConstraints[text][2].returnPressed.connect(
            self.onFitWidgetItemChanged)
        self.fitConstraints[text][-1].currentIndexChanged.connect(
            self.onFitWidgetItemChanged)
        self._set_constraint_visibility(self.QDMObj._CONSTRAINT_TYPES[constraint], self.fitConstraints[text][1],
                                        self.fitConstraints[text][2])

        # add them to the layout
        for col, item in enumerate(self.fitConstraints[text]):
            self.fitGridLayout.addWidget(item, row, col)

    def _set_constraint_visibility(self, tpe, vmaxItem, vminItem):
        if tpe in [0, 1]:
            vminItem.setEnabled(False)
        else:
            vminItem.setEnabled(True)
        if tpe in [0, 2]:
            vmaxItem.setEnabled(False)
        else:
            vmaxItem.setEnabled(True)

    def onFitWidgetItemChanged(self):
        for k, v in self.fitConstraints.items():
            self.QDMObj.set_constraints(
                k, [float(v[1].text()), float(v[2].text())], v[-1].currentIndex())
        self.fillFitWidget()

    # info widget #todo make into txt
    def get_infotableWidget(self):
        self.infotableWidget = QGroupBox("Infos")
        layout = QVBoxLayout()

        self.infoTable = QTableWidget(6, 2, self) #todo rewrite info table
        # self.infoTable.setStyleSheet('font-size: 11px; alternate-background-color: #F8F8F8;')
        self.infoTable.setStyleSheet("*{"
                                     "background-color: #F8F8F8;"
                                     "border: 0px solid #F8F8F8;"
                                     "color: #1E1E1E;"
                                     "selection-background-color: #F8F8F8;"
                                     "selection-color: #FFF;"
                                     "}"
                                     )
        self.infoTable.horizontalHeader().hide()
        self.infoTable.verticalHeader().hide()
        self.infoTable.setEnabled(False)

        layout.addWidget(self.infoTable)
        self.infotableWidget.setLayout(layout)

        self.infotableWidget.setWindowTitle("Measurement Information")
        self.infotableWidget.resize(180, 230)

        self.infotableWidget.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.infoTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.infoTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    # status bar # todo add statusbar updates
    def get_statusbar(self):
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Welcome to pyqdm")

    # loading dialog # todo fix opening dialog
    def get_loadingProgressDialog(self, title, text):
        loadingProgressDialog = QProgressDialog()
        loadingProgressDialog.setLabelText(text)
        loadingProgressDialog.setMinimum(0)
        loadingProgressDialog.setMaximum(0)
        loadingProgressDialog.setWindowTitle(title)
        loadingProgressDialog.setAutoReset(True)
        loadingProgressDialog.setAutoClose(True)
        loadingProgressDialog.setFixedSize(250, 100)
        loadingProgressDialog.setWindowModality(Qt.WindowModal)
        loadingProgressDialog.show()
        loadingProgressDialog.setValue(0)
        time.sleep(0.1)
        # loadingProgressDialog.setCancelButton()
        return loadingProgressDialog

    def pygpufitNotAvailableDialog(self):
        """
        Show a warning dialog if pyGPUfit is not installed
        """
        if not self.debug:
            dlg = pyGPUfitNotInstalledDialog()
            if dlg.exec():
                logging.debug("pyGPUfit not installed continuing without it")
            else:
                logging.debug("pyGPUfit not installed closing")
                sys.exit()
        self.statusBar().showMessage("pygpufit is not available")

    def get_label_box(self, label, value, decimals, step, min, max, callback):
        label = QLabel(label)
        selector = QDoubleSpinBox(self)
        selector.setValue(value)
        selector.setDecimals(decimals)
        selector.setSingleStep(step)
        selector.setMinimum(min)
        selector.setMaximum(max)
        selector.setKeyboardTracking(False)
        selector.valueChanged.connect(callback)
        selector.setEnabled(False)
        selector.setFixedWidth(50)
        self._visible_if_QDMObj_present.append(selector)
        return label, selector

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.LOG = logging.getLogger('pyqdm.' + self.__class__.__name__)
        self.debug = kwargs.pop('debug', False)
        self.outlier_pd = pd.DataFrame(columns=['idx', 'x', 'y'])

        if not pyqdm.pygpufit_present:
            self.pygpufitNotAvailableDialog()
        self.fluorescenceWindow = None
        self.laserWindow = None
        self.ledWindow = None
        self.main_content_figure = None
        self.qualityWindow = None

        self.QDMObj = None
        self.fitConstraintsWidget = None
        self.fitConstraints = {}

        self._current_idx = None

        self.setWindowTitle("pyqdm")
        self.resize(1200, 800)

        self.get_menu()
        self.get_toolbar()
        self.get_statusbar()

        self.init_main_content()
        self.get_infotableWidget()

        self._change_tool_visibility()

        self._data_windows = []

        if self.debug:
            self.debug_call()

    # MAIN WINDOW
    def init_main_content(self):
        self.main_content_layout = QHBoxLayout()
        self.main_label = QLabel(
            "No QDM data loaded, yet.\nNo fitting possible.")
        self.main_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_content_layout.addWidget(self.main_label)
        self.main_content_figure = None

        widget = QWidget()
        widget.setLayout(self.main_content_layout)
        self.setCentralWidget(widget)

    def update_main_content(self):
        self.LOG.debug("Updating main content")
        if self.QDMObj is None:
            self.LOG.debug("No QDM data loaded, yet.\nNo fitting possible.")
            self.init_main_content()
        elif not self.QDMObj.fitted:
            self.LOG.debug("QDM data loaded, fitting possible.")
            self.main_label.setText("No fit calculated yet.")
        else:
            self.LOG.debug("QDM data fitted, plotting results.")
            if self.main_content_figure is None:
                self._extracted_from_update_main_content_12()
            else:
                self.main_content_figure.redraw_all_plots()

    # TODO Rename this here and in `update_main_content`
    def _extracted_from_update_main_content_12(self):
        self.main_content_layout.removeWidget(self.main_label)
        self.main_content_figure = fit_window.FitWindow(
            self, self.QDMObj, parent=self)
        self._data_windows.append(self.main_content_figure)
        self.main_content_layout.addWidget(self.main_content_figure)
        self.setCentralWidget(self.main_content_figure)
        self.main_content_figure.show()

    @property
    def _need_marker_update(self):
        return [self.laserWindow, self.ledWindow, self.main_content_figure, self.qualityWindow]

    @property
    def _need_pixel_update(self):
        return [self.main_content_figure]

    def update_marker(self):
        self.LOG.debug(f"update_marker in {self._need_marker_update}")
        for window in self._need_marker_update:
            if window is not None:
                window.update_marker()

    def update_pixel(self):
        self.LOG.debug(f"Updating pixel in {self._need_pixel_update}")
        for window in self._need_pixel_update:
            if window is not None:
                window.update_pixel_lines()
                window.update_fit_lines()

    # DATA RELATED
    def set_current_idx(self, x=None, y=None, idx=None):
        if x is not None and y is not None:
            self._current_idx = self.QDMObj.odmr.rc2idx([y, x])
        elif idx is not None:
            self._current_idx = idx

    @property
    def _current_xy(self):
        """
        Returns the current xy-coordinates in data coordinates.
        :return:
        """
        return self.QDMObj.odmr.idx2rc(self._current_idx)[::-1]

    # BUTTON ACTION
    # toolbar / menu
    def onToolBarFluorescenceButtonClick(self, s):
        if self.fluorescenceWindow is None:
            self.fluorescenceWindow = FluorescenceWindow(self.QDMObj)
            self.fluorescenceWindow.show()
        elif self.fluorescenceWindow.isVisible():
            self.fluorescenceWindow.hide()
        else:
            self.fluorescenceWindow.show()

    def onToolBarLaserButtonClick(self, s):
        if self.laserWindow is None:
            self.laserWindow = SimplePlotWindow(QDMObj=self.QDMObj, title="Laser Scan", caller=self, cmap='magma',
                                                cbar=True)
            self.laserWindow.add_laser_img(
                self.laserWindow.ax, cax=self.laserWindow.cax)
            self.laserWindow.show()
        elif self.laserWindow.isVisible():
            self.laserWindow.hide()
        else:
            self.laserWindow.show()

    def onToolBarLEDButtonClick(self, s):
        if self.ledWindow is None:
            self.ledWindow = SimplePlotWindow(QDMObj=self.QDMObj, title="Reflected Light Image", caller=self,
                                              cmap='bone', cbar=False)
            self.ledWindow.add_light_img(self.ledWindow.ax)
            self.ledWindow.show()
        elif self.ledWindow.isVisible():
            self.ledWindow.hide()
        else:
            self.ledWindow.show()

    def onToolBarinfoButtonClick(self, s):
        if self.infotableWidget.isVisible():
            self.infotableWidget.hide()
        else:
            self.infotableWidget.show()

    def onImportQDMioButtonClick(self, s):
        work_directory = QFileDialog.getExistingDirectory(
            self, 'Open path', './')
        self.import_file(work_directory, dialect='QDMio')

    def onDetectOutlierButtonClick(self, s):
        self.LOG.debug("Detecting outliers")
        self.QDMObj.detect_outliers('width', method='IsolationForest')

        self.outlier_pd = pd.DataFrame(columns=['idx', 'x', 'y'])
        self.outlier_pd['x'] = self.QDMObj.outliers_xy[:, 0]
        self.outlier_pd['y'] = self.QDMObj.outliers_xy[:, 1]
        self.outlier_pd['idx'] = self.QDMObj.outliers_idx
        self.statusBar().showMessage(
            f"{self.QDMObj.outliers_idx.size:8b} Outliers detected")

    def onMarkOutlierButtonClick(self, s):
        for w in self._data_windows:
            if s:
                self.LOG.debug("Marking outliers")
                w._add_outlier_mask()
            else:
                self.LOG.debug("Removing outlier markers")
                w._toggle_outlier_mask('off')

    def onOutlierListButtonClick(self, s):
        self.LOG.debug("Showing outliers list")

        self.outlierListWidget = pandasWidget(self, self.outlier_pd)

        self.outlierListWidget.show()

    def onAboutpyqdmButtonClick(self, s):
        aboutMessageBox = QMessageBox.about(self, 'About pyqdm', 'pyqdm is written by me...',
                                            )

    def onQuickStartButtonClick(self):
        if self.QDMObj.bin_factor == 1:
            self.binfactor_select.setValue(4)
            self.onBinButtonClick()
        self.gf_select.setValue(0.2)
        self.onGFApplyButtonClick()
        self.onFitButtonClick()
        self.onDetectOutlierButtonClick(None)

    # main content
    def onPixelSizeChanged(self):
        self.LOG.debug(
            f"Pixel Size changed to {self.pixel_size_select.value()}")

    def onBinFactorChanged(self):
        self.LOG.debug(
            f"Bin Factor changed to {self.binfactor_select.value()}")

    def onBinButtonClick(self):
        self.LOG.debug("Bin Button clicked")
        self.QDMObj.bin_data(self.binfactor_select.value())
        if self.QDMObj.fitted:
            self.QDMObj.fit_ODMR()
        self._current_idx = self.QDMObj.odmr.get_most_divergent_from_mean()[-1]
        self.update_main_content()
        self.fillFitWidget()

    def onFitButtonClick(self):
        self.QDMObj.fit_ODMR()
        self.update_main_content()
        self.update_marker()
        self.update_pixel()
        self._fill_infoTable()
        self.fillFitWidget()

    def onGFDetectButtonClick(self):
        self.gf_window = GlobalFluorescenceWindow(self, self.QDMObj)
        self.LOG.debug("GF Button clicked")

    def onGFApplyButtonClick(self):
        self.LOG.debug("GF Apply Button clicked")
        self.QDMObj.correct_glob_fluorescence(self.gf_select.value())
        if not self.debug:
            GFAppliedWindow(self.gf_select.value())

    def onGFChanged(self):
        self.LOG.debug(f"GF changed to {self.gf_select.value()}")

    def onSetFitConstraintsButtonClick(self):
        self.onFitConstraintsButtonPress()

    # additional functions
    def set_pixel_size_from_QDMObj(self):
        if self.QDMObj is not None:
            with contextlib.suppress(AttributeError):
                self.pixel_size_select.setValue(self.QDMObj.pixel_size)

    def get_infos(self):
        return [self.QDMObj.ODMRobj.scan_dimensions,
                self.QDMObj.ODMRobj.n_pol,
                self.QDMObj.ODMRobj.n_frange,
                self.QDMObj.ODMRobj.data.size,
                self.QDMObj.ODMRobj.n_freqs,
                self.QDMObj.bin_factor,
                ]

    def _change_tool_visibility(self):
        for action in self._visible_if_QDMObj_present:
            action.setEnabled(self.QDMObj is not None)

    def _fill_infoTable(self, infos=None):
        entries = ["Image dimensions", "Field directions", "Frequency ranges", "# pixels", "# frequencies",
                   'bin factor']
        for i, e in enumerate(entries):
            self.infoTable.setItem(i, 0, QTableWidgetItem(e))

        header = self.infoTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)

        if infos is not None:
            for i, n in enumerate(infos):
                self.infoTable.setItem(i, 1, QTableWidgetItem(str(n)))
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.infoTable.resizeColumnsToContents()
        self.infoTable.resizeRowsToContents()
        self.infoTable.resize(180, 230)

    def import_file(self, workDirectory, dialect='QDMio'):
        loadingProgressDialog = self.get_loadingProgressDialog("Importing QDMio like files",
                                                               "Importing QDMio like files, please wait.")
        self.workDirectory = Path(workDirectory)
        self.init_main_content()

        # time.sleep(1)

        if dialect == 'QDMio':
            self.QDMObj = self.import_QDMio(workDirectory)

        if self.QDMObj is not None:
            self._init_with_new_QDMObj()
        else:
            QMessageBox.warning(self, "Import Error", "Could not import file.")
        loadingProgressDialog.close()

    def _init_with_new_QDMObj(self):
        """
        Initializes the app for a new QDMobj.
        """
        self._change_tool_visibility()
        self._fill_infoTable(self.get_infos())
        # in data coordinates
        self.set_current_idx(idx=self.QDMObj.odmr.get_most_divergent_from_mean()[-1])
        self.fileImported()
        self.update_main_content()

    # IMPORT FUNCTIONS
    def import_QDMio(self, work_directory):
        self.statusBar().showMessage(
            f"Importing QDMio like files from {work_directory}")

        self.workDirectory = Path(work_directory)
        try:
            qdm_obj = QDM.from_QDMio(self.workDirectory)
            self.statusBar().showMessage(
                f"Successfully imported QDMio like files from {self.workDirectory}")

            return qdm_obj
        except CantImportError:
            self.statusBar().showMessage(
                f"Cant import QDMio like files from {self.workDirectory}")

            return

    def fileImported(self):
        self.statusBar().showMessage(
            f"Successfully imported QDM files from {self.workDirectory}")

        self._change_tool_visibility()
        self._fill_infoTable(self.get_infos())
        self.set_pixel_size_from_QDMObj()
        self.binfactor_select.setValue(self.QDMObj.bin_factor)
        self.main_label.setText("No fits calculated yet.")

    def debug_call(self):
        self.import_file(r'C:\Users\micha\Desktop\diamond_testing\FOV18x')
        self.onQuickStartButtonClick()
        self.onFitButtonClick()


class pandasWidget(QGroupBox):
    def __init__(self, caller, pd, parent=None, title='Outliers', **kwargs):
        super().__init__(parent)
        vLayout = QVBoxLayout(self)
        self.caller = caller
        self.pandasTv = QTableView(self)

        vLayout.addWidget(self.pandasTv)
        self.pandasTv.setSortingEnabled(True)

        model = pandasModel(pd)
        self.pandasTv.setModel(model)
        self.pandasTv.resizeColumnsToContents()
        self.pandasTv.setSelectionBehavior(QAbstractItemView.SelectRows)
        selection = self.pandasTv.selectionModel()
        selection.selectionChanged.connect(self.handleSelectionChanged)
        self.resize(150, 200)
        self.setContentsMargins(0, 0, 0, 0)

    def handleSelectionChanged(self, selected, deselected):
        for index in self.pandasTv.selectionModel().selectedRows():
            self.caller.set_current_idx(idx=int(index.data()))
            self.caller.update_marker()
            self.caller.update_pixel()


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


def main(**kwargs):
    app = QApplication(sys.argv)
    mainwindow = PyQDMMainWindow(**kwargs)
    mainwindow.show()

    center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
    geo = mainwindow.frameGeometry()
    geo.moveCenter(center)
    mainwindow.move(geo.topLeft())

    app.exec()


if __name__ == '__main__':
    main(debug=False)