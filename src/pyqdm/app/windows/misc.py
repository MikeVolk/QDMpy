from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QMessageBox,
    QTableView,
    QVBoxLayout,
)

from pyqdm.app.models import PandasModel


def gf_applied_window(value):
    dlg = QMessageBox()
    dlg.setWindowTitle("Global fluorescence correction")
    dlg.setText(f"Global fluorescence correction of {value} applied")
    dlg.setStandardButtons(QMessageBox.Ok)
    dlg.exec()


class PandasWidget(QGroupBox):
    def __init__(self, caller, pdd, title="Outliers", parent=None):
        super().__init__(parent)
        v_layout = QVBoxLayout(self)
        self.setTitle(title)
        self.caller = caller
        self.pandasTv = QTableView(self)

        v_layout.addWidget(self.pandasTv)
        self.pandasTv.setSortingEnabled(True)

        model = PandasModel(pdd)
        self.pandasTv.setModel(model)
        self.pandasTv.resizeColumnsToContents()
        self.pandasTv.setSelectionBehavior(QAbstractItemView.SelectRows)
        selection = self.pandasTv.selectionModel()
        selection.selectionChanged.connect(self.handle_selection_changed)
        self.resize(150, 200)
        self.setContentsMargins(0, 0, 0, 0)

    def handle_selection_changed(self):
        for index in self.pandasTv.selectionModel().selectedRows():
            self.caller.set_current_idx(idx=int(index.data()))
            self.caller.update_marker()
            self.caller.update_pixel()
