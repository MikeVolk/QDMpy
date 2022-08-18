from PySide6.QtWidgets import (
    QMessageBox
)

def GFAppliedWindow(value):
    dlg = QMessageBox()
    dlg.setWindowTitle("Global fluorescence correction")
    dlg.setText(f"Global fluorescence correction of {value} applied")
    dlg.setStandardButtons(QMessageBox.Ok)
    button = dlg.exec()
