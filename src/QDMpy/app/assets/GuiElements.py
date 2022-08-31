from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QLabel, QDoubleSpinBox


def LabeledDoubleSpinBox(label, value, decimals, step, vmin, vmax, callback):
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
    label = QLabel(label)
    selector = QDoubleSpinBox()
    selector.setDecimals(decimals)
    selector.setSingleStep(step)
    selector.setMinimum(vmin)
    selector.setMaximum(vmax)
    selector.setValue(value)
    selector.setKeyboardTracking(False)
    selector.valueChanged.connect(callback)
    return label, selector
