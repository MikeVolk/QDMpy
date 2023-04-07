import sys
import matplotlib
from PySide6 import QtCore
import PySide6.QtWidgets as QtW
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector


class MainWindow(QtW.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('MyWindow')
        self._main = QtW.QWidget()
        self.setCentralWidget(self._main)

        # Set canvas properties
        self.fig = plt.Figure(figsize=(5,5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(1,1,1)
        self.canvas.draw()

        self.rs = RectangleSelector(self.ax, self.line_select_callback,
                                                drawtype='box', useblit=True,
                                                button=[1, 3],  # don't use middle button
                                                minspanx=5, minspany=5,
                                                spancoords='pixels',
                                                interactive=True,
                                                rectprops=dict(facecolor='none', edgecolor = 'black', alpha=0.5, fill=False))

        # set Qlayout properties and show window
        self.gridLayout = QtW.QGridLayout(self._main)
        self.gridLayout.addWidget(self.canvas)
        self._add_button = QtW.QPushButton('Add')
        self.gridLayout.addWidget(self._add_button, 1, 0)
        self._add_button.clicked.connect(self.on_add_button_clicked)
        self.setLayout(self.gridLayout)
        self.show()

        # connect mouse events to canvas
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_add_button_clicked(self):
        print('Add button clicked')
        print("curren xy: ", self.rs.extents)
    def on_click(self, event):
        if event.button == 1 or event.button == 3 and not self.rs.active:
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None: app = QtW.QApplication(sys.argv)
    win = MainWindow()
    app.aboutToQuit.connect(app.deleteLater)
    app.exec_()