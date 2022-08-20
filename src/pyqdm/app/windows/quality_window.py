import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from pyqdm.app.canvas import QualityCanvas
from pyqdm.app.windows.pyqdm_plot_window import PyQdmWindow


class QualityWindow(PyQdmWindow):
    TITLES = {
        "center": r"f$_{\mathrm{resonance}}$",
        "contrast": r"$\sigma$contrast",
        "contrast_0": "contrast$_0$",
        "contrast_1": "contrast$_1$",
        "contrast_2": "contrast$_2$",
        "contrast_3": "contrast$_3$",
        "width": "width",
        "chi_squared": r"$\chi^2$",
    }
    UNITS = {
        "center": "f [GHz]",
        "contrast": "[a.u.]",
        "contrast_0": "[%]",
        "contrast_1": "[%]",
        "contrast_2": "[%]",
        "contrast_3": "[%]",
        "width": "[GHz]",
        "chi_squared": "[a.u.]",
    }

    def __init__(self, caller, *args, **kwargs):
        canvas = QualityCanvas(self, width=12, height=12, dpi=100)
        super().__init__(caller, canvas, *args, **kwargs)
        self._add_dtypeSelector(self.mainToolbar)
        self._init_lines()
        self.init_plots()

    def _add_dtypeSelector(self, toolbar):
        dtypeSelectWidget = QWidget()
        parameterBox = QHBoxLayout()
        parameterLabel = QLabel("param: ")
        self.dataSelect = QComboBox()
        self.dataSelect.addItems(self.qdm._fitting_params_unique + ["contrast", "chi_squared"])
        self.dataSelect.setCurrentText("chi_squared")
        self.dataSelect.currentTextChanged.connect(self.update_img_plots)
        parameterBox.addWidget(parameterLabel)
        parameterBox.addWidget(self.dataSelect)
        dtypeSelectWidget.setLayout(parameterBox)
        toolbar.addWidget(dtypeSelectWidget)

    def init_plots(self):
        self.LOG.debug("init_plots")
        d = self.qdm._chi_squares.reshape(self.qdm.odmr.n_pol, self.qdm.odmr.n_frange, *self.qdm.odmr.scan_dimensions)
        vmin, vmax = d.min(), d.max()

        for f in range(self.qdm.odmr.n_frange):
            for p in range(self.qdm.odmr.n_pol):
                ax = self.canvas.ax[p][f]
                img = ax.imshow(d[p, f], cmap="viridis", interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
                plt.colorbar(img, cax=self.canvas.caxes[p][f])
        self.canvas.fig.suptitle(r"$\chi^2$")
        self.update_marker()

    def updateClimS(self):
        for i, ax in enumerate(self.canvas.ax.flatten()):
            im = ax.images[0]
            d = im.get_array()
            vmin, vmax = np.min(d), np.max(d)

            if self.fix_clim_check_box.isChecked():
                vmin, vmax = np.percentile(d, [100 - self.cLimSelector.value(), self.cLimSelector.value()])

            im.set(norm=colors.Normalize(vmin=vmin, vmax=vmax))

            cax = self.canvas.caxes.flatten()[i]
            cax.clear()
            cax.set_axes_locator(self.canvas.original_cax_locator.flatten()[i])

            plt.colorbar(
                im,
                cax=cax,
                extend="both" if self.fix_clim_check_box.isChecked() else "neither",
                label=self.UNITS[self.dataSelect.currentText()],
            )
            self.canvas.draw()

    def update_img_plots(self):
        if self.dataSelect.currentText() == "chi_squared":
            data = self.qdm._chi_squares.reshape(
                self.qdm.odmr.n_pol, self.qdm.odmr.n_frange, *self.qdm.odmr.scan_dimensions
            )
        else:
            data = self.qdm.get_param(self.dataSelect.currentText())

        if self.dataSelect.currentText() == "contrast":
            data = np.sum(data, axis=2)

        for f in range(self.qdm.odmr.n_frange):
            for p in range(self.qdm.odmr.n_pol):
                im = self.canvas.ax[p][f].images[0]
                d = data[p, f]
                im.set_data(d)

        self.canvas.fig.suptitle(self.TITLES[self.dataSelect.currentText()])
        self.updateClimS()

    def closeEvent(self, event):
        self.LOG.debug("closeEvent")
        self.caller.qualityWindow = None
        self.close()
