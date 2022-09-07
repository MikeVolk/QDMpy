import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget
from matplotlib import colors
from matplotlib import pyplot as plt

from QDMpy.app.canvas import QualityCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget

AU = "[a.u.]"
PERCENT = "[%]"
GHZ = "[GHz]"


class QualityWidget(QDMWidget):
    TITLES = {
        "center": r"f$_{\mathrm{resonance}}$",
        "contrast": r"$\Sigma$contrast",
        "contrast_0": "contrast$_0$",
        "contrast_1": "contrast$_1$",
        "contrast_2": "contrast$_2$",
        "contrast_3": "contrast$_3$",
        "offset": "offset",
        "width": "width",
        "chi_squared": r"$\chi^2$",
    }
    UNITS = {
        "center": GHZ,
        "contrast": PERCENT,
        "contrast_0": PERCENT,
        "contrast_1": PERCENT,
        "contrast_2": PERCENT,
        "contrast_3": PERCENT,
        "offset": AU,
        "width": GHZ,
        "chi_squared": AU,
    }

    def __init__(self, *args, **kwargs):
        canvas = QualityCanvas(self, width=12, height=12, dpi=100)
        super().__init__(canvas, *args, **kwargs)
        self._add_dtype_selector(self.main_toolbar)
        self._init_lines()
        self.init_plots()
        self.resize(1000, 700)
        self.setWindowTitle("Quality plots")

    def _add_dtype_selector(self, toolbar):
        dtype_select_widget = QWidget()
        parameter_box = QHBoxLayout()
        parameter_label = QLabel("param: ")
        self.data_select = QComboBox()
        self.data_select.addItems(self.qdm.fit.model_params_unique + ["contrast", "chi_squared"])
        self.data_select.setCurrentText("chi_squared")
        self.data_select.currentTextChanged.connect(self.update_img_plots)
        parameter_box.addWidget(parameter_label)
        parameter_box.addWidget(self.data_select)
        dtype_select_widget.setLayout(parameter_box)
        toolbar.addWidget(dtype_select_widget)

    def init_plots(self):
        self.LOG.debug("init_plots")
        d = self.qdm.get_param("chi2")
        vmin, vmax = d.min(), d.max()

        for f in range(self.qdm.odmr.n_frange):
            for p in range(self.qdm.odmr.n_pol):
                ax = self.canvas.ax[p][f]
                img = ax.imshow(
                    d[p, f],
                    cmap="viridis",
                    interpolation="none",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(img, cax=self.canvas.caxes[p][f])
        self.canvas.fig.suptitle(r"$\chi^2$")
        self.update_marker()

    def update_clims(self):
        for i, ax in enumerate(self.canvas.ax.flatten()):
            im = ax.images[0]
            d = im.get_array()
            vmin, vmax = np.min(d), np.max(d)

            if self.fix_clim_check_box.isChecked():
                vmin, vmax = np.percentile(d, [100 - self.clims_selector.value(), self.clims_selector.value()])

            im.set(norm=colors.Normalize(vmin=vmin, vmax=vmax))

            cax = self.canvas.caxes.flatten()[i]
            cax.clear()
            cax.set_axes_locator(self.canvas.original_cax_locator.flatten()[i])

            plt.colorbar(
                im,
                cax=cax,
                extend="both" if self.fix_clim_check_box.isChecked() else "neither",
                label=self.UNITS[self.data_select.currentText()],
            )
            self.canvas.draw()

    def update_img_plots(self):

        data = self.qdm.get_param(self.data_select.currentText())

        if self.data_select.currentText() == "contrast":
            data = np.sum(data, axis=2)

        for f in range(self.qdm.odmr.n_frange):
            for p in range(self.qdm.odmr.n_pol):
                im = self.canvas.ax[p][f].images[0]
                d = data[p, f]
                im.set_data(d)

        self.canvas.fig.suptitle(self.TITLES[self.data_select.currentText()])
        self.update_clims()

    def closeEvent(self, event):
        self.LOG.debug("closeEvent")
        self.caller.qualityWindow = None
        self.close()
