import numpy as np
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
from pyqdm_plot_window import pyqdmWindow
from canvas import QualityCanvas
from matplotlib import pyplot as plt, colors


class QualityWindow(pyqdmWindow):
    TITLES = {'center':'f$_{\mathrm{resonance}}$',
              'contrast':'$\sigma$contrast',
              'contrast_0':'contrast$_0$',
              'contrast_1':'contrast$_1$',
              'contrast_2':'contrast$_2$',
              'contrast_3':'contrast$_3$',
              'width' : 'width',
              'chi_squared':'$\chi^2$'
              }
    UNITS = {'center':'f [GHz]',
              'contrast':'[a.u.]',
              'contrast_0':'[%]',
              'contrast_1':'[%]',
              'contrast_2':'[%]',
              'contrast_3':'[%]',
              'width' : '[GHz]',
              'chi_squared':'[a.u.]'
              }
    def __init__(self, caller, *args, **kwargs):
        canvas = QualityCanvas(self, width=12, height=12, dpi=100)
        super(QualityWindow, self).__init__(caller, canvas, *args, **kwargs)
        self._add_dtypeSelector(self.mainToolbar)
        self._init_lines()
        self.init_plots()

    def _add_dtypeSelector(self, toolbar):
        dtypeSelectWidget = QWidget()
        parameterBox = QHBoxLayout()
        parameterLabel = QLabel('param: ')
        self.dataSelect = QComboBox()
        self.dataSelect.addItems(self.QDMObj._fitting_params_unique + ['contrast', 'chi_squared'])
        self.dataSelect.setCurrentText('chi_squared')
        self.dataSelect.currentTextChanged.connect(self.update_img_plots)
        parameterBox.addWidget(parameterLabel)
        parameterBox.addWidget(self.dataSelect)
        dtypeSelectWidget.setLayout(parameterBox)
        toolbar.addWidget(dtypeSelectWidget)

    def init_plots(self):
        self.LOG.debug('init_plots')
        d = self.QDMObj._chi_squares.reshape(self.QDMObj.ODMRobj.n_pol, self.QDMObj.ODMRobj.n_frange,
                                             *self.QDMObj.ODMRobj.scan_dimensions)
        vmin, vmax = d.min(), d.max()

        for f in range(self.QDMObj.ODMRobj.n_frange):
            for p in range(self.QDMObj.ODMRobj.n_pol):
                ax = self.canvas.ax[p][f]
                img = ax.imshow(d[p, f], cmap='viridis', interpolation='none',
                                origin='lower', vmin=vmin, vmax=vmax)
                plt.colorbar(img, cax=self.canvas.caxes[p][f])
        self.canvas.fig.suptitle('$\chi^2$')
        self.update_marker()

    def updateClimS(self):
        for i, ax in enumerate(self.canvas.ax.flatten()):
            im = ax.images[0]
            d = im.get_array()
            vmin, vmax = np.min(d), np.max(d)

            if self.fixClimCheckBox.isChecked():
                vmin, vmax = np.percentile(d, [100 - self.cLimSelector.value(), self.cLimSelector.value()])

            if not (vmin < 0 < vmax):
                vcenter = (vmin + vmax) / 2
            else:
                vcenter = 0

            im.set(norm=colors.Normalize(vmin=vmin, vmax=vmax))

            cax = self.canvas.caxes.flatten()[i]
            cax.clear()
            cax.set_axes_locator(self.canvas.original_cax_locator.flatten()[i])

            im_ratio = d.shape[0] / d.shape[1]
            plt.colorbar(im, cax=cax,  # fraction=0.047,# * im_ratio, pad=0.01,
                         extend='both' if self.fixClimCheckBox.isChecked() else 'neither',
                         label=self.UNITS[self.dataSelect.currentText()])
            self.canvas.draw()

    def update_img_plots(self):
        if self.dataSelect.currentText() == 'chi_squared':
            data = self.QDMObj._chi_squares.reshape(self.QDMObj.ODMRobj.n_pol, self.QDMObj.ODMRobj.n_frange,
                                                    *self.QDMObj.ODMRobj.scan_dimensions)
        else:
            data = self.QDMObj.get_param(self.dataSelect.currentText())

        if self.dataSelect.currentText() == 'contrast':
            data = np.sum(data, axis=2)

        for f in range(self.QDMObj.ODMRobj.n_frange):
            for p in range(self.QDMObj.ODMRobj.n_pol):
                im = self.canvas.ax[p][f].images[0]
                d = data[p, f]
                im.set_data(d)

        self.canvas.fig.suptitle(self.TITLES[self.dataSelect.currentText()])
        self.updateClimS()

    def closeEvent(self, event):
        self.LOG.debug('closeEvent')
        self.caller.qualityWindow = None
        self.close()
