import matplotlib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider

from QDMpy.app.canvas import GlobalFluorescenceCanvas
from QDMpy.app.widgets.misc import gf_applied_window
from QDMpy.app.widgets.qdm_widget import QDMWidget

matplotlib.rcParams.update(
    {  # 'font.size': 8,
        # 'axes.labelsize': 8,
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    }
)


class GlobalWidget(QDMWidget):
    def add_odmr(self, mean=True):
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_uncorrected_odmr(),
            uncorrected=self.get_current_odmr(),
            corrected=self.get_corrected_odmr(self.global_slider.value()),
            mean=self.qdm.odmr.mean_odmr,
        )

    def __init__(self, qdm_instance, *args, **kwargs):
        canvas = GlobalFluorescenceCanvas()
        super().__init__(canvas=canvas, *args, **kwargs)
        self.LOG.debug("GlobalFluorescenceWindow.__init__")
        self.setWindowTitle("Global Fluorescence")

        # global fluorescence slider
        self.global_label = QLabel(f"Global Fluorescence: {self.qdm.global_factor:.2f}")
        self.global_slider = QSlider()
        self.global_slider.setValue(self.caller.gf_select.value())
        self.global_slider.setRange(0, 100)
        self.global_slider.setOrientation(Qt.Horizontal)
        self.global_slider.valueChanged.connect(self.on_global_slider_change)
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.apply_global_factor)

        # finish main layout
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.global_label)
        horizontal_layout.addWidget(self.global_slider)
        horizontal_layout.addWidget(self.applyButton)
        self.mainVerticalLayout.addLayout(horizontal_layout)
        self.set_main_window()

        # add plotting elements
        self.add_odmr(mean=True)
        self.add_light()
        self.add_laser()
        self.add_scalebars()
        self.update_marker()
        self.canvas.set_img()
        self.update_clims()
        self.canvas.draw_idle()

    def on_global_slider_change(self):
        self.global_label.setText(
            f"Global Fluorescence: {self.global_slider.value() / 100:.2f}"
        )
        self.update_odmr()
        self.canvas.draw_idle()

    def update_odmr(self):
        """
        Update the marker position on the image plots.
        """
        self.canvas.update_odmr(
            freq=self.qdm.odmr.f_ghz,
            data=self.get_uncorrected_odmr(),
            uncorrected=self.get_current_odmr(),
            corrected=self.get_corrected_odmr(self.global_slider.value()),
            mean=self.qdm.odmr.mean_odmr,
        )
        self.canvas.update_odmr_lims(
            [
                self.qdm.odmr.mean_odmr,
                self.get_uncorrected_odmr(),
                self.get_current_odmr(),
                self.get_corrected_odmr(self.global_slider.value()),
            ]
        )

    def apply_global_factor(self):
        self.LOG.debug(f"applying global factor {self.global_slider.value() / 100:.2f}")
        self.qdm.odmr.correct_glob_fluorescence(self.global_slider.value() / 100)
        gf_applied_window(self.global_slider.value() / 100)
        self.caller.gf_select.setValue(self.global_slider.value() / 100)
        self.update_odmr()
        self.close()
