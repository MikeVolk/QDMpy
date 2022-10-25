from QDMpy.app.canvas import SimpleCanvas
from QDMpy.app.widgets.qdm_widget import QDMWidget


class SimpleWidget(QDMWidget):
    def __init__(self, dtype, set_main_window=True, *args, **kwargs):
        canvas = SimpleCanvas(dtype=dtype)

        super().__init__(canvas=canvas, *args, **kwargs)

        if dtype == "laser":
            self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)
            self.canvas.fig.subplots_adjust(
                top=0.97, bottom=0.124, left=0.049, right=0.951, hspace=0.2, wspace=0.2
            )
            self.setWindowTitle("Laser scan")
        elif dtype == "light":
            self.canvas.add_light(self.qdm.light, self.qdm.data_shape)
            self.setWindowTitle("Reflected light")
        elif dtype == "outlier":
            self.canvas.add_data(self.qdm.b111[0], self.qdm.data_shape)
            self.setWindowTitle("Reflected light")
        else:
            raise ValueError(f"dtype {dtype} not recognized")

        if set_main_window:
            self.set_main_window()
        self.update_clims()
        self.add_scalebars()
        self.canvas.draw_idle()
