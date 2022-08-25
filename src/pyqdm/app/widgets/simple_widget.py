from pyqdm.app.canvas import SimpleCanvas
from pyqdm.app.widgets.qdm_widget import PyQdmWindow


class SimpleWidget(PyQdmWindow):
    def __init__(self, dtype, *args, **kwargs):
        canvas = SimpleCanvas(dtype=dtype)

        super().__init__(canvas=canvas, *args, **kwargs)
        self.set_main_window()
        if dtype == "laser":
            self.canvas.add_laser(self.qdm.laser, self.qdm.data_shape)
            self.canvas.fig.subplots_adjust(top=0.97, bottom=0.124, left=0.049, right=0.951, hspace=0.2, wspace=0.2)
            self.setWindowTitle("Laser scan")
        elif dtype == "light":
            self.canvas.add_light(self.qdm.light, self.qdm.data_shape)
            self.setWindowTitle("Reflected light")
        else:
            raise ValueError(f"dtype {dtype} not recognized")

        self.canvas.add_scalebars(self.qdm.pixel_size)
        self.update_clims()
        self.update_marker()
        self.canvas.set_img()
        self.canvas.draw()
