import itertools

import numpy as np
from PySide6.QtCore import QAbstractTableModel, Qt

from pyqdm.utils import rc2idx, idx2rc


class Pix:
    """
    Pixel Singleton class. This class is used to store the current pixel index and reference.
    Also includes methods to convert between index and xy coordinates as well as between data and image.
    """
    class __Pix:
        def __init__(self):
            self._idx = None  # current index in img coordinates
            self.data_shape = None
            self.img_shape = None

        def __str__(self):
            return f'current index is: {self._idx} with scan dimensions: ' \
                   f'data {self.data_shape}; ' \
                   f'img: {self.img_shape}'

        def set_idx(self, x=None, y=None, idx=None, ref='data'):
            if ref == 'data':
                scan_dimensions = self.data_shape
            elif ref == 'img':
                scan_dimensions = self.img_shape
            else:
                raise ValueError(f'{ref} is not a valid reference')

            if x is not None and y is not None:
                self._idx = rc2idx([y, x], shape=scan_dimensions)
            elif idx is not None:
                if ref == 'data':
                    self._idx = idx
                elif ref == 'img':
                    self._idx = idx * self.bin_factor
            else:
                raise ValueError('x and y or idx must be specified')

        @property
        def bin_factor(self):
            return  self.img_shape[0] / self.data_shape[0]

        @property
        def data_idx(self):
            r, c = idx2rc(self._idx, self.img_shape)
            return rc2idx([(r + self.bin_factor / 2) / self.bin_factor,
                           (c + self.bin_factor / 2) / self.bin_factor],
                          self.data_shape)

        @property
        def binned_pixel_idx(self):
            """
            Return the indices of the binned pixels. Reference is the image index.
            :return: numpy.ndarray
            """
            rc_idx = list(
                itertools.product(
                    np.arange(self.data_y * self.bin_factor, (self.data_y + 1) * self.bin_factor),
                    np.arange(self.data_x * self.bin_factor, (self.data_x + 1) * self.bin_factor),
                ))
            return rc2idx(np.array(rc_idx).T, self.img_shape)

        @property
        def idx(self):
            return self._idx

        @property
        def x(self):
            return idx2rc(self._idx, self.img_shape)[1]

        @property
        def y(self):
            return idx2rc(self._idx, self.img_shape)[0]

        @property
        def data_x(self):
            return idx2rc(self.data_idx, self.data_shape)[1]

        @property
        def data_y(self):
            return idx2rc(self.data_idx, self.data_shape)[0]

    instance = None

    def __new__(cls):
        if Pix.instance is None:
            Pix.instance = Pix.__Pix()
        return Pix.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)

    def set_idx(self, x=None, y=None, idx=None, ref='data'):
        self.instance.set_idx(x=x, y=y, idx=idx, ref=ref)


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
