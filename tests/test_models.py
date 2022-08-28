import numpy.testing as npt
import numpy as np
from skimage.transform import downscale_local_mean
import itertools
import unittest
from QDMpy.app.models import Pix


class Pix_test(unittest.TestCase):
    def setUp(self):
        self.img_array = np.random.normal(0, 100, (6, 4))
        self.data_array = downscale_local_mean(self.img_array, (2, 2))

    def test_several_instances(self):
        p0 = Pix()
        p0.data_shape = self.data_array.shape
        p0.img_shape = self.img_array.shape
        p0.set_idx(x=5, y=5)

        p1 = Pix()
        npt.assert_array_equal(p0.data_shape, p1.data_shape)
        npt.assert_array_equal(p0.img_shape, p1.img_shape)
        npt.assert_array_equal(p0.idx, p1.idx)

        p1.set_idx(x=5, y=5)
        npt.assert_equal(p0.idx, p1.idx)

    def test_set_idx(self):

        p0 = Pix()
        p0.data_shape = self.data_array.shape
        p0.img_shape = self.img_array.shape

        for i in range(self.data_array.size):
            data_y = np.unravel_index(i, self.data_array.shape)[0]
            data_x = np.unravel_index(i, self.data_array.shape)[1]
            p0.set_idx(idx=i, ref="data")
            self.assertEqual(i, p0.data_idx)
            self.assertEqual(data_x, p0.data_x)
            self.assertEqual(data_y, p0.data_y)

        for i in range(self.img_array.size):
            img_y = np.unravel_index(i, self.img_array.shape)[0]
            img_x = np.unravel_index(i, self.img_array.shape)[1]
            p0.set_idx(idx=i, ref="img")
            self.assertEqual(i, p0.idx)
            self.assertEqual(img_x, p0.x)
            self.assertEqual(img_y, p0.y)

        for y, x in itertools.product(np.arange(self.data_array.shape[0], self.data_array.shape[1])):
            p0.set_idx(x=x, y=y, ref="data")
            self.assertEqual(x, p0.data_x)
            self.assertEqual(y, p0.data_y)
            self.assertEqual(np.ravel_multi_index((y, x), self.data_array.shape), p0.data_idx)

        for y, x in itertools.product(np.arange(self.img_array.shape[0], self.img_array.shape[1])):
            p0.set_idx(x=x, y=y, ref="img")
            self.assertEqual(x, p0.x)
            self.assertEqual(y, p0.y)
            self.assertEqual(np.ravel_multi_index((y, x), self.img_array.shape), p0.idx)

        with self.assertRaises(ValueError):
            p0.set_idx(x=data_x, y=data_y, ref="notimplemented")
        with self.assertRaises(ValueError):
            p0.set_idx()
