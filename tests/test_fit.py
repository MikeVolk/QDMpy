import unittest

from numpy import testing as npt

from QDMpy.core import fit


class Fit_object(unittest.TestCase):
    def test_fit_odmr(self):
        models = fit.MODELS

        for model in models:
            data, freq, true_parameter = fit.make_dummy_data(model=model, n_freq=1000)
            fit_instance = fit.Fit(data=data, frequencies=freq, model=model)
            fit_instance.fit_odmr()
            print(model)
            npt.assert_equal(fit_instance._parameter.shape, true_parameter.shape)
            npt.assert_allclose(
                true_parameter[:, :, :, 0], fit_instance._parameter[:, :, :, 0], 1e-6
            )
            npt.assert_allclose(
                true_parameter[:, :, :, 1], fit_instance._parameter[:, :, :, 1], 1e-3
            )
            npt.assert_allclose(
                true_parameter[:, :, :, 2], fit_instance._parameter[:, :, :, 2], 1e-3
            )
            npt.assert_allclose(
                true_parameter[:, :, :, -1],
                fit_instance._parameter[:, :, :, -1],
                atol=1e-4,
            )
