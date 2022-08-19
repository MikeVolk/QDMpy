from numpy import testing as npt

from pyqdm.core import fit


class Fit_object:
    def __init__(self):
        pass

    def test_fit_odmr(self):
        models = fit.MODELS

        for model in models:
            data, freq, true_parameter = fit.make_dummy_data(model=model.__name__, n_freq=1000)
            fit = fit.Fit(data=data, freq=freq, model=model.__name__)
            npt.assert_array_almost_equal(fit._parameter, true_parameter)
