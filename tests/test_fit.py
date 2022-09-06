import unittest
from QDMpy.core import fit, models
import numpy.testing as npt


class test_Fit_class(unittest.TestCase):
    def setUp(self) -> None:
        self.model = 'ESR14N'
        self.m_all, self.f_ghz, self.p_all = fit.make_dummy_data(model=self.model)
        self.fit_instance = fit.Fit(data=self.m_all, frequencies=self.f_ghz*1e9, model='ESR14N')

    # def test_fit_odmr(self):
    #     models = [m['model_func'] for m in models.IMPLEMENTED]
    #
    #     for model in models:
    #         data, freq, true_parameter = fit.make_dummy_data(model=model, n_freq=1000)
    #         fit_instance = fit.Fit(data=data, frequencies=freq, model=model)
    #         fit_instance.fit_odmr()
    #         print(model)
    #         npt.assert_equal(fit_instance._fit_results.shape, true_parameter.shape)
    #         npt.assert_allclose(
    #             true_parameter[:, :, :, 0], fit_instance._fit_results[:, :, :, 0], 1e-6
    #         )
    #         npt.assert_allclose(
    #             true_parameter[:, :, :, 1], fit_instance._fit_results[:, :, :, 1], 1e-3
    #         )
    #         npt.assert_allclose(
    #             true_parameter[:, :, :, 2], fit_instance._fit_results[:, :, :, 2], 1e-3
    #         )
    #         npt.assert_allclose(
    #             true_parameter[:, :, :, -1],
    #             fit_instance._fit_results[:, :, :, -1],
    #             atol=1e-4,
    #         )

    def test_model_params(self):
        self.assertEqual(sorted(self.fit_instance.model_params), sorted(['center', 'contrast','contrast','contrast', 'offset', 'width']))


    def test__reset_fit(self):
        assert False


    def test_fitted(self):
        assert False


    def test_data(self):
        npt.assert_allclose(self.fit_instance.data, self.m_all)
        npt.assert_allclose(self.fit_instance._data, self.m_all)


    def test_data_setter(self):
        self.fit_instance.data = self.m_all + 0.1
        npt.assert_allclose(self.fit_instance.data, self.m_all + 0.1)
        npt.assert_allclose(self.fit_instance._data, self.m_all + 0.1)


    def test_model(self):
        self.assertEqual(self.fit_instance.model, self.model)

    def test_model_func(self):
        self.assertEqual(self.fit_instance.model_func, models.IMPLEMENTED[self.model]['func'])


    def test_model_setter(self):
        model = 'ESR15N'
        self.fit_instance.model = model
        self.assertEqual(self.fit_instance.model, model)
        self.assertEqual(self.fit_instance._model, model)
        self.assertEqual(self.fit_instance.model_func, models.IMPLEMENTED[model]['func'])
        self.assertEqual(self.fit_instance.model_params, models.IMPLEMENTED[model]['params'])
        self.assertEqual(self.fit_instance._model_dict, models.IMPLEMENTED[model])


    def test_initial_parameter(self):
        p_all = self.p_all
        print(p_all[0,0,0])
        p_all[:, :, :, 0] *= 1e9 # from GHz to Hz
        print(p_all[0,0,0])
        print(self.fit_instance.initial_parameter[0,0,0])
        npt.assert_allclose(self.fit_instance.initial_parameter, p_all, atol=1e-2)

    def test_model_id(self):
        assert False


    def test_fitting_parameter(self):
        assert False


    def test_fitting_parameter_unique(self):
        assert False


    def test_n_parameter(self):
        assert False


    def test_set_constraints(self):
        assert False


    def test_set_free_constraints(self):
        assert False


    def test__set_initial_constraints(self):
        assert False


    def test_constraints(self):
        assert False


    def test_constraints_changed(self):
        assert False


    def test_get_constraints_array(self):
        assert False


    def test_get_constraint_types(self):
        assert False


    def test_parameter(self):
        assert False


    def test_get_param(self):
        assert False


    def test__param_idx(self):
        assert False


    def test__guess_center(self):
        assert False


    def test__guess_contrast(self):
        assert False


    def test__guess_width(self):
        assert False


    def test__guess_offset(self):
        assert False


    def test_get_initial_parameter(self):
        assert False


    def test_fit_odmr(self):
        assert False


    def test_fit_frange(self):
        assert False


    def test_reshape_results(self):
        assert False


    def test_reshape_result(self):
        assert False
