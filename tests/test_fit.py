import unittest

import numpy.testing as npt
import pygpufit.gpufit as gf

import QDMpy
from QDMpy._core import fit, models


class test_Fit_class(unittest.TestCase):
    def setUp(self) -> None:
        self.model = "ESR15N"
        self.model_dict = models.IMPLEMENTED[self.model]
        self.m_all, self.f_ghz, self.p_all = fit.make_dummy_data(
            model=self.model, n_freq=500, scan_dimensions=(12, 19)
        )
        self.fit_instance = fit.Fit(
            data=self.m_all, frequencies=self.f_ghz * 1e9, model_name=self.model
        )

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
        self.assertEqual(
            sorted(self.fit_instance.model_params),
            sorted(["center", "contrast", "contrast", "contrast", "offset", "width"]),
        )

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
        self.assertEqual(self.fit_instance.model_name, self.model)

    def test_model_func(self):
        self.assertEqual(
            self.fit_instance.model_func, models.IMPLEMENTED[self.model]["func"]
        )

    def test_model_setter(self):
        model = "ESR15N"
        self.fit_instance.model_name = model
        self.assertEqual(self.fit_instance.model_name, model)
        self.assertEqual(self.fit_instance._model_name, model)
        self.assertEqual(
            self.fit_instance.model_func, models.IMPLEMENTED[model]["func"]
        )
        self.assertEqual(
            self.fit_instance.model_params, models.IMPLEMENTED[model]["params"]
        )
        self.assertEqual(self.fit_instance._model, models.IMPLEMENTED[model])

    def test_initial_parameter(self):
        p_all = self.p_all

        # check if center is within 1 MHz of the true value
        npt.assert_allclose(
            self.fit_instance.initial_parameter[:, :, :, 0],
            p_all[:, :, :, 0] * 1e9,
            atol=1000,
            rtol=1e-5,
        )
        # check if width is within 100 MHz of the true value
        npt.assert_allclose(
            self.fit_instance.initial_parameter[:, :, :, 1],
            p_all[:, :, :, 1] * 1e9,
            atol=100e3,
            rtol=1.5,
        )
        # check if contrast is within 1% of the true value
        npt.assert_allclose(
            self.fit_instance.initial_parameter[:, :, :, 2],
            p_all[:, :, :, 2],
            atol=1e-2,
            rtol=1e-5,
        )

    def test_model_id(self):
        self.assertEqual(self.fit_instance.model_id, getattr(gf.ModelID, self.model))

    def test_fitting_parameter_unique(self):
        if self.model == "ESR15N":
            self.assertEqual(
                self.fit_instance.model_params_unique,
                ["center", "width", "contrast_0", "contrast_1", "offset"],
            )
        elif self.model == "ESR14N":
            self.assertEqual(
                self.fit_instance.model_params_unique,
                ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"],
            )
        elif self.model == "ESRSINGLE":
            self.assertEqual(
                self.fit_instance.model_params_unique,
                ["center", "width", "contrast", "offset"],
            )
        else:
            assert False

    def test_n_parameter(self):
        self.assertEqual(self.fit_instance.n_parameter, len(self.model_dict["params"]))

    def test_set_constraints(self):
        for value in self.fit_instance.model_params_unique:
            for ctype in fit.CONSTRAINT_TYPES:
                self.fit_instance.set_constraints(value, 0, 1, ctype)
                self.assertEqual(
                    self.fit_instance.constraints[value][:-1], [0, 1, ctype]
                )

    def test_set_free_constraints(self):
        self.fit_instance.set_free_constraints()
        for value in self.fit_instance.model_params_unique:
            # onl;y test for the first 3 entries (4 == unit)
            self.assertEqual(
                self.fit_instance.constraints[value][:-1], [None, None, "FREE"]
            )

    def test__set_initial_constraints(self):
        self.fit_instance._set_initial_constraints()

        constraints = QDMpy.load_config(QDMpy.SRC_PATH / "QDMpy" / "config.ini")["fit"][
            "constraints"
        ]

        for value in self.fit_instance.model_params_unique:
            v = value.split("_")[0]
            self.assertEqual(
                self.fit_instance.constraints[value][:-1],
                [
                    constraints[f"{v}_min"],
                    constraints[f"{v}_max"],
                    constraints[f"{v}_type"],
                ],
            )

    def test_constraints(self):
        self.assertEqual(self.fit_instance.constraints, self.fit_instance._constraints)

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
