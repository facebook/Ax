# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
import math
from typing import cast
from unittest.mock import patch

import torch
from ax.adapter.base import Adapter
from ax.adapter.data_utils import DataLoaderConfig
from ax.adapter.registry import Generators
from ax.adapter.torch import TorchAdapter
from ax.core.data import MAP_KEY
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.random import set_rng_seed
from ax.utils.common.testutils import TestCase
from ax.utils.sensitivity.derivative_gp import posterior_derivative
from ax.utils.sensitivity.derivative_measures import (
    compute_derivatives_from_model_list,
    GpDGSMGpMean,
    GpDGSMGpSampling,
    sample_discrete_parameters,
)
from ax.utils.sensitivity.sobol_measures import (
    _get_model_per_metric,
    ax_parameter_sens,
    compute_sobol_indices_from_model_list,
    ProbitLinkMean,
    SobolSensitivity,
    SobolSensitivityGPMean,
    SobolSensitivityGPSampling,
)
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
)
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.utils.transforms import unnormalize
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


@mock_botorch_optimize
def get_adapter(saasbo: bool = False) -> Adapter:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    if saasbo:
        return Generators.SAASBO(experiment=exp, data=exp.fetch_data())
    return Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())


@mock_botorch_optimize
def get_adapter_with_map_metric() -> TorchAdapter:
    """Create a TorchAdapter with a map metric (step feature)."""
    exp = get_branin_experiment_with_timestamp_map_metric(with_trials_and_data=True)
    return cast(
        TorchAdapter,
        Generators.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.lookup_data(),
            data_loader_config=DataLoaderConfig(),
        ),
    )


class SensitivityAnalysisTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        set_rng_seed(0)
        self.model = get_adapter().botorch_model
        self.saas_model = get_adapter(saasbo=True).botorch_model

    def test_DgsmGpMean(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_mean = GpDGSMGpMean(self.model, bounds=bounds, num_mc_samples=10)
        gradients_measure = sensitivity_mean.gradient_measure()
        gradients_absolute_measure = sensitivity_mean.gradient_absolute_measure()
        gradients_square_measure = sensitivity_mean.gradients_square_measure()
        self.assertIsInstance(gradients_measure, Tensor)
        self.assertIsInstance(gradients_absolute_measure, Tensor)
        self.assertIsInstance(gradients_square_measure, Tensor)
        self.assertEqual(gradients_measure.shape, torch.Size([2]))
        self.assertEqual(gradients_absolute_measure.shape, torch.Size([2]))
        self.assertEqual(gradients_square_measure.shape, torch.Size([2]))

        sensitivity_mean_bootstrap = GpDGSMGpMean(
            self.model, bounds=bounds, num_mc_samples=10, num_bootstrap_samples=10
        )
        gradients_measure = sensitivity_mean_bootstrap.gradient_measure()
        gradients_absolute_measure = (
            sensitivity_mean_bootstrap.gradient_absolute_measure()
        )
        gradients_square_measure = sensitivity_mean_bootstrap.gradients_square_measure()
        self.assertIsInstance(gradients_measure, Tensor)
        self.assertIsInstance(gradients_absolute_measure, Tensor)
        self.assertIsInstance(gradients_square_measure, Tensor)
        self.assertEqual(gradients_measure.shape, torch.Size([2, 3]))
        self.assertEqual(gradients_absolute_measure.shape, torch.Size([2, 3]))
        self.assertEqual(gradients_square_measure.shape, torch.Size([2, 3]))

        # Test derivatives from model list
        res = compute_derivatives_from_model_list(
            [self.model for _ in range(2)],
            bounds=bounds,
        )
        self.assertEqual(res.shape, (2, 2))

    def test_DgsmGpSampling(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_sampling = GpDGSMGpSampling(
            self.model, bounds=bounds, num_mc_samples=10, num_gp_samples=10
        )
        gradients_measure = sensitivity_sampling.gradient_measure()
        gradients_absolute_measure = sensitivity_sampling.gradient_absolute_measure()
        gradients_square_measure = sensitivity_sampling.gradients_square_measure()
        self.assertIsInstance(gradients_measure, Tensor)
        self.assertIsInstance(gradients_absolute_measure, Tensor)
        self.assertIsInstance(gradients_square_measure, Tensor)
        self.assertEqual(gradients_measure.shape, torch.Size([2, 3]))
        self.assertEqual(gradients_absolute_measure.shape, torch.Size([2, 3]))
        self.assertEqual(gradients_square_measure.shape, torch.Size([2, 3]))

        sensitivity_sampling_bootstrap = GpDGSMGpSampling(
            self.model,
            bounds=bounds,
            num_mc_samples=10,
            num_gp_samples=10,
            num_bootstrap_samples=10,
        )
        gradients_measure = sensitivity_sampling_bootstrap.gradient_measure()
        gradients_absolute_measure = (
            sensitivity_sampling_bootstrap.gradient_absolute_measure()
        )
        gradients_square_measure = (
            sensitivity_sampling_bootstrap.gradients_square_measure()
        )
        self.assertIsInstance(gradients_measure, Tensor)
        self.assertIsInstance(gradients_absolute_measure, Tensor)
        self.assertIsInstance(gradients_square_measure, Tensor)
        self.assertEqual(gradients_measure.shape, torch.Size([2, 5]))
        self.assertEqual(gradients_absolute_measure.shape, torch.Size([2, 5]))
        self.assertEqual(gradients_square_measure.shape, torch.Size([2, 5]))

    def _test_sobol_gp_mean(
        self,
        sensitivity: SobolSensitivityGPMean,
        expected_first_order: Tensor,
        expected_total_order: Tensor,
        expected_second_order: Tensor | None = None,
    ) -> None:
        """
        Check that outputs are as expected. The `assertAllClose` checks are
        characterization tests rather than correctness tests; they check that
        behavior is the same as it was in the past, not that it is
        quantitatively correct. Innocuous changes such as changing a random seed
        could potentially break these tests, and it may become necessary to
        delete them.
        """
        atol = 4e-3
        rtol = 2e-3
        first_order = sensitivity.first_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertAllClose(first_order, expected_first_order, atol=atol, rtol=rtol)
        self.assertEqual(first_order.shape, expected_first_order.shape)

        total_order = sensitivity.total_order_indices()
        self.assertIsInstance(total_order, Tensor)
        self.assertAllClose(total_order, expected_total_order, atol=atol, rtol=rtol)
        self.assertEqual(total_order.shape, expected_total_order.shape)

        if expected_second_order is not None:
            second_order = sensitivity.second_order_indices()
            self.assertIsInstance(second_order, Tensor)
            self.assertAllClose(
                second_order, expected_second_order, atol=atol, rtol=rtol
            )
            self.assertEqual(second_order.shape, expected_second_order.shape)

    def test_SobolGPMean(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_mean = SobolSensitivityGPMean(
            self.model, num_mc_samples=10, bounds=bounds, second_order=True
        )
        self._test_sobol_gp_mean(
            sensitivity=sensitivity_mean,
            expected_first_order=torch.tensor([1.1547, -0.4024], dtype=torch.float64),
            expected_total_order=torch.tensor([0.4299, 0.4894], dtype=torch.float64),
            expected_second_order=torch.tensor([-1.4845], dtype=torch.float64),
        )

    def test_SobolGPMean_SAASBO(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_mean_saas = SobolSensitivityGPMean(
            self.saas_model, num_mc_samples=10, bounds=bounds, second_order=True
        )
        self._test_sobol_gp_mean(
            sensitivity=sensitivity_mean_saas,
            expected_first_order=torch.tensor([0.5752, 0.5143], dtype=torch.double),
            expected_total_order=torch.tensor([0.9897, 0.0979], dtype=torch.float64),
            expected_second_order=torch.tensor([0.8332], dtype=torch.double),
        )

        sensitivity_mean_bootstrap = SobolSensitivityGPMean(
            self.model,
            num_mc_samples=10,
            bounds=bounds,
            second_order=True,
            num_bootstrap_samples=10,
            input_qmc=True,
        )
        self._test_sobol_gp_mean(
            sensitivity=sensitivity_mean_bootstrap,
            expected_first_order=torch.tensor(
                [[0.6327, 10.0889, 1.0044], [0.2089, 0.7322, 0.2706]],
                dtype=torch.float64,
            ),
            expected_total_order=torch.tensor(
                [[0.8013, 0.1824, 0.1351], [0.2203, 0.0304, 0.0551]],
                dtype=torch.float64,
            ),
            expected_second_order=torch.tensor(
                [[0.7978, 22.6598, 1.5053]], dtype=torch.float64
            ),
        )

        sensitivity_mean_bootstrap = SobolSensitivityGPMean(
            self.model,
            num_mc_samples=10,
            bounds=bounds,
            second_order=True,
            num_bootstrap_samples=10,
            link_function=ProbitLinkMean,
        )
        self._test_sobol_gp_mean(
            sensitivity=sensitivity_mean_bootstrap,
            expected_first_order=torch.tensor(
                [[3.4512, 32.4428, 1.8012], [0.2069, 121.8610, 3.4909]],
                dtype=torch.float64,
            ),
            expected_total_order=torch.tensor(
                [[0.4288, 0.0903, 0.0950], [0.7923, 0.2218, 0.1489]],
                dtype=torch.float64,
            ),
            expected_second_order=torch.tensor(
                [[-6.3790, 397.4363, 6.3043]], dtype=torch.float64
            ),
        )

        sensitivity_mean = SobolSensitivityGPMean(
            self.model, num_mc_samples=10, bounds=bounds, second_order=False
        )
        self._test_sobol_gp_mean(
            sensitivity=sensitivity_mean,
            expected_first_order=torch.tensor([0.9566, -0.4183], dtype=torch.float64),
            expected_total_order=torch.tensor([0.3440, 0.3685], dtype=torch.float64),
        )

        with self.assertRaisesRegex(ValueError, "Second order indices"):
            sensitivity_mean.second_order_indices()

        # testing compute_sobol_indices_from_model_list
        num_models = 3
        num_mc_samples = 10
        for order in ["first", "total", "second"]:
            with self.subTest(order=order):
                indices = compute_sobol_indices_from_model_list(
                    [self.model for _ in range(num_models)],
                    bounds=bounds,
                    order=order,
                    num_mc_samples=num_mc_samples,
                    input_qmc=True,
                )
                if order == "second":
                    self.assertEqual(indices.shape, (num_models, 1))
                else:
                    self.assertEqual(indices.shape, (num_models, 2))
                if order == "total":
                    self.assertTrue((indices >= 0).all())

                sobol_gp_mean = SobolSensitivityGPMean(
                    self.model,
                    bounds=bounds,
                    num_mc_samples=num_mc_samples,
                    input_qmc=True,
                    second_order=order == "second",
                )
                base_indices = getattr(sobol_gp_mean, f"{order}_order_indices")()
                # can compare values because we sample with deterministic seeds
                self.assertAllClose(
                    indices,
                    base_indices.unsqueeze(0).expand(num_models, indices.shape[1]),
                )

    def test_SobolGPMean_SAASBO_Ax_utils(self) -> None:
        num_mc_samples = 10
        adapter = cast(TorchAdapter, get_adapter())
        with self.assertRaisesRegex(
            NotImplementedError,
            "but only TorchAdapter is supported",
        ):
            # pyre-ignore
            ax_parameter_sens(1, adapter.outcomes)

        with patch.object(adapter, "generator", return_value=None):
            with self.assertRaisesRegex(
                NotImplementedError,
                "but only BoTorchGenerator",
            ):
                ax_parameter_sens(adapter, adapter.outcomes)

        torch_model = cast(BoTorchGenerator, adapter.generator)
        for order in ["first", "total"]:
            with self.subTest(order=order):
                ind_dict = ax_parameter_sens(
                    adapter,
                    input_qmc=True,
                    num_mc_samples=num_mc_samples,
                    order=order,
                    signed=False,
                )
                self.assertIsInstance(ind_dict, dict)

                ind_tnsr = compute_sobol_indices_from_model_list(
                    _get_model_per_metric(torch_model, adapter.outcomes),
                    torch.tensor(torch_model.search_space_digest.bounds).T,
                    input_qmc=True,
                    num_mc_samples=num_mc_samples,
                    order=order,
                )
                self.assertIsInstance(ind_tnsr, Tensor)

                # can compare values because we sample with deterministic seeds
                for i, row in enumerate(ind_dict):
                    for j, col in enumerate(ind_dict[row]):
                        self.assertAlmostEqual(
                            ind_dict[row][col],
                            # pyre-fixme[6]: For 2nd argument expected
                            #  `SupportsRSub[Variable[_T],
                            #  SupportsAbs[SupportsRound[object]]]` but got
                            #  `Union[bool, float, int]`.
                            ind_tnsr[i, j].item(),
                        )
        with self.subTest(order="second"):
            second_ind_dict = ax_parameter_sens(
                adapter,
                input_qmc=True,
                num_mc_samples=num_mc_samples,
                order="second",
                signed=False,
            )

            so_ind_tnsr = compute_sobol_indices_from_model_list(
                _get_model_per_metric(torch_model, adapter.outcomes),
                torch.tensor(torch_model.search_space_digest.bounds).T,
                input_qmc=True,
                num_mc_samples=num_mc_samples,
                order="second",
            )
            fo_ind_tnsr = compute_sobol_indices_from_model_list(
                _get_model_per_metric(torch_model, adapter.outcomes),
                torch.tensor(torch_model.search_space_digest.bounds).T,
                input_qmc=True,
                num_mc_samples=num_mc_samples,
                order="first",
            )
            # check that the first and second order indices are the same
            self.assertAlmostEqual(
                second_ind_dict["branin"]["x1"],
                # pyre-fixme[6]: For 2nd argument expected
                #  `SupportsRSub[Variable[_T], SupportsAbs[SupportsRound[object]]]`
                #  but got `Union[bool, float, int]`.
                fo_ind_tnsr[0, 0].item(),
            )
            self.assertAlmostEqual(
                second_ind_dict["branin"]["x2"],
                # pyre-fixme[6]: For 2nd argument expected
                #  `SupportsRSub[Variable[_T], SupportsAbs[SupportsRound[object]]]`
                #  but got `Union[bool, float, int]`.
                fo_ind_tnsr[0, -1].item(),
            )
            self.assertAlmostEqual(
                second_ind_dict["branin"]["x1 & x2"],
                # pyre-fixme[6]: For 2nd argument expected
                #  `SupportsRSub[Variable[_T], SupportsAbs[SupportsRound[object]]]`
                #  but got `Union[bool, float, int]`.
                so_ind_tnsr[0, 0].item(),
            )

        # Test with signed
        base_adapter = get_adapter()

        # adding a categorical feature
        cat_adapter = copy.deepcopy(base_adapter)
        digest = cat_adapter.generator.search_space_digest
        digest.categorical_features = [0]

        sobol_kwargs = {"input_qmc": True, "num_mc_samples": 10}
        seed = 1234
        for adapter in [base_adapter, cat_adapter]:
            discrete_features = (
                adapter.generator.search_space_digest.categorical_features
            )
            with self.subTest(adapter=adapter):
                set_rng_seed(seed)
                # Unsigned
                ind_dict = ax_parameter_sens(
                    adapter=adapter,
                    metrics=None,
                    order="total",
                    signed=False,
                    exclude_map_key=False,
                    **sobol_kwargs,
                )
                ind_deriv = compute_derivatives_from_model_list(
                    model_list=[adapter.botorch_model],
                    bounds=torch.tensor(adapter.generator.search_space_digest.bounds).T,
                    discrete_features=discrete_features,
                    fixed_features=None,
                    **sobol_kwargs,
                )
                set_rng_seed(seed)  # reset seed to keep discrete features the same
                cat_indices = adapter.generator.search_space_digest.categorical_features
                ind_dict_signed = ax_parameter_sens(
                    adapter=adapter,
                    metrics=None,
                    order="total",
                    signed=True,
                    exclude_map_key=False,
                    **sobol_kwargs,
                )
                for i, pname in enumerate(["x1", "x2"]):
                    if i in cat_indices:  # special case for categorical features
                        expected_sign = 1
                    else:
                        expected_sign = torch.sign(ind_deriv[0, i]).item()

                    self.assertEqual(
                        expected_sign,
                        math.copysign(1, ind_dict_signed["branin"][pname]),
                    )
                    self.assertAlmostEqual(
                        (expected_sign * ind_dict["branin"][pname]).item(),
                        ind_dict_signed["branin"][pname],
                    )  # signed
                    self.assertTrue(ind_dict["branin"][pname] >= 0)  # unsigned

    def test_SobolGPSampling(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_sampling = SobolSensitivityGPSampling(
            self.model,
            num_mc_samples=10,
            num_gp_samples=10,
            bounds=bounds,
            second_order=True,
        )
        first_order = sensitivity_sampling.first_order_indices()
        total_order = sensitivity_sampling.total_order_indices()
        second_order = sensitivity_sampling.second_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertIsInstance(total_order, Tensor)
        self.assertIsInstance(second_order, Tensor)
        self.assertEqual(first_order.shape, torch.Size([2, 3]))
        self.assertEqual(total_order.shape, torch.Size([2, 3]))
        self.assertEqual(second_order.shape, torch.Size([1, 3]))

        sensitivity_sampling_bootstrap = SobolSensitivityGPSampling(
            self.model,
            num_mc_samples=10,
            num_gp_samples=10,
            bounds=bounds,
            second_order=True,
            num_bootstrap_samples=10,
        )
        first_order = sensitivity_sampling_bootstrap.first_order_indices()
        total_order = sensitivity_sampling_bootstrap.total_order_indices()
        second_order = sensitivity_sampling_bootstrap.second_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertIsInstance(total_order, Tensor)
        self.assertIsInstance(second_order, Tensor)
        self.assertEqual(first_order.shape, torch.Size([2, 5]))
        self.assertEqual(total_order.shape, torch.Size([2, 5]))
        self.assertEqual(second_order.shape, torch.Size([1, 5]))

        with self.assertRaises(ValueError):
            sensitivity_sampling = SobolSensitivityGPSampling(
                self.model,
                num_mc_samples=10,
                num_gp_samples=10,
                bounds=bounds,
                second_order=False,
            )
            first_order = sensitivity_sampling.first_order_indices()
            total_order = sensitivity_sampling.total_order_indices()
            second_order = sensitivity_sampling.second_order_indices()

        discrete_feature = 0
        sensitivity_sampling_discrete = SobolSensitivityGPSampling(
            self.model,
            num_mc_samples=10,
            num_gp_samples=10,
            bounds=bounds,
            second_order=True,
            discrete_features=[discrete_feature],
        )
        sens = sensitivity_sampling_discrete.sensitivity
        A = sens.A
        B = sens.B
        Arnd = A.round()
        Brnd = B.round()
        # testing that the discrete feature is integer valued
        self.assertAllClose(Arnd[:, discrete_feature], A[:, discrete_feature])
        self.assertAllClose(Brnd[:, discrete_feature], B[:, discrete_feature])

        # testing that the other features are not integer valued
        self.assertFalse(torch.allclose(Arnd, A))
        self.assertFalse(torch.allclose(Brnd, B))

    def test_Sobol_raises(self) -> None:
        adapter = get_adapter()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Order third and fourth is not supported. Plese choose one of"
            " 'first', 'total' or 'second'.",
        ):
            ax_parameter_sens(
                adapter=adapter,
                metrics=None,
                order="third and fourth",
                signed=False,
            )

    def test_Sobol_computed_correctly_first_order(self) -> None:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        sensitivity_x1 = SobolSensitivity(
            bounds, input_function=lambda x: x[:, 0], num_mc_samples=10
        )
        sensitivity_x2 = SobolSensitivity(bounds, input_function=lambda x: x[:, 1])

        sensitivity_x1.evalute_function()
        sensitivity_x2.evalute_function()
        fo_indices1 = sensitivity_x1.first_order_indices()
        fo_indices2 = sensitivity_x2.first_order_indices()
        self.assertEqual(fo_indices1[0] / fo_indices1.sum(), 1.0)
        self.assertEqual(fo_indices1[1] / fo_indices1.sum(), 0.0)
        self.assertEqual(fo_indices2[1] / fo_indices2.sum(), 1.0)
        self.assertEqual(fo_indices2[0] / fo_indices2.sum(), 0.0)

    def test_Sobol_computed_correctly_second_order(self) -> None:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        sensitivity_x1x2 = SobolSensitivity(
            bounds,
            input_function=lambda x: Tensor([1234.5]),  # not used
            second_order=True,
            num_mc_samples=10,
        )
        f_A = torch.rand(10)
        f_B = -f_A
        sensitivity_x1x2.f_total_var = torch.cat((f_A, f_B)).var()
        sensitivity_x1x2.f_A = f_A
        sensitivity_x1x2.f_B = -f_A
        sensitivity_x1x2.f_ABis = [f_A, f_A]
        sensitivity_x1x2.f_BAis = [f_A, -f_B]

        fo_indices = sensitivity_x1x2.first_order_indices()
        so_indices = sensitivity_x1x2.second_order_indices()

        self.assertTrue(torch.all(fo_indices == 0.0))
        self.assertTrue(torch.all(so_indices > 0.0))

    def test_DerivativeGp(self) -> None:
        test_x = torch.rand(2, 2)
        posterior = posterior_derivative(self.model, test_x, kernel_type="matern")
        self.assertIsInstance(posterior, MultivariateNormal)

        with self.assertRaises(ValueError):
            posterior = posterior_derivative(self.model, test_x, kernel_type="xyz")

    def test_sample_discrete_parameters(self) -> None:
        dim = 5
        bounds = torch.stack((torch.zeros(dim), torch.arange(1, dim + 1)))
        num_mc_samples = 8
        A = unnormalize(torch.rand(num_mc_samples, dim), bounds=bounds)
        discrete_features, continuous_features = [1, 3], [0, 2, 4]
        B = sample_discrete_parameters(
            input_mc_samples=A.clone(),
            discrete_features=discrete_features,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )
        self.assertTrue(  # Non-discrete parameters should be untouched
            torch.equal(A[:, continuous_features], B[:, continuous_features])
        )
        for i in discrete_features:  # Make sure we sampled integers in the right range
            self.assertTrue(B[:, i].min() >= bounds[0, i])
            self.assertTrue(B[:, i].max() <= bounds[1, i])
            self.assertAllClose(B[:, i], B[:, i].round())
        # discrete_features=None should be a no-op
        B = sample_discrete_parameters(
            input_mc_samples=A.clone(),
            discrete_features=None,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )
        self.assertTrue(torch.equal(A, B))

    def test_ax_parameter_sens_exclude_map_key(self) -> None:
        """Test that exclude_map_key excludes/includes MAP_KEY from output."""
        adapter = get_adapter_with_map_metric()
        generator = cast(BoTorchGenerator, adapter.generator)
        feature_names = list(generator.search_space_digest.feature_names)

        # Verify that the adapter has a "step" feature (MAP_KEY)
        self.assertIn(MAP_KEY, feature_names)

        sobol_kwargs = {"input_qmc": True, "num_mc_samples": 10}

        # Call with exclude_map_key=True
        result_excluded = ax_parameter_sens(
            adapter=adapter,
            metrics=None,
            order="first",
            signed=False,
            exclude_map_key=True,
            **sobol_kwargs,
        )

        # Call with exclude_map_key=False
        result_included = ax_parameter_sens(
            adapter=adapter,
            metrics=None,
            order="first",
            signed=False,
            exclude_map_key=False,
            **sobol_kwargs,
        )

        # Verify results structure
        for metric_name, param_sens_excluded in result_excluded.items():
            # MAP_KEY should not be in output when excluded
            self.assertNotIn(MAP_KEY, param_sens_excluded.keys())

            # MAP_KEY should be in output when included
            param_sens_included = result_included[metric_name]
            self.assertIn(MAP_KEY, param_sens_included.keys())

            # Non-MAP_KEY parameters should be present in both
            for param_name in param_sens_excluded.keys():
                self.assertIn(param_name, param_sens_included.keys())
