# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
import math
from typing import cast
from unittest.mock import patch, PropertyMock

import torch
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
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
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import unnormalize
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


@mock_botorch_optimize
def get_modelbridge(modular: bool = False, saasbo: bool = False) -> ModelBridge:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    if modular:
        return Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
        )
    if saasbo:
        return Models.SAASBO(
            experiment=exp,
            data=exp.fetch_data(),
        )
    else:
        return Models.LEGACY_BOTORCH(experiment=exp, data=exp.fetch_data())


class SensitivityAnalysisTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = get_modelbridge().model.model
        self.saas_model = get_modelbridge(saasbo=True).model.surrogate.model

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

    def test_SobolGPMean(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_mean = SobolSensitivityGPMean(
            self.model, num_mc_samples=10, bounds=bounds, second_order=True
        )
        first_order = sensitivity_mean.first_order_indices()
        total_order = sensitivity_mean.total_order_indices()
        second_order = sensitivity_mean.second_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertIsInstance(total_order, Tensor)
        self.assertIsInstance(second_order, Tensor)
        self.assertEqual(first_order.shape, torch.Size([2]))
        self.assertEqual(total_order.shape, torch.Size([2]))
        self.assertEqual(second_order.shape, torch.Size([1]))

    def test_SobolGPMean_SAASBO(self) -> None:
        bounds = torch.tensor([(0.0, 1.0) for _ in range(2)]).t()
        sensitivity_mean_saas = SobolSensitivityGPMean(
            self.saas_model, num_mc_samples=10, bounds=bounds, second_order=True
        )
        first_order = sensitivity_mean_saas.first_order_indices()
        total_order = sensitivity_mean_saas.total_order_indices()
        second_order = sensitivity_mean_saas.second_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertIsInstance(total_order, Tensor)
        self.assertIsInstance(second_order, Tensor)
        self.assertEqual(first_order.shape, torch.Size([2]))
        self.assertEqual(total_order.shape, torch.Size([2]))
        self.assertEqual(second_order.shape, torch.Size([1]))

        sensitivity_mean_bootstrap = SobolSensitivityGPMean(
            self.model,
            num_mc_samples=10,
            bounds=bounds,
            second_order=True,
            num_bootstrap_samples=10,
            input_qmc=True,
        )
        first_order = sensitivity_mean_bootstrap.first_order_indices()
        total_order = sensitivity_mean_bootstrap.total_order_indices()
        second_order = sensitivity_mean_bootstrap.second_order_indices()
        self.assertIsInstance(first_order, Tensor)
        self.assertIsInstance(total_order, Tensor)
        self.assertIsInstance(second_order, Tensor)
        self.assertEqual(first_order.shape, torch.Size([2, 3]))
        self.assertEqual(total_order.shape, torch.Size([2, 3]))
        self.assertEqual(second_order.shape, torch.Size([1, 3]))

        sensitivity_mean_bootstrap = SobolSensitivityGPMean(
            self.model,
            num_mc_samples=10,
            bounds=bounds,
            second_order=True,
            num_bootstrap_samples=10,
            link_function=ProbitLinkMean,
        )
        first_order = sensitivity_mean_bootstrap.first_order_indices()
        self.assertEqual(first_order.shape, torch.Size([2, 3]))

        with self.assertRaises(ValueError):
            sensitivity_mean = SobolSensitivityGPMean(
                self.model, num_mc_samples=10, bounds=bounds, second_order=False
            )
            first_order = sensitivity_mean.first_order_indices()
            total_order = sensitivity_mean.total_order_indices()
            second_order = sensitivity_mean.second_order_indices()

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
                self.assertTrue(
                    torch.allclose(
                        indices,
                        base_indices.unsqueeze(0).expand(num_models, 2),
                    )
                )

        # testing ax sensitivity utils
        # model_bridge = cast(TorchModelBridge, get_modelbridge())
        for modular in [False, True]:
            model_bridge = cast(TorchModelBridge, get_modelbridge(modular=modular))
            with self.assertRaisesRegex(
                NotImplementedError,
                "but only TorchModelBridge is supported",
            ):
                # pyre-ignore
                ax_parameter_sens(1, model_bridge.outcomes)

            with patch.object(model_bridge, "model", return_value=None):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"but only Union\[BotorchModel, ModularBoTorchModel\] is supported",
                ):
                    ax_parameter_sens(model_bridge, model_bridge.outcomes)

            torch_model = cast(BotorchModel, model_bridge.model)
            if not modular:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "but only ModelList is supported",
                ):
                    # only applies if the number of outputs of model is greater than 1
                    with patch.object(
                        BatchedMultiOutputGPyTorchModel,
                        "num_outputs",
                        new_callable=PropertyMock,
                    ) as mock:
                        mock.return_value = 2
                        ax_parameter_sens(model_bridge, model_bridge.outcomes)

                # since only ModelList is supported for BotorchModel:
                gpytorch_model = ModelListGP(cast(GPyTorchModel, torch_model.model))
                torch_model.model = gpytorch_model

            for order in ["first", "total"]:
                with self.subTest(order=order):
                    ind_dict = ax_parameter_sens(
                        model_bridge,
                        input_qmc=True,
                        num_mc_samples=num_mc_samples,
                        order=order,
                        signed=False,
                    )
                    self.assertIsInstance(ind_dict, dict)

                    ind_tnsr = compute_sobol_indices_from_model_list(
                        _get_model_per_metric(torch_model, model_bridge.outcomes),
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
                    model_bridge,
                    input_qmc=True,
                    num_mc_samples=num_mc_samples,
                    order="second",
                    signed=False,
                )

                so_ind_tnsr = compute_sobol_indices_from_model_list(
                    _get_model_per_metric(torch_model, model_bridge.outcomes),
                    torch.tensor(torch_model.search_space_digest.bounds).T,
                    input_qmc=True,
                    num_mc_samples=num_mc_samples,
                    order="second",
                )
                fo_ind_tnsr = compute_sobol_indices_from_model_list(
                    _get_model_per_metric(torch_model, model_bridge.outcomes),
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
        model_bridge = get_modelbridge(modular=True)

        # adding a categorical feature
        cat_model_bridge = copy.deepcopy(model_bridge)
        digest = cat_model_bridge.model.search_space_digest
        digest.categorical_features = [0]

        sobol_kwargs = {"input_qmc": True, "num_mc_samples": 10}
        seed = 1234
        for bridge in [model_bridge, cat_model_bridge]:
            discrete_features = bridge.model.search_space_digest.categorical_features
            with self.subTest(model_bridge=bridge):
                set_rng_seed(seed)
                # Unsigned
                ind_dict = ax_parameter_sens(
                    model_bridge=bridge,
                    metrics=None,
                    order="total",
                    signed=False,
                    **sobol_kwargs,
                )
                ind_deriv = compute_derivatives_from_model_list(
                    model_list=[bridge.model.surrogate.model],
                    bounds=torch.tensor(bridge.model.search_space_digest.bounds).T,
                    discrete_features=discrete_features,
                    **sobol_kwargs,
                )
                set_rng_seed(seed)  # reset seed to keep discrete features the same
                cat_indices = bridge.model.search_space_digest.categorical_features
                ind_dict_signed = ax_parameter_sens(
                    model_bridge=bridge,
                    metrics=None,
                    order="total",
                    signed=True,
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
        self.assertTrue(
            torch.allclose(Arnd[:, discrete_feature], A[:, discrete_feature])
        )
        self.assertTrue(
            torch.allclose(Brnd[:, discrete_feature], B[:, discrete_feature])
        )

        # testing that the other features are not integer valued
        self.assertFalse(torch.allclose(Arnd, A))
        self.assertFalse(torch.allclose(Brnd, B))

    def test_Sobol_raises(self) -> None:
        bridge = get_modelbridge(modular=True)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Order third and fourth is not supported. Plese choose one of"
            " 'first', 'total' or 'second'.",
        ):
            ax_parameter_sens(
                model_bridge=bridge,
                metrics=None,
                order="third and fourth",
                signed=False,
            )
        with self.assertRaisesRegex(
            NotImplementedError,
            "Second order is not supported for signed indices.",
        ):
            ax_parameter_sens(
                model_bridge=bridge,
                metrics=None,
                order="second",
                signed=True,
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
            self.assertTrue(torch.allclose(B[:, i], B[:, i].round()))
        # discrete_features=None should be a no-op
        B = sample_discrete_parameters(
            input_mc_samples=A.clone(),
            discrete_features=None,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )
        self.assertTrue(torch.equal(A, B))
