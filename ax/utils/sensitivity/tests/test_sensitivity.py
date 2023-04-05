# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast
from unittest.mock import patch, PropertyMock

import torch
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
from ax.utils.common.testutils import TestCase
from ax.utils.sensitivity.derivative_gp import posterior_derivative
from ax.utils.sensitivity.derivative_measures import GpDGSMGpMean, GpDGSMGpSampling
from ax.utils.sensitivity.sobol_measures import (
    _get_model_per_metric,
    ax_parameter_sens,
    compute_sobol_indices_from_model_list,
    ProbitLinkMean,
    SobolSensitivityGPMean,
    SobolSensitivityGPSampling,
)
from ax.utils.testing.core_stubs import get_branin_experiment
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


def get_modelbridge(modular: bool = False) -> ModelBridge:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    return (Models.BOTORCH_MODULAR if modular else Models.BOTORCH)(
        # Model bridge kwargs
        experiment=exp,
        data=exp.fetch_data(),
    )


class SensitivityAnanlysisTest(TestCase):
    def setUp(self) -> None:
        self.model = get_modelbridge().model.model

    def testDgsmGpMean(self) -> None:
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

    def testDgsmGpSampling(self) -> None:
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

    def testSobolGpMean(self) -> None:
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
        for order in ["first", "total"]:
            with self.subTest(order=order):
                indices = compute_sobol_indices_from_model_list(
                    [self.model for _ in range(num_models)],
                    bounds=bounds,
                    order=order,
                    num_mc_samples=num_mc_samples,
                    input_qmc=True,
                )
                self.assertEqual(indices.shape, (num_models, 2))
                if order == "total":
                    self.assertTrue((indices >= 0).all())

                sobol_gp_mean = SobolSensitivityGPMean(
                    self.model,
                    bounds=bounds,
                    num_mc_samples=num_mc_samples,
                    input_qmc=True,
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
                    "but only IndependentModelList is supported",
                ):
                    # only applies if the number of outputs of model is greater than 1
                    with patch.object(
                        BatchedMultiOutputGPyTorchModel,
                        "num_outputs",
                        new_callable=PropertyMock,
                    ) as mock:
                        mock.return_value = 2
                        ax_parameter_sens(model_bridge, model_bridge.outcomes)

                # since only IndependentModelList is supported for BotorchModel:
                gpytorch_model = ModelListGP(cast(GPyTorchModel, torch_model.model))
                torch_model.model = gpytorch_model

            for order in ["first", "total"]:
                with self.subTest(order=order):
                    ind_dict = ax_parameter_sens(
                        model_bridge,
                        input_qmc=True,
                        num_mc_samples=num_mc_samples,
                        order=order,
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
                            self.assertAlmostEqual(ind_dict[row][col], ind_tnsr[i, j])

    def testSobolGpSampling(self) -> None:
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

    def testDerivativeGp(self) -> None:
        test_x = torch.rand(2, 2)
        posterior = posterior_derivative(self.model, test_x, kernel_type="matern_l1")
        self.assertIsInstance(posterior, MultivariateNormal)

        with self.assertRaises(ValueError):
            posterior = posterior_derivative(self.model, test_x, kernel_type="xyz")
