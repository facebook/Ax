# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.sensitivity.derivative_gp import posterior_derivative
from ax.utils.sensitivity.derivative_measures import GpDGSMGpMean, GpDGSMGpSampling
from ax.utils.sensitivity.sobol_measures import (
    ProbitLinkMean,
    SobolSensitivityGPMean,
    SobolSensitivityGPSampling,
)
from ax.utils.testing.core_stubs import get_branin_experiment
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


def get_modelbridge() -> ModelBridge:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    return Models.BOTORCH(
        # Model bridge kwargs
        experiment=exp,
        data=exp.fetch_data(),
    )


class SensitivityAnanlysisTest(TestCase):
    def setUp(self) -> None:
        self.model = get_modelbridge().model.model

    def testDgsmGpMean(self) -> None:
        bounds = torch.tensor([(0, 1) for _ in range(2)]).t()
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
        bounds = torch.tensor([(0, 1) for _ in range(2)]).t()
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
        bounds = torch.tensor([(0, 1) for _ in range(2)]).t()
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

    def testSobolGpSampling(self) -> None:
        bounds = torch.tensor([(0, 1) for _ in range(2)]).t()
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
