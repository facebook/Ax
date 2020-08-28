#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.models.torch.botorch_modular.utils import choose_mll_class, choose_model_class
from ax.utils.common.testutils import TestCase
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.containers import TrainingData
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class UtilsTest(TestCase):
    def test_choose_model_class(self):
        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.task_features = []
        # Task features is not implemented yet
        with self.assertRaisesRegex(
            NotImplementedError, "Currently do not support `task_features`!"
        ):
            choose_model_class(
                training_data=self.training_data,
                task_features=[1],
                fidelity_features=[],
            )
        # Only a single fidelity feature can be used
        with self.assertRaisesRegex(
            NotImplementedError, ".* only a single fidelity parameter!"
        ):
            choose_model_class(
                training_data=self.training_data,
                task_features=self.task_features,
                fidelity_features=[1, 2],
            )
        # Yvars is not all nan
        self.assertEqual(
            SingleTaskMultiFidelityGP,
            choose_model_class(
                training_data=self.training_data,
                task_features=self.task_features,
                fidelity_features=[2],
            ),
        )
        self.assertEqual(
            FixedNoiseGP,
            choose_model_class(
                training_data=self.training_data,
                task_features=self.task_features,
                fidelity_features=[],
            ),
        )
        # Yvars is all nan
        self.Yvars = [torch.tensor([[float("nan")], [float("nan")]])]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.assertEqual(
            SingleTaskGP,
            choose_model_class(
                training_data=self.training_data,
                task_features=self.task_features,
                fidelity_features=[],
            ),
        )

    def test_choose_mll_class(self):
        self.assertEqual(
            SumMarginalLogLikelihood,
            choose_mll_class(model_class=ModelListGP, state_dict=None, refit=True),
        )
        self.assertEqual(
            ExactMarginalLogLikelihood,
            choose_mll_class(model_class=SingleTaskGP, state_dict=None, refit=True),
        )
