#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_mll_class,
    choose_model_class,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class BoTorchModelUtilsTest(TestCase):
    def test_choose_model_class(self):
        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.task_features = []
        # Task features is not implemented yet.
        with self.assertRaisesRegex(
            NotImplementedError, "do not support `task_features`"
        ):
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                task_features=[1],
                fidelity_features=[],
            )
        # Only a single fidelity feature can be used.
        with self.assertRaisesRegex(
            NotImplementedError, "only a single fidelity parameter"
        ):
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                task_features=self.task_features,
                fidelity_features=[1, 2],
            )
        # With fidelity features, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            SingleTaskMultiFidelityGP,
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                task_features=self.task_features,
                fidelity_features=[2],
            ),
        )
        # Without fidelity features but with Yvar specifications, use FixedNoiseGP.
        self.assertEqual(
            FixedNoiseGP,
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                task_features=self.task_features,
                fidelity_features=[],
            ),
        )
        # Without fidelity features and without Yvar specifications, use SingleTaskGP.
        self.assertEqual(
            SingleTaskGP,
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=[torch.tensor([[float("nan")], [float("nan")]])],
                task_features=self.task_features,
                fidelity_features=[],
            ),
        )
        # Mix of known and unknown variances.
        with self.assertRaisesRegex(
            ValueError, "Variances should all be specified, or none should be."
        ):
            choose_model_class(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=[torch.tensor([[0.0], [float("nan")]])],
                task_features=self.task_features,
                fidelity_features=[],
            )

    def test_choose_mll_class(self):
        # Use ExactMLL when `state_dict` is not None and `refit` is False.
        self.assertEqual(
            ExactMarginalLogLikelihood,
            choose_mll_class(
                model_class=ModelListGP, state_dict={"non-empty": None}, refit=False
            ),
        )

        # Otherwise, when `state_dict` is None or `refit` is True:
        # Use SumMLL when using a `ModelListGP`.
        self.assertEqual(
            SumMarginalLogLikelihood,
            choose_mll_class(
                model_class=ModelListGP, state_dict={"non-empty": None}, refit=True
            ),
        )
        self.assertEqual(
            SumMarginalLogLikelihood,
            choose_mll_class(model_class=ModelListGP, state_dict=None, refit=False),
        )
        # Use ExactMLL otherwise.
        self.assertEqual(
            ExactMarginalLogLikelihood,
            choose_mll_class(
                model_class=SingleTaskGP, state_dict={"non-empty": None}, refit=True
            ),
        )
        self.assertEqual(
            ExactMarginalLogLikelihood,
            choose_mll_class(model_class=SingleTaskGP, state_dict=None, refit=False),
        )

    def test_choose_botorch_acqf_class(self):
        self.assertEqual(qNoisyExpectedImprovement, choose_botorch_acqf_class())
