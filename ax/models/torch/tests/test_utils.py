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
    construct_acquisition_and_optimizer_options,
    construct_training_data,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.containers import TrainingData
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class BoTorchModelUtilsTest(TestCase):
    def setUp(self):
        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.task_features = []

    def test_choose_model_class(self):
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

    def test_construct_acquisition_and_optimizer_options(self):
        # Two dicts for `Acquisition` should be concatenated
        acqf_options = {Keys.NUM_FANTASIES: 64}

        acquisition_function_kwargs = {Keys.CURRENT_VALUE: torch.tensor([1.0])}
        optimizer_kwargs = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        model_gen_options = {
            Keys.ACQF_KWARGS: acquisition_function_kwargs,
            Keys.OPTIMIZER_KWARGS: optimizer_kwargs,
        }

        (
            final_acq_options,
            final_opt_options,
        ) = construct_acquisition_and_optimizer_options(
            acqf_options=acqf_options, model_gen_options=model_gen_options
        )
        self.assertEqual(
            final_acq_options,
            {Keys.NUM_FANTASIES: 64, Keys.CURRENT_VALUE: torch.tensor([1.0])},
        )
        self.assertEqual(final_opt_options, optimizer_kwargs)

    def test_construct_training_data(self):
        # len(Xs) == len(Ys) == len(Yvars) == 1 case
        self.assertEqual(
            construct_training_data(
                Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars, model_class=SingleTaskGP
            ),
            TrainingData(X=self.Xs[0], Y=self.Ys[0], Yvar=self.Yvars[0]),
        )
        # len(Xs) == len(Ys) == len(Yvars) > 1 case, batched multi-output
        td = construct_training_data(
            Xs=self.Xs * 2,
            Ys=self.Ys * 2,
            Yvars=self.Yvars * 2,
            model_class=SingleTaskGP,
        )
        expected = TrainingData(
            X=self.Xs[0],
            Y=torch.cat(self.Ys * 2, dim=-1),
            Yvar=torch.cat(self.Yvars * 2, dim=-1),
        )
        self.assertTrue(torch.equal(td.X, expected.X))
        self.assertTrue(torch.equal(td.Y, expected.Y))
        self.assertTrue(torch.equal(td.Yvar, expected.Yvar))
        # len(Xs) == len(Ys) == len(Yvars) > 1 case, not supporting batched
        # multi-output (`Model` not a subclass of `BatchedMultiOutputGPyTorchModel`)
        with self.assertRaisesRegex(ValueError, "Unexpected training data format"):
            td = construct_training_data(
                Xs=self.Xs * 2, Ys=self.Ys * 2, Yvars=self.Yvars * 2, model_class=Model
            )
