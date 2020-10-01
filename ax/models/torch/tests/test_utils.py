#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    construct_single_training_data,
    construct_training_data_list,
    use_model_list,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.utils.containers import TrainingData


class BoTorchModelUtilsTest(TestCase):
    def setUp(self):
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, _, _, _, _ = get_torch_test_data(dtype=self.dtype)
        self.Xs2, self.Ys2, self.Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, offset=1.0  # Making this data different.
        )
        # self.Xs = Xs1
        # self.Ys = [torch.tensor([[3.0], [4.0]])]
        # self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.none_Yvars = [torch.tensor([[np.nan], [np.nan]])]
        self.task_features = []

    def test_choose_model_class_fidelity_features(self):
        # Only a single fidelity feature can be used.
        with self.assertRaisesRegex(
            NotImplementedError, "Only a single fidelity feature"
        ):
            choose_model_class(
                Yvars=self.Yvars, task_features=[], fidelity_features=[1, 2]
            )
        # No support for non-empty task & fidelity features yet.
        with self.assertRaisesRegex(NotImplementedError, "Multi-task multi-fidelity"):
            choose_model_class(
                Yvars=self.Yvars, task_features=[1], fidelity_features=[1]
            )
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            SingleTaskMultiFidelityGP,
            choose_model_class(
                Yvars=self.none_Yvars, task_features=[], fidelity_features=[2]
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiFidelityGP,
            choose_model_class(
                Yvars=self.Yvars, task_features=[], fidelity_features=[2]
            ),
        )

    def test_choose_model_class_task_features(self):
        # Only a single task feature can be used.
        with self.assertRaisesRegex(NotImplementedError, "Only a single task feature"):
            choose_model_class(
                Yvars=self.Yvars, task_features=[1, 2], fidelity_features=[]
            )
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            MultiTaskGP,
            choose_model_class(
                Yvars=self.none_Yvars, task_features=[1], fidelity_features=[]
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiTaskGP,
            choose_model_class(
                Yvars=self.Yvars, task_features=[1], fidelity_features=[]
            ),
        )

    def test_choose_model_class(self):
        # Mix of known and unknown variances.
        with self.assertRaisesRegex(
            ValueError, "Variances should all be specified, or none should be."
        ):
            choose_model_class(
                Yvars=[torch.tensor([[0.0], [np.nan]])],
                task_features=[],
                fidelity_features=[],
            )
        # Without fidelity/task features but with Yvar specifications, use FixedNoiseGP.
        self.assertEqual(
            FixedNoiseGP,
            choose_model_class(
                Yvars=self.Yvars, task_features=[], fidelity_features=[]
            ),
        )
        # W/out fidelity/task features and w/out Yvar specifications, use SingleTaskGP.
        self.assertEqual(
            SingleTaskGP,
            choose_model_class(
                Yvars=[torch.tensor([[float("nan")], [float("nan")]])],
                task_features=[],
                fidelity_features=[],
            ),
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

    def test_construct_single_training_data(self):
        # len(Xs) == len(Ys) == len(Yvars) == 1 case
        self.assertEqual(
            construct_single_training_data(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars),
            TrainingData(X=self.Xs[0], Y=self.Ys[0], Yvar=self.Yvars[0]),
        )
        # len(Xs) == len(Ys) == len(Yvars) > 1 case, batched multi-output
        td = construct_single_training_data(
            Xs=self.Xs * 2, Ys=self.Ys * 2, Yvars=self.Yvars * 2
        )
        expected = TrainingData(
            X=self.Xs[0],
            Y=torch.cat(self.Ys * 2, dim=-1),
            Yvar=torch.cat(self.Yvars * 2, dim=-1),
        )
        self.assertTrue(torch.equal(td.X, expected.X))
        self.assertTrue(torch.equal(td.Y, expected.Y))
        self.assertTrue(torch.equal(td.Yvar, expected.Yvar))
        # len(Xs) == len(Ys) == len(Yvars) > 1 case with not all Xs equal,
        # not supported and should go to `construct_training_data_list` instead.
        with self.assertRaisesRegex(ValueError, "Unexpected training data format"):
            td = construct_single_training_data(
                Xs=self.Xs + self.Xs2,  # Unequal Xs.
                Ys=self.Ys * 2,
                Yvars=self.Yvars * 2,
            )

    def test_construct_training_data_list(self):
        td_list = construct_training_data_list(
            Xs=self.Xs + self.Xs2, Ys=self.Ys + self.Ys2, Yvars=self.Yvars + self.Yvars2
        )
        self.assertEqual(len(td_list), 2)
        self.assertEqual(
            td_list[0], TrainingData(X=self.Xs[0], Y=self.Ys[0], Yvar=self.Yvars[0])
        )
        self.assertEqual(
            td_list[1], TrainingData(X=self.Xs2[0], Y=self.Ys2[0], Yvar=self.Yvars2[0])
        )

    def test_use_model_list(self):
        self.assertFalse(use_model_list(Xs=self.Xs, botorch_model_class=SingleTaskGP))
        self.assertFalse(  # Batched multi-output case.
            use_model_list(Xs=self.Xs * 2, botorch_model_class=SingleTaskGP)
        )
        self.assertTrue(
            use_model_list(Xs=self.Xs + self.Xs2, botorch_model_class=SingleTaskGP)
        )
        self.assertTrue(use_model_list(Xs=self.Xs, botorch_model_class=MultiTaskGP))
