#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    use_model_list,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset


class BoTorchModelUtilsTest(TestCase):
    def setUp(self):
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, _, _, _, _ = get_torch_test_data(dtype=self.dtype)
        self.Xs2, self.Ys2, self.Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, offset=1.0  # Making this data different.
        )
        self.fixed_noise_datasets = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(self.Xs, self.Ys, self.Yvars)
        ]
        self.supervised_datasets = [
            SupervisedDataset(X=X, Y=Y) for X, Y, in zip(self.Xs, self.Ys)
        ]
        self.none_Yvars = [torch.tensor([[np.nan], [np.nan]])]
        self.task_features = []
        self.objective_thresholds = torch.tensor([0.5, 1.5])

    def test_choose_model_class_fidelity_features(self):
        # Only a single fidelity feature can be used.
        with self.assertRaisesRegex(
            NotImplementedError, "Only a single fidelity feature"
        ):
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], fidelity_features=[1, 2]
                ),
            )
        # No support for non-empty task & fidelity features yet.
        with self.assertRaisesRegex(NotImplementedError, "Multi-task multi-fidelity"):
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[1],
                    fidelity_features=[1],
                ),
            )
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            SingleTaskMultiFidelityGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    fidelity_features=[2],
                ),
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiFidelityGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    fidelity_features=[2],
                ),
            ),
        )

    def test_choose_model_class_task_features(self):
        # Only a single task feature can be used.
        with self.assertRaisesRegex(NotImplementedError, "Only a single task feature"):
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1, 2]
                ),
            )
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            MultiTaskGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1]
                ),
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiTaskGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1]
                ),
            ),
        )

    def test_choose_model_class_discrete_features(self):
        # With discrete features, use MixedSingleTaskyGP.
        self.assertEqual(
            MixedSingleTaskGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[],
                    categorical_features=[1],
                ),
            ),
        )

    def test_choose_model_class(self):
        # Mix of known and unknown variances.
        with self.assertRaisesRegex(
            ValueError, "Variances should all be specified, or none should be."
        ):
            choose_model_class(
                datasets=[self.fixed_noise_datasets[0], self.supervised_datasets[0]],
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            )
        # Without fidelity/task features but with Yvar specifications, use FixedNoiseGP.
        self.assertEqual(
            FixedNoiseGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            ),
        )
        # W/out fidelity/task features and w/out Yvar specifications, use SingleTaskGP.
        self.assertEqual(
            SingleTaskGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            ),
        )

    def test_choose_botorch_acqf_class(self):
        self.assertEqual(qNoisyExpectedImprovement, choose_botorch_acqf_class())
        self.assertEqual(
            qNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(objective_thresholds=self.objective_thresholds),
        )
        self.assertEqual(
            qNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(objective_weights=torch.tensor([0.5, 0.5])),
        )
        self.assertEqual(
            qNoisyExpectedImprovement,
            choose_botorch_acqf_class(objective_weights=torch.tensor([1.0, 0.0])),
        )

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

    def test_use_model_list(self):
        self.assertFalse(
            use_model_list(
                datasets=self.supervised_datasets, botorch_model_class=SingleTaskGP
            )
        )
        self.assertFalse(  # Batched multi-output case.
            use_model_list(
                datasets=[SupervisedDataset(X=self.Xs[0], Y=Y) for Y in self.Ys],
                botorch_model_class=SingleTaskGP,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(X=self.Xs[0], Y=self.Ys[0]),
                    SupervisedDataset(X=self.Xs2[0], Y=self.Ys2[0]),
                ],
                botorch_model_class=SingleTaskGP,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=self.supervised_datasets, botorch_model_class=MultiTaskGP
            )
        )
