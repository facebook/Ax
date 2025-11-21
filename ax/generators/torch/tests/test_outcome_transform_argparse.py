# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.botorch_modular.input_constructors.outcome_transform import (
    outcome_transform_argparse,
)
from ax.utils.common.testutils import TestCase
from botorch.models.transforms.outcome import (
    OutcomeTransform,
    Standardize,
    StratifiedStandardize,
)
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from pyre_extensions import assert_is_instance
from torch import Tensor


class DummyOutcomeTransform(OutcomeTransform):
    pass


class OutcomeTransformArgparseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        X = torch.randn((10, 4))
        Y = torch.randn((10, 1))
        self.dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=["chicken", "eggs", "pigeons", "bunnies"],
            outcome_names=["farm"],
        )

    def test_notImplemented(self) -> None:
        with self.assertRaises(NotImplementedError) as e:
            outcome_transform_argparse[type(None)]
            self.assertTrue("Could not find signature for" in str(e))

    def test_register(self) -> None:
        @outcome_transform_argparse.register(DummyOutcomeTransform)
        def _argparse(outcome_transform: DummyOutcomeTransform) -> None:
            pass

        self.assertEqual(_argparse, outcome_transform_argparse[DummyOutcomeTransform])

    def test_argparse_outcome_transform(self) -> None:
        outcome_transform_kwargs_a = outcome_transform_argparse(OutcomeTransform)
        outcome_transform_kwargs_b = outcome_transform_argparse(
            OutcomeTransform, outcome_transform_options={"x": 5}, dataset=self.dataset
        )

        self.assertEqual(outcome_transform_kwargs_a, {})
        self.assertEqual(outcome_transform_kwargs_b, {"x": 5})

    def test_argparse_standardize(self) -> None:
        outcome_transform_kwargs_a = outcome_transform_argparse(
            Standardize, dataset=self.dataset
        )
        outcome_transform_kwargs_b = outcome_transform_argparse(
            Standardize, dataset=self.dataset, outcome_transform_options={"m": 10}
        )
        self.assertEqual(outcome_transform_kwargs_a, {"m": 1})
        self.assertEqual(outcome_transform_kwargs_b, {"m": 10})

    def test_argparse_stratified_standardize(self) -> None:
        X = self.dataset.X
        X[:5, 3] = 0
        X[5:, 3] = 1
        ssd = SearchSpaceDigest(
            feature_names=self.dataset.feature_names,
            bounds=[(0.0, 1.0)] * 3 + [(0.0, 2.0)],
            task_features=[3],
            target_values={3: 1},
        )
        mt_dataset = MultiTaskDataset.from_joint_dataset(
            dataset=self.dataset,
            task_feature_index=3,
            target_task_value=1,
        )
        outcome_transform_kwargs_a = outcome_transform_argparse(
            StratifiedStandardize,
            dataset=mt_dataset,
            search_space_digest=ssd,
        )
        options_b = {"stratification_idx": 2, "default_task_value": 4}
        outcome_transform_kwargs_b = outcome_transform_argparse(
            StratifiedStandardize,
            dataset=mt_dataset,
            outcome_transform_options=options_b,
            search_space_digest=ssd,
        )
        expected_options_a = {
            "stratification_idx": 3,
            "observed_task_values": torch.tensor([0, 1], dtype=torch.long),
            "all_task_values": torch.tensor([0, 1, 2], dtype=torch.long),
            "default_task_value": 1,
        }
        expected_options_b = {
            "stratification_idx": 2,
            "observed_task_values": torch.tensor([0, 1], dtype=torch.long),
            "all_task_values": torch.tensor([0, 1, 2], dtype=torch.long),
            "default_task_value": 4,
        }
        for expected_options, actual_options in zip(
            (expected_options_a, expected_options_b),
            (outcome_transform_kwargs_a, outcome_transform_kwargs_b),
        ):
            self.assertEqual(len(actual_options), 4)
            for k in ("stratification_idx", "stratification_idx"):
                self.assertEqual(actual_options[k], expected_options[k])
            for k in ("observed_task_values", "all_task_values"):
                self.assertTrue(
                    torch.equal(
                        actual_options[k],
                        assert_is_instance(expected_options[k], Tensor),
                    )
                )
