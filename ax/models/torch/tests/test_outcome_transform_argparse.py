# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.models.torch.botorch_modular.input_constructors.outcome_transform import (
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
        mt_dataset = MultiTaskDataset.from_joint_dataset(
            dataset=self.dataset,
            task_feature_index=3,
            target_task_value=1,
        )
        outcome_transform_kwargs_a = outcome_transform_argparse(
            StratifiedStandardize, dataset=mt_dataset
        )
        options_b = {
            "stratification_idx": 2,
            "task_values": torch.tensor([0, 3]),
        }
        outcome_transform_kwargs_b = outcome_transform_argparse(
            StratifiedStandardize,
            dataset=mt_dataset,
            outcome_transform_options=options_b,
        )
        expected_options_a = {
            "stratification_idx": 3,
            "task_values": torch.tensor([0, 1]),
        }
        for expected_options, actual_options in zip(
            (expected_options_a, options_b),
            (outcome_transform_kwargs_a, outcome_transform_kwargs_b),
        ):
            self.assertEqual(len(actual_options), 2)
            self.assertEqual(
                actual_options["stratification_idx"],
                expected_options["stratification_idx"],
            )
            self.assertTrue(
                torch.equal(
                    actual_options["task_values"],
                    assert_is_instance(expected_options["task_values"], torch.Tensor),
                )
            )
