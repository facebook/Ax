#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.botorch_modular.input_constructors.input_transforms import (
    input_transform_argparse,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.models.transforms.input import (
    FilterFeatures,
    InputStandardize,
    InputTransform,
    LearnedFeatureImputation,
    Normalize,
    Warp,
)
from botorch.utils.datasets import (
    ContextualDataset,
    MultiTaskDataset,
    SupervisedDataset,
)


class DummyInputTransform(InputTransform):  # pyre-ignore [13]
    pass


class InputTransformArgparseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        X = torch.randn((10, 4))
        Y = torch.randn((10, 2))
        self.dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=[f"x{i}" for i in range(4)],
            outcome_names=[f"y{i}" for i in range(2)],
        )
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0.0, 1.0), (0, 2), (0, 4)],
            ordinal_features=[0],
            categorical_features=[1],
            discrete_choices={0: [0, 1, 2], 1: [0, 0.25, 4.0]},
            task_features=[2],
            fidelity_features=[0],
            target_values={0: 1.0},
        )

    def test_notImplemented(self) -> None:
        with self.assertRaises(NotImplementedError) as e:
            input_transform_argparse[
                type(None)
            ]  # passing `None` produces a different error
            self.assertTrue("Could not find signature for" in str(e))

    def test_register(self) -> None:
        @input_transform_argparse.register(DummyInputTransform)
        def _argparse(input_transform: DummyInputTransform) -> None:
            pass

        self.assertEqual(input_transform_argparse[DummyInputTransform], _argparse)

    def test_fallback(self) -> None:
        with patch.dict(input_transform_argparse.funcs, {}):

            @input_transform_argparse.register(InputTransform)
            def _argparse(input_transform_class: InputTransform) -> None:
                pass

            self.assertEqual(input_transform_argparse[InputTransform], _argparse)

    def test_argparse_input_transform(self) -> None:
        input_transform_kwargs = input_transform_argparse(
            InputTransform,
            dataset=self.dataset,
        )

        self.assertEqual(input_transform_kwargs, {})

        input_transform_kwargs = input_transform_argparse(
            InputStandardize, dataset=self.dataset, input_transform_options={"d": 10}
        )

        self.assertEqual(input_transform_kwargs, {"d": 10})

    def test_argparse_normalize(self) -> None:
        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
        )

        self.assertTrue(
            torch.all(
                torch.isclose(
                    input_transform_kwargs["bounds"],
                    torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]],
                        dtype=input_transform_kwargs["bounds"].dtype,
                    ),
                )
            )
        )
        # Use the dimension of the datasets as the ground truth as
        # that's what will be fed to the model.
        self.assertEqual(input_transform_kwargs["d"], 4)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1, 3])

        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={
                "d": 4,
                "bounds": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]], dtype=torch.float64
                ),
            },
        )

        self.assertEqual(input_transform_kwargs["d"], 4)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1, 3])

        self.assertTrue(
            torch.all(
                torch.isclose(
                    input_transform_kwargs["bounds"],
                    torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]], dtype=torch.float64
                    ),
                )
            )
        )

        # Test with MultiTaskDataset.
        dataset1 = SupervisedDataset(
            X=torch.cat([torch.rand(5, 4), torch.zeros(5, 1)], dim=-1),
            Y=torch.randn(5, 1),
            feature_names=[f"x{i}" for i in range(4)] + [Keys.TASK_FEATURE_NAME.value],
            outcome_names=["y0"],
        )

        dataset2 = SupervisedDataset(
            X=torch.cat([torch.rand(5, 2), torch.zeros(5, 1)], dim=-1),
            Y=torch.randn(5, 1),
            feature_names=[f"x{i}" for i in range(2)] + [Keys.TASK_FEATURE_NAME.value],
            outcome_names=["y1"],
        )
        mtds = MultiTaskDataset(
            datasets=[dataset1, dataset2],
            target_outcome_name="y0",
            task_feature_index=-1,
        )
        mt_ssd = dataclasses.replace(
            self.search_space_digest,
            feature_names=self.search_space_digest.feature_names
            + [Keys.TASK_FEATURE_NAME.value],
            task_features=[-1],
            bounds=self.search_space_digest.bounds + [(0.0, 1.0)],
        )
        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=mtds,
            search_space_digest=mt_ssd,
        )
        self.assertEqual(input_transform_kwargs["d"], 5)
        # task feature should be omitted
        self.assertEqual(input_transform_kwargs["indices"], [0, 1, 2, 3])

        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={
                "bounds": None,
            },
        )

        self.assertEqual(input_transform_kwargs["d"], 4)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1, 3])
        self.assertTrue(input_transform_kwargs["bounds"] is None)

    def test_argparse_normalize_contextual(self) -> None:
        dataset = ContextualDataset(
            datasets=[
                SupervisedDataset(
                    X=torch.ones(3, 4),
                    Y=torch.zeros(3, 1),
                    feature_names=["x_a", "y_a", "x_b", "y_b"],
                    outcome_names=["metric_a"],
                )
            ],
            parameter_decomposition={"a": ["x_a", "y_a"], "b": ["x_b", "y_b"]},
        )
        search_space_digest = SearchSpaceDigest(
            feature_names=["x_a", "y_a", "x_b", "y_b"],
            bounds=[(0.0, 1.0), (2.0, 3.0), (0.0, 1.0), (2.0, 3.0)],
        )
        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=dataset,
            search_space_digest=search_space_digest,
        )
        self.assertEqual(len(input_transform_kwargs), 3)
        self.assertEqual(input_transform_kwargs["d"], 3)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1])
        self.assertAllClose(
            input_transform_kwargs["bounds"],
            torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        )

    def test_argparse_warp(self) -> None:
        self.search_space_digest.task_features = [0, 3]
        input_transform_kwargs = input_transform_argparse(
            Warp,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
        )

        self.assertEqual(input_transform_kwargs["indices"], [1, 2])
        self.assertEqual(input_transform_kwargs["d"], 4)
        self.assertTrue(
            torch.equal(
                input_transform_kwargs["bounds"],
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]]),
            )
        )

        input_transform_kwargs = input_transform_argparse(
            Warp,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={"indices": [0, 1]},
        )
        self.assertEqual(
            input_transform_kwargs["indices"],
            [0, 1],
        )
        self.assertEqual(input_transform_kwargs["d"], 4)
        self.assertTrue(
            torch.equal(
                input_transform_kwargs["bounds"],
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]]),
            )
        )
        input_transform_kwargs = input_transform_argparse(
            Warp,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={"indices": [0, 1]},
            torch_dtype=torch.float64,
        )
        self.assertTrue(
            torch.equal(
                input_transform_kwargs["bounds"],
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]], dtype=torch.float64),
            )
        )

    def test_argparse_filter_features(self) -> None:
        # Test with no input_transform_options - should return all feature indices
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
        )
        all_feature_indices = torch.arange(
            len(self.dataset.feature_names), dtype=torch.int64
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(input_transform_kwargs["feature_indices"], all_feature_indices)
        )

        # Test with empty input_transform_options - should return all feature indices
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={},
        )
        all_feature_indices = torch.arange(
            len(self.dataset.feature_names), dtype=torch.int64
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(input_transform_kwargs["feature_indices"], all_feature_indices)
        )

        # Test with explicit feature_indices - should pass through unchanged
        feature_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={"feature_indices": feature_indices},
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(input_transform_kwargs["feature_indices"], feature_indices)
        )
        # Test with ignored_params - should convert to feature_indices
        ignored_params = ["x1", "x3"]
        expected_feature_indices = torch.tensor(
            [0, 2], dtype=torch.int64
        )  # Keep x0 and x2, ignore x1 and x3
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={"ignored_params": ignored_params},
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(
                input_transform_kwargs["feature_indices"], expected_feature_indices
            )
        )

        # Test with empty ignored_params - should return all feature indices
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={"ignored_params": []},
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(input_transform_kwargs["feature_indices"], all_feature_indices)
        )

        # Test when both feature_indices and ignored_params are specified - and
        # they are consistent
        feature_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
        input_transform_kwargs = input_transform_argparse(
            FilterFeatures,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={
                "feature_indices": feature_indices,
                "ignored_params": ["x1"],
            },
        )
        self.assertEqual(set(input_transform_kwargs.keys()), {"feature_indices"})
        self.assertTrue(
            torch.equal(input_transform_kwargs["feature_indices"], feature_indices)
        )

        # Test when both feature_indices and ignored_params are specified - and
        # they are inconsistent
        feature_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
        with self.assertRaisesRegex(
            ValueError,
            r"Filtered features passed in by feature_indices .* "
            r"is inconsistent with filtered feature indices computed from "
            r"ignored_params",
        ):
            input_transform_kwargs = input_transform_argparse(
                FilterFeatures,
                dataset=self.dataset,
                search_space_digest=self.search_space_digest,
                input_transform_options={
                    "feature_indices": feature_indices,
                    "ignored_params": ["x0", "x1"],
                },
            )

    def test_argparse_learned_feature_imputation(self) -> None:
        task_feature_name = Keys.TASK_FEATURE_NAME.value

        # Task 0 (target): features x0, x1, x2, x3 + task
        dataset_target = SupervisedDataset(
            X=torch.cat([torch.rand(5, 4), torch.zeros(5, 1)], dim=-1),
            Y=torch.randn(5, 1),
            feature_names=["x0", "x1", "x2", "x3", task_feature_name],
            outcome_names=["y0"],
        )
        # Task 1 (aux): features x0, x1 + task (subset of target features)
        dataset_aux = SupervisedDataset(
            X=torch.cat([torch.rand(5, 2), torch.ones(5, 1)], dim=-1),
            Y=torch.randn(5, 1),
            feature_names=["x0", "x1", task_feature_name],
            outcome_names=["y1"],
        )
        mtds = MultiTaskDataset(
            datasets=[dataset_target, dataset_aux],
            target_outcome_name="y0",
            task_feature_index=-1,
        )
        mt_ssd = dataclasses.replace(
            self.search_space_digest,
            feature_names=["x0", "x1", "x2", "x3", task_feature_name],
            task_features=[-1],
            bounds=[(0.0, 1.0)] * 5,
        )

        kwargs = input_transform_argparse(
            LearnedFeatureImputation,
            dataset=mtds,
            search_space_digest=mt_ssd,
        )

        # all_features uses target-first ordering: ["x0","x1","x2","x3"]
        self.assertEqual(kwargs["d"], 4)
        # Target has all 4 features, aux has x0, x1 -> indices [0, 1]
        self.assertEqual(kwargs["feature_indices"], {0: [0, 1, 2, 3], 1: [0, 1]})
        self.assertEqual(kwargs["task_feature_index"], -1)
        self.assertTrue(torch.equal(kwargs["bounds"][0], torch.zeros(4)))
        self.assertTrue(torch.equal(kwargs["bounds"][1], torch.ones(4)))
        self.assertEqual(kwargs["dtype"], torch.float64)

        with self.subTest("non_multitask_dataset_raises"):
            with self.assertRaisesRegex(ValueError, "requires a MultiTaskDataset"):
                input_transform_argparse(
                    LearnedFeatureImputation,
                    dataset=self.dataset,
                    search_space_digest=self.search_space_digest,
                )

        with self.subTest("homogeneous_features_raises"):
            homogeneous_ds = MultiTaskDataset(
                datasets=[
                    SupervisedDataset(
                        X=torch.rand(5, 3),
                        Y=torch.randn(5, 1),
                        feature_names=["x0", "x1", task_feature_name],
                        outcome_names=["y0"],
                    ),
                    SupervisedDataset(
                        X=torch.rand(5, 3),
                        Y=torch.randn(5, 1),
                        feature_names=["x0", "x1", task_feature_name],
                        outcome_names=["y1"],
                    ),
                ],
                target_outcome_name="y0",
                task_feature_index=-1,
            )
            with self.assertRaisesRegex(ValueError, "heterogeneous features"):
                input_transform_argparse(
                    LearnedFeatureImputation,
                    dataset=homogeneous_ds,
                    search_space_digest=mt_ssd,
                )

        with self.subTest("non_last_task_feature_raises"):
            bad_ds = MultiTaskDataset(
                datasets=[
                    SupervisedDataset(
                        X=torch.rand(5, 5),
                        Y=torch.randn(5, 1),
                        feature_names=[task_feature_name, "x0", "x1", "x2", "x3"],
                        outcome_names=["y0"],
                    ),
                    SupervisedDataset(
                        X=torch.rand(5, 3),
                        Y=torch.randn(5, 1),
                        feature_names=[task_feature_name, "x0", "x1"],
                        outcome_names=["y1"],
                    ),
                ],
                target_outcome_name="y0",
                task_feature_index=0,
            )
            with self.assertRaisesRegex(
                NotImplementedError, "task_feature_index == -1"
            ):
                input_transform_argparse(
                    LearnedFeatureImputation,
                    dataset=bad_ds,
                    search_space_digest=mt_ssd,
                )

        with self.subTest("options_override"):
            kwargs = input_transform_argparse(
                LearnedFeatureImputation,
                dataset=mtds,
                search_space_digest=mt_ssd,
                torch_dtype=torch.float32,
            )
            self.assertEqual(kwargs["dtype"], torch.float32)

    def test_argparse_learned_feature_imputation_feature_ordering(self) -> None:
        """Test that feature ordering preserves target's order, not alphabetical."""
        task_feature_name = Keys.TASK_FEATURE_NAME.value

        with self.subTest("target_order_preserved_not_alphabetical"):
            # Target: features C, A, B (NOT alphabetical)
            target_ds = SupervisedDataset(
                X=torch.cat([torch.rand(3, 3), torch.zeros(3, 1)], dim=-1),
                Y=torch.randn(3, 1),
                feature_names=["C", "A", "B", task_feature_name],
                outcome_names=["target"],
            )
            # Source: features A, B (subset, different order)
            source_ds = SupervisedDataset(
                X=torch.cat([torch.rand(2, 2), torch.ones(2, 1)], dim=-1),
                Y=torch.randn(2, 1),
                feature_names=["A", "B", task_feature_name],
                outcome_names=["source"],
            )
            mtds = MultiTaskDataset(
                datasets=[target_ds, source_ds],
                target_outcome_name="target",
                task_feature_index=-1,
            )
            ssd = dataclasses.replace(
                self.search_space_digest,
                feature_names=["C", "A", "B", task_feature_name],
                task_features=[-1],
                bounds=[(0.0, 1.0)] * 4,
            )
            kwargs = input_transform_argparse(
                LearnedFeatureImputation,
                dataset=mtds,
                search_space_digest=ssd,
            )
            # Canonical order should be C, A, B (target's order), not A, B, C
            self.assertEqual(kwargs["d"], 3)
            # Target: C, A, B -> [0, 1, 2]; Source: A, B -> [1, 2]
            self.assertEqual(kwargs["feature_indices"], {0: [0, 1, 2], 1: [1, 2]})

        with self.subTest("source_only_features_appended_at_end"):
            # Target: A, B; Source: B, C, D (C, D are source-only)
            target_ds = SupervisedDataset(
                X=torch.cat([torch.rand(3, 2), torch.zeros(3, 1)], dim=-1),
                Y=torch.randn(3, 1),
                feature_names=["A", "B", task_feature_name],
                outcome_names=["target"],
            )
            source_ds = SupervisedDataset(
                X=torch.cat([torch.rand(2, 3), torch.ones(2, 1)], dim=-1),
                Y=torch.randn(2, 1),
                feature_names=["B", "C", "D", task_feature_name],
                outcome_names=["source"],
            )
            mtds = MultiTaskDataset(
                datasets=[target_ds, source_ds],
                target_outcome_name="target",
                task_feature_index=-1,
            )
            ssd = dataclasses.replace(
                self.search_space_digest,
                feature_names=["A", "B", "C", "D", task_feature_name],
                task_features=[-1],
                bounds=[(0.0, 1.0)] * 5,
            )
            kwargs = input_transform_argparse(
                LearnedFeatureImputation,
                dataset=mtds,
                search_space_digest=ssd,
            )
            # Canonical order: A, B (target), then C, D (source-only, appended)
            self.assertEqual(kwargs["d"], 4)
            # Target: A, B -> [0, 1]; Source: B, C, D -> [1, 2, 3]
            self.assertEqual(kwargs["feature_indices"], {0: [0, 1], 1: [1, 2, 3]})
