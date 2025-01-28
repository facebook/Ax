#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import torch
from ax.core.search_space import RobustSearchSpaceDigest, SearchSpaceDigest
from ax.models.torch.botorch_modular.input_constructors.input_transforms import (
    input_transform_argparse,
)
from ax.utils.common.testutils import TestCase
from botorch.models.transforms.input import (
    InputPerturbation,
    InputStandardize,
    InputTransform,
    Normalize,
    Warp,
)
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset


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
            robust_digest=None,
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
        self.assertEqual(input_transform_kwargs["d"], 3)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1])

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
            X=torch.rand(5, 4),
            Y=torch.randn(5, 1),
            feature_names=[f"x{i}" for i in range(4)],
            outcome_names=["y0"],
        )

        dataset2 = SupervisedDataset(
            X=torch.rand(5, 2),
            Y=torch.randn(5, 1),
            feature_names=[f"x{i}" for i in range(2)],
            outcome_names=["y1"],
        )
        mtds = MultiTaskDataset(datasets=[dataset1, dataset2], target_outcome_name="y0")
        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=mtds,
            search_space_digest=self.search_space_digest,
        )
        self.assertEqual(input_transform_kwargs["d"], 3)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1])

        input_transform_kwargs = input_transform_argparse(
            Normalize,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
            input_transform_options={
                "bounds": None,
            },
        )

        self.assertEqual(input_transform_kwargs["d"], 3)
        self.assertEqual(input_transform_kwargs["indices"], [0, 1])
        self.assertTrue(input_transform_kwargs["bounds"] is None)

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

    def test_argparse_input_perturbation(self) -> None:
        self.search_space_digest.robust_digest = RobustSearchSpaceDigest(
            sample_param_perturbations=lambda: np.zeros((2, 2)),
        )

        input_transform_kwargs = input_transform_argparse(
            InputPerturbation,
            dataset=self.dataset,
            search_space_digest=self.search_space_digest,
        )

        self.assertEqual(input_transform_kwargs["multiplicative"], False)

        self.assertTrue(
            torch.all(
                torch.isclose(
                    input_transform_kwargs["perturbation_set"],
                    torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
                )
            )
        )
