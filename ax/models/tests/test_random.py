#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.models.random.base import RandomModel
from ax.utils.common.testutils import TestCase


class RandomModelTest(TestCase):
    def setUp(self) -> None:
        self.random_model = RandomModel()

    def testRandomModelGenSamples(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_samples(n=1, tunable_d=1)

    def testRandomModelGenUnconstrained(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_unconstrained(
                n=1, d=2, tunable_feature_indices=np.array([])
            )

    def testConvertEqualityConstraints(self) -> None:
        fixed_features = {3: 0.7, 1: 0.5}
        d = 4
        # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Tensor, Tensor]]` into 2
        #  values.
        C, c = self.random_model._convert_equality_constraints(d, fixed_features)
        c_expected = torch.tensor([[0.5], [0.7]], dtype=torch.double)
        C_expected = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.double)
        c_comparison = c == c_expected
        C_comparison = C == C_expected
        self.assertEqual(c_comparison.any(), True)
        self.assertEqual(C_comparison.any(), True)
        self.assertEqual(self.random_model._convert_equality_constraints(d, None), None)

    def testConvertInequalityConstraints(self) -> None:
        A = np.array([[1, 2], [3, 4]])
        b = np.array([[5], [6]])
        # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Tensor, Tensor]]` into 2
        #  values.
        A_result, b_result = self.random_model._convert_inequality_constraints((A, b))
        A_expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
        b_expected = torch.tensor([[5], [6]], dtype=torch.double)
        A_comparison = A_result == A_expected
        b_comparison = b_result == b_expected
        self.assertEqual(A_comparison.any(), True)
        self.assertEqual(b_comparison.any(), True)
        self.assertEqual(self.random_model._convert_inequality_constraints(None), None)

    def testConvertBounds(self) -> None:
        bounds = [(1, 2), (3, 4), (5, 6)]
        # pyre-fixme[6]: For 1st param expected `List[Tuple[float, float]]` but got
        #  `List[Tuple[int, int]]`.
        bounds_result = self.random_model._convert_bounds(bounds)
        bounds_expected = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.double)
        bounds_comparison = bounds_result == bounds_expected
        # pyre-fixme[16]: `bool` has no attribute `any`.
        self.assertEqual(bounds_comparison.any(), True)
        # pyre-fixme[6]: For 1st param expected `List[Tuple[float, float]]` but got
        #  `None`.
        self.assertEqual(self.random_model._convert_bounds(None), None)

    def testGetLastPoint(self) -> None:
        generated_points = np.array([[1, 2, 3], [4, 5, 6]])
        RandomModelWithPoints = RandomModel(generated_points=generated_points)
        result = RandomModelWithPoints._get_last_point()
        expected = torch.tensor([[4], [5], [6]])
        comparison = result == expected
        # pyre-fixme[16]: `bool` has no attribute `any`.
        self.assertEqual(comparison.any(), True)
