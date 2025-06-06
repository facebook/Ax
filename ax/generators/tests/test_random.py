#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import torch
from ax.generators.random.base import RandomGenerator
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class RandomGeneratorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.random_model = RandomGenerator()

    def test_properties(self) -> None:
        self.assertFalse(self.random_model.can_predict)
        self.assertFalse(self.random_model.can_model_in_sample)

    def test_seed(self) -> None:
        # With manual seed.
        random_model = RandomGenerator(seed=5)
        self.assertEqual(random_model.seed, 5)
        # With no seed.
        self.assertIsInstance(self.random_model.seed, int)

    def test_state(self) -> None:
        for model in (self.random_model, RandomGenerator(seed=5)):
            state = model._get_state()
            self.assertEqual(state["seed"], model.seed)
            self.assertEqual(state["init_position"], model.init_position)

    def test_RandomGeneratorGenSamples(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_samples(n=1, tunable_d=1)

    def test_RandomGeneratorGenUnconstrained(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.random_model._gen_unconstrained(
                n=1, d=2, tunable_feature_indices=np.array([])
            )

    def test_ConvertEqualityConstraints(self) -> None:
        fixed_features = {3: 0.7, 1: 0.5}
        d = 4
        C, c = none_throws(
            self.random_model._convert_equality_constraints(d, fixed_features)
        )
        c_expected = torch.tensor([[0.5], [0.7]], dtype=torch.double)
        C_expected = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.double)
        c_comparison = c == c_expected
        C_comparison = C == C_expected
        self.assertEqual(c_comparison.any(), True)
        self.assertEqual(C_comparison.any(), True)
        self.assertEqual(self.random_model._convert_equality_constraints(d, None), None)

    def test_ConvertInequalityConstraints(self) -> None:
        A = np.array([[1, 2], [3, 4]])
        b = np.array([[5], [6]])
        A_result, b_result = none_throws(
            self.random_model._convert_inequality_constraints((A, b))
        )
        A_expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.double)
        b_expected = torch.tensor([[5], [6]], dtype=torch.double)
        A_comparison = A_result == A_expected
        b_comparison = b_result == b_expected
        self.assertEqual(A_comparison.any(), True)
        self.assertEqual(b_comparison.any(), True)
        self.assertEqual(self.random_model._convert_inequality_constraints(None), None)

    def test_ConvertBounds(self) -> None:
        bounds = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        bounds_result = self.random_model._convert_bounds(bounds)
        bounds_expected = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.double)
        bounds_comparison = bounds_result == bounds_expected
        # pyre-fixme[16]: `bool` has no attribute `any`.
        self.assertEqual(bounds_comparison.any(), True)
        # pyre-fixme[6]: For 1st param expected `List[Tuple[float, float]]` but got
        #  `None`.
        self.assertEqual(self.random_model._convert_bounds(None), None)
