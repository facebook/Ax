#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
from ax.core.search_space import SearchSpaceDigest
from ax.generators.utils import (
    best_observed_point,
    enumerate_discrete_combinations,
    mk_discrete_choices,
    rejection_sample,
    remove_duplicates,
)
from ax.utils.common.testutils import TestCase


class UtilsTest(TestCase):
    def test_BestObservedPoint(self) -> None:
        model = MagicMock()

        X1 = np.array(list(product(np.arange(0.0, 10.0), np.arange(0.0, 10.0))))
        X2 = np.array(list(product(np.arange(5.0, 15.0), np.arange(5.0, 15.0))))
        # Overlap of 5x5=25 points
        X3 = np.array(list(product(np.arange(20.0, 30.0), np.arange(20.0, 30.0))))
        # X3 not used in objective or constraints
        model.Xs = [X1, X2, X3]

        bounds = [(0.0, 8.0), (0.0, 8.0)]  # Filters to 4x4=16 points
        fixed_features = {1: 6.0}  # Filters to 4 points
        linear_constraints = (
            np.array([[2.0, 2.0], [0.0, 1.0]]),
            np.array([[27.0], [7.0]]),
        )
        # Filters to 3

        objective_weights = np.array([[-1.0, 1.0, 0.0]])
        outcome_constraints = (
            np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            np.array([[10.0], [24.0]]),
        )

        # f and cov constructed to give objectives [0, 4, 6] and pfeas [1, 0.5, 0.25]
        f = np.array([[1.0, 1.0, -1.0], [6.0, 10.0, -1.0], [5.0, 11.0, -1.0]])
        cov = np.tile(np.diag([1, 1, 1]), (3, 1, 1))
        model.predict.return_value = (f, cov)

        # Test with defaults
        xbest = best_observed_point(
            model=model,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
        X_obs = model.predict.mock_calls[0][1][0]
        self.assertEqual(X_obs.shape, (3, 2))
        # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing...
        self.assertTrue(np.array_equal(X_obs[1, :], xbest))  # 1 should be best

        # Test with specified utility baseline
        xbest = best_observed_point(
            model=model,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options={"utility_baseline": 4.0},
        )
        X_obs = model.predict.mock_calls[1][1][0]
        self.assertEqual(X_obs.shape, (3, 2))
        # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing...
        self.assertTrue(np.array_equal(X_obs[2, :], xbest))  # 2 should be best

        # Test with feasibility threshold
        xbest = best_observed_point(
            model=model,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options={"best_point_method": "feasible_threshold"},
        )
        X_obs = model.predict.mock_calls[2][1][0]
        self.assertEqual(X_obs.shape, (3, 2))
        # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing...
        self.assertTrue(np.array_equal(X_obs[0, :], xbest))  # 0 should be best

        # Parameter infeasible
        xbest = best_observed_point(
            model=model,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features={1: 100},
            options={"best_point_method": "feasible_threshold"},
        )
        self.assertIsNone(xbest)

        # Outcome infeasible
        xbest = best_observed_point(
            model=model,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=(np.array([[1.0, 0.0, 0.0]]), np.array([[-100.0]])),
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options={"best_point_method": "feasible_threshold"},
        )
        self.assertIsNone(xbest)

        # No objective.
        with self.assertRaises(ValueError):
            xbest = best_observed_point(
                model=model,
                bounds=bounds,
                objective_weights=np.zeros((1, 3)),
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features={1: 100},
                options={"method": "feasible_threshold"},
            )

        with self.assertRaises(ValueError):
            delattr(model, "Xs")
            xbest = best_observed_point(
                model=model,
                bounds=bounds,
                objective_weights=np.zeros((1, 3)),
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features={1: 100},
                options={"method": "feasible_threshold"},
            )

    def test_RemoveDuplicates(self) -> None:
        existing_points = np.array([[0, 1], [0, 2]])

        points_with_duplicates = np.array([[0, 1], [0, 2], [0, 3], [0, 1]])
        unique_points = remove_duplicates(points_with_duplicates)
        expected_points = np.array([[0, 1], [0, 2], [0, 3]])
        self.assertTrue(np.array_equal(expected_points, unique_points))

        unique_points = remove_duplicates(points_with_duplicates, existing_points)
        expected_points = np.array([[0, 3]])
        self.assertTrue(np.array_equal(expected_points, unique_points))

        points_without_duplicates = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        expected_points = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        unique_points = remove_duplicates(points_without_duplicates)
        self.assertTrue(np.array_equal(expected_points, unique_points))

        unique_points = remove_duplicates(points_without_duplicates, existing_points)
        expected_points = np.array([[0, 3], [0, 4]])
        self.assertTrue(np.array_equal(expected_points, unique_points))

    def test_MkDiscreteChoices(self) -> None:
        ssd1 = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 1), (0, 2)],
            ordinal_features=[1],
            discrete_choices={1: [0, 1, 2]},
        )
        dc1 = mk_discrete_choices(ssd1)
        self.assertEqual(dc1, {1: [0, 1, 2]})
        dc1_ff = mk_discrete_choices(ssd1, fixed_features={1: 0})
        self.assertEqual(dc1_ff, {1: [0]})
        ssd2 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 1), (0, 2), (3, 4)],
            ordinal_features=[1],
            categorical_features=[2],
            discrete_choices={1: [0, 1, 2], 2: [3, 4]},
        )
        dc2_ff = mk_discrete_choices(ssd2, fixed_features={1: 0})
        self.assertEqual(dc2_ff, {1: [0], 2: [3, 4]})

    def test_EnumerateDiscreteCombinations(self) -> None:
        dc1 = {1: [0, 1, 2]}
        dc1_enum = enumerate_discrete_combinations(dc1)
        self.assertEqual(dc1_enum, [{1: 0}, {1: 1}, {1: 2}])
        dc2 = {1: [0, 1, 2], 2: [3, 4]}
        dc2_enum = enumerate_discrete_combinations(dc2)
        self.assertEqual(
            dc2_enum,
            [
                {1: 0, 2: 3},
                {1: 0, 2: 4},
                {1: 1, 2: 3},
                {1: 1, 2: 4},
                {1: 2, 2: 3},
                {1: 2, 2: 4},
            ],
        )

    def test_rejection_sample(self) -> None:
        """Test rejection sampling with constraints."""
        # --- Post-rounding constraint enforcement ---
        with self.subTest("post_rounding_constraint_check"):
            # Constraint: x0 + x1 <= 1.5
            # We'll mock gen_unconstrained to return points that:
            # 1. (0.6, 0.6): sum=1.2 satisfies, but rounds to (1,1): sum=2 violates
            # 2. (0.4, 0.4): sum=0.8 satisfies, rounds to (0,0): sum=0 satisfies
            call_count = 0
            values_to_return: list[npt.NDArray[np.floating[Any]]] = [
                np.array([[0.6, 0.6]]),
                np.array([[0.4, 0.4]]),
            ]

            def mock_gen_unconstrained(
                n: int,
                d: int,
                tunable_feature_indices: npt.NDArray[np.intp],
                fixed_features: dict[int, float] | None,
            ) -> npt.NDArray[np.floating[Any]]:
                nonlocal call_count
                result = values_to_return[min(call_count, len(values_to_return) - 1)]
                call_count += 1
                return result

            def rounding_func(
                point: npt.NDArray[np.floating[Any]],
            ) -> npt.NDArray[np.floating[Any]]:
                return np.round(point)

            linear_constraints = (
                np.array([[1.0, 1.0]]),  # A
                np.array([[1.5]]),  # b: x0 + x1 <= 1.5
            )

            points, attempted = rejection_sample(
                gen_unconstrained=mock_gen_unconstrained,
                n=1,
                d=2,
                tunable_feature_indices=np.array([0, 1]),
                linear_constraints=linear_constraints,
                rounding_func=rounding_func,
            )

            # Should have attempted 2 draws: first was rejected after rounding
            self.assertEqual(attempted, 2)
            self.assertEqual(len(points), 1)
            # The returned point should be (0, 0) - rounded version of (0.4, 0.4)
            self.assertTrue(np.array_equal(points[0], np.array([0.0, 0.0])))

        # --- Basic rejection sampling without rounding ---
        with self.subTest("basic_without_rounding"):
            rng: np.random.Generator = np.random.default_rng(123)

            def gen_unconstrained_basic(
                n: int,
                d: int,
                tunable_feature_indices: npt.NDArray[np.intp],
                fixed_features: dict[int, float] | None,
                rng: np.random.Generator = rng,
            ) -> npt.NDArray[np.float64]:
                return rng.uniform(0, 1, size=(n, d))

            # Simple constraint: x0 <= 0.5
            basic_constraints = (
                np.array([[1.0, 0.0]]),  # A
                np.array([[0.5]]),  # b
            )

            points, attempted = rejection_sample(
                gen_unconstrained=gen_unconstrained_basic,
                n=10,
                d=2,
                tunable_feature_indices=np.array([0, 1]),
                linear_constraints=basic_constraints,
            )

            self.assertEqual(len(points), 10)
            for point in points:
                self.assertLessEqual(point[0], 0.5)
