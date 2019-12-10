#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from unittest.mock import MagicMock

import numpy as np
from ax.models.model_utils import best_observed_point, check_duplicate
from ax.utils.common.testutils import TestCase


class ModelUtilsTest(TestCase):
    def setUp(self):
        pass

    def testBestObservedPoint(self):
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

        objective_weights = np.array([-1.0, 1.0, 0.0])
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
                objective_weights=np.zeros(3),
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
                objective_weights=np.zeros(3),
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features={1: 100},
                options={"method": "feasible_threshold"},
            )

    def testCheckDuplicate(self):
        duplicate_point = np.array([0, 1])
        not_duplicate_point = np.array([9, 9])
        points = np.array([[0, 1], [0, 2], [0, 1]])
        self.assertTrue(check_duplicate(duplicate_point, points))
        self.assertFalse(check_duplicate(not_duplicate_point, points))
