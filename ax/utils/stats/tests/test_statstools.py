#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import inverse_variance_weight, marginal_effects


class InverseVarianceWeightingTest(TestCase):
    def test_bad_arg_ivw(self) -> None:
        with self.assertRaises(ValueError):
            inverse_variance_weight(
                np.array([0]), np.array([1]), conflicting_noiseless="foo"
            )
        with self.assertRaises(ValueError):
            inverse_variance_weight(np.array([1, 2]), np.array([1]))

    def test_very_simple_ivw(self) -> None:
        means = np.array([1, 1, 1])
        variances = np.array([1, 1, 1])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.0)
        self.assertEqual(new_var, 1 / 3)

    def test_simple_ivw(self) -> None:
        means = np.array([1, 2, 3])
        variances = np.array([1, 1, 1])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 2.0)
        self.assertEqual(new_var, 1 / 3)

    def test_another_simple_ivw(self) -> None:
        means = np.array([1, 3])
        variances = np.array([1, 3])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.5)
        self.assertEqual(new_var, 0.75)

    def test_conflicting_noiseless_ivw(self) -> None:
        means = np.array([1, 2, 1])
        variances = np.array([0, 0, 1])

        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.5)
        self.assertEqual(new_var, 0.0)

        with self.assertRaises(ValueError):
            inverse_variance_weight(means, variances, conflicting_noiseless="raise")


class MarginalEffectsTest(TestCase):
    def test_marginal_effects(self) -> None:
        df = pd.DataFrame(
            {
                "mean": [1, 2, 3, 4],
                "sem": [0.1, 0.1, 0.1, 0.1],
                "factor_1": ["a", "a", "b", "b"],
                "factor_2": ["A", "B", "A", "B"],
            }
        )
        fx = marginal_effects(df)
        self.assertTrue(
            np.allclose(
                fx["Beta"].values, [-40.024, 39.944, -20.032, 19.952], atol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(fx["SE"].values, [2.154, 2.154, 2.040, 2.040], atol=1e-3)
        )

    def test_marginal_effects_se_matches_simulation(self) -> None:
        # The group means are part of the overall mean, so the standard errors
        # must account for both the overall SEM and their covariance.
        means = np.array([1.0, 2.0, 3.0, 4.0])
        sems = np.array([0.05, 0.2, 0.1, 0.3])
        df = pd.DataFrame({"mean": means, "sem": sems, "factor": ["a", "a", "b", "b"]})
        fx = marginal_effects(df)

        rng = np.random.default_rng(0)
        draws = means + sems * rng.standard_normal((200_000, 4))
        weights = 1 / sems**2
        overall = draws @ weights / weights.sum()
        for level, idx in (("a", [0, 1]), ("b", [2, 3])):
            group = draws[:, idx] @ weights[idx] / weights[idx].sum()
            simulated_se = np.std(100 * (group / overall - 1))
            se = fx.loc[fx["Level"] == level, "SE"].item()
            self.assertAlmostEqual(se, simulated_se, delta=0.02 * simulated_se)
