#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import inverse_variance_weight, marginal_effects


class InverseVarianceWeightingTest(TestCase):
    def test_bad_arg_ivw(self):
        with self.assertRaises(ValueError):
            inverse_variance_weight(
                np.array([0]), np.array([1]), conflicting_noiseless="foo"
            )
        with self.assertRaises(ValueError):
            inverse_variance_weight(np.array([1, 2]), np.array([1]))

    def test_very_simple_ivw(self):
        means = np.array([1, 1, 1])
        variances = np.array([1, 1, 1])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.0)
        self.assertEqual(new_var, 1 / 3)

    def test_simple_ivw(self):
        means = np.array([1, 2, 3])
        variances = np.array([1, 1, 1])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 2.0)
        self.assertEqual(new_var, 1 / 3)

    def test_another_simple_ivw(self):
        means = np.array([1, 3])
        variances = np.array([1, 3])
        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.5)
        self.assertEqual(new_var, 0.75)

    def test_conflicting_noiseless_ivw(self):
        means = np.array([1, 2, 1])
        variances = np.array([0, 0, 1])

        new_mean, new_var = inverse_variance_weight(means, variances)
        self.assertEqual(new_mean, 1.5)
        self.assertEqual(new_var, 0.0)

        with self.assertRaises(ValueError):
            inverse_variance_weight(means, variances, conflicting_noiseless="raise")


class MarginalEffectsTest(TestCase):
    def test_marginal_effects(self):
        df = pd.DataFrame(
            {
                "mean": [1, 2, 3, 4],
                "sem": [0.1, 0.1, 0.1, 0.1],
                "factor_1": ["a", "a", "b", "b"],
                "factor_2": ["A", "B", "A", "B"],
            }
        )
        fx = marginal_effects(df)
        self.assertTrue(np.allclose(fx["Beta"].values, [-40, 40, -20, 20], atol=1e-3))
        self.assertTrue(np.allclose(fx["SE"].values, [2.83] * 4, atol=1e-2))
