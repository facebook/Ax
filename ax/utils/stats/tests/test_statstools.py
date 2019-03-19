#!/usr/bin/env python3

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import (
    benjamini_hochberg,
    inverse_variance_weight,
)


class BenjaminiHochbergTest(TestCase):
    def test_benjamini_hochberg(self):
        indcs = benjamini_hochberg(p_values=[0.01, 0.5, 0.3, 0.05], alpha=0.1)
        self.assertEqual(indcs, [0, 3])

    def test_benjamini_hochberg_invalid(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg(p_values=[2], alpha=0.1)


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
