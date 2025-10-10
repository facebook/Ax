#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import (
    inverse_variance_weight,
    marginal_effects,
    relativize_data,
)


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
        self.assertTrue(np.allclose(fx["Beta"].values, [-40, 40, -20, 20], atol=1e-3))
        self.assertTrue(np.allclose(fx["SE"].values, [2.83] * 4, atol=1e-2))


class RelativizeDataTest(TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": 2,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 5,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 10,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )

        self.expected_relativized_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": -0.5,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )
        self.expected_relativized_df_with_sq = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "mean": 0,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": -0.5,
                    "sem": 0,
                    "metric_name": "foobar",
                    "metric_signature": "foobar",
                    "arm_name": "0_0",
                },
                {
                    "trial_index": 0,
                    "mean": 0,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "status_quo",
                },
                {
                    "trial_index": 0,
                    "mean": 1,
                    "sem": 0,
                    "metric_name": "foobaz",
                    "metric_signature": "foobaz",
                    "arm_name": "0_0",
                },
            ]
        )

    def test_relativize_data(self) -> None:
        data = Data(
            df=self.df,
        )
        expected_relativized_data = Data(df=self.expected_relativized_df)

        expected_relativized_data_with_sq = Data(
            df=self.expected_relativized_df_with_sq
        )

        actual_relativized_data = relativize_data(data=data)
        self.assertEqual(expected_relativized_data, actual_relativized_data)

        actual_relativized_data_with_sq = relativize_data(data=data, include_sq=True)
        self.assertEqual(
            expected_relativized_data_with_sq, actual_relativized_data_with_sq
        )

    def test_relativize_data_no_sem(self) -> None:
        df = self.df.copy()
        df["sem"] = np.nan
        data = Data(df=df)

        expected_relativized_df = self.expected_relativized_df.copy()
        expected_relativized_df["sem"] = np.nan
        expected_relativized_data = Data(df=expected_relativized_df)

        expected_relativized_df_with_sq = self.expected_relativized_df_with_sq.copy()
        expected_relativized_df_with_sq.loc[
            expected_relativized_df_with_sq["arm_name"] != "status_quo", "sem"
        ] = np.nan
        expected_relativized_data_with_sq = Data(df=expected_relativized_df_with_sq)

        actual_relativized_data = relativize_data(data=data)
        self.assertEqual(expected_relativized_data, actual_relativized_data)

        actual_relativized_data_with_sq = relativize_data(data=data, include_sq=True)
        self.assertEqual(
            expected_relativized_data_with_sq, actual_relativized_data_with_sq
        )
