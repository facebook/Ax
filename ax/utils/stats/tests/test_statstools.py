#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product

import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import (
    inverse_variance_weight,
    marginal_effects,
    relativize,
    relativize_data,
    unrelativize,
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
    def test_relativize_data(self) -> None:
        data = Data(
            df=pd.DataFrame(
                [
                    {
                        "mean": 2,
                        "sem": 0,
                        "metric_name": "foobar",
                        "arm_name": "status_quo",
                    },
                    {
                        "mean": 5,
                        "sem": 0,
                        "metric_name": "foobaz",
                        "arm_name": "status_quo",
                    },
                    {"mean": 1, "sem": 0, "metric_name": "foobar", "arm_name": "0_0"},
                    {"mean": 10, "sem": 0, "metric_name": "foobaz", "arm_name": "0_0"},
                ]
            )
        )
        expected_relativized_data = Data(
            df=pd.DataFrame(
                [
                    {
                        "mean": -0.5,
                        "sem": 0,
                        "metric_name": "foobar",
                        "arm_name": "0_0",
                    },
                    {"mean": 1, "sem": 0, "metric_name": "foobaz", "arm_name": "0_0"},
                ]
            )
        )
        expected_relativized_data_with_sq = Data(
            df=pd.DataFrame(
                [
                    {
                        "mean": 0,
                        "sem": 0,
                        "metric_name": "foobar",
                        "arm_name": "status_quo",
                    },
                    {
                        "mean": -0.5,
                        "sem": 0,
                        "metric_name": "foobar",
                        "arm_name": "0_0",
                    },
                    {
                        "mean": 0,
                        "sem": 0,
                        "metric_name": "foobaz",
                        "arm_name": "status_quo",
                    },
                    {"mean": 1, "sem": 0, "metric_name": "foobaz", "arm_name": "0_0"},
                ]
            )
        )

        actual_relativized_data = relativize_data(data=data)
        self.assertEqual(expected_relativized_data, actual_relativized_data)

        actual_relativized_data_with_sq = relativize_data(data=data, include_sq=True)
        self.assertEqual(
            expected_relativized_data_with_sq, actual_relativized_data_with_sq
        )


class UnrelativizeTest(TestCase):
    def test_unrelativize(self) -> None:
        means_t = np.array([-100.0, 101.0, 200.0, 300.0, 400.0])
        sems_t = np.array([2.0, 3.0, 2.0, 4.0, 0.0])
        mean_c = 200.0
        sem_c = 2.0

        for bias_correction, cov_means, as_percent, control_as_constant in product(
            (True, False, None),
            (0.5, 0.0),
            (True, False, None),
            (True, False, None),
        ):
            rel_mean_t, rel_sems_t = relativize(
                means_t,
                sems_t,
                mean_c,
                sem_c,
                cov_means=cov_means,
                # pyre-fixme[6]: For 6th argument expected `bool` but got
                #  `Optional[bool]`.
                bias_correction=bias_correction,
                # pyre-fixme[6]: For 7th argument expected `bool` but got
                #  `Optional[bool]`.
                as_percent=as_percent,
                # pyre-fixme[6]: For 8th argument expected `bool` but got
                #  `Optional[bool]`.
                control_as_constant=control_as_constant,
            )
            unrel_mean_t, unrel_sems_t = unrelativize(
                rel_mean_t,
                rel_sems_t,
                mean_c,
                sem_c,
                cov_means=cov_means,
                # pyre-fixme[6]: For 6th argument expected `bool` but got
                #  `Optional[bool]`.
                bias_correction=bias_correction,
                # pyre-fixme[6]: For 7th argument expected `bool` but got
                #  `Optional[bool]`.
                as_percent=as_percent,
                # pyre-fixme[6]: For 8th argument expected `bool` but got
                #  `Optional[bool]`.
                control_as_constant=control_as_constant,
            )
            self.assertTrue(np.allclose(means_t, unrel_mean_t))
            self.assertTrue(np.allclose(sems_t, unrel_sems_t))
