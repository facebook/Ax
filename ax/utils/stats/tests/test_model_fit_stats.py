#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.stats.model_fit_stats import _fisher_exact_test_p, entropy_of_observations
from scipy.stats import fisher_exact


class TestModelFitStats(TestCase):
    def test_entropy_of_observations(self) -> None:
        np.random.seed(1234)
        n = 16
        yc = np.ones(n)
        yc[: n // 2] = -1
        yc += np.random.randn(n) * 0.05
        yr = np.random.randn(n)

        # standardize both observations
        yc = yc / yc.std()
        yr = yr / yr.std()

        ones = np.ones(n)

        # compute entropy of observations
        ec = entropy_of_observations(y_obs=yc, y_pred=ones, se_pred=ones, bandwidth=0.1)
        er = entropy_of_observations(y_obs=yr, y_pred=ones, se_pred=ones, bandwidth=0.1)

        # testing that the Gaussian distributed data has a much larger entropy than
        # the clustered distribution
        self.assertTrue(er - ec > 10.0)

        ec2 = entropy_of_observations(
            y_obs=yc, y_pred=ones, se_pred=ones, bandwidth=0.2
        )
        er2 = entropy_of_observations(
            y_obs=yr, y_pred=ones, se_pred=ones, bandwidth=0.2
        )
        # entropy increases with larger bandwidth
        self.assertGreater(ec2, ec)
        self.assertGreater(er2, er)

        # ordering of entropies stays the same, though the difference is smaller
        self.assertTrue(er2 - ec2 > 3)

        # test warning if y is not standardized
        module_name = "ax.utils.stats.model_fit_stats"
        expected_warning = (
            "WARNING:ax.utils.stats.model_fit_stats:Standardization of observations "
            "was not applied. The default bandwidth of 0.1 is a reasonable "
            "choice if observations are standardize, but may not be otherwise."
        )
        with self.assertLogs(module_name, level="WARNING") as logger:
            ec = entropy_of_observations(y_obs=10 * yc, y_pred=ones, se_pred=ones)
            self.assertEqual(len(logger.output), 1)
            self.assertEqual(logger.output[0], expected_warning)

        with self.assertLogs(module_name, level="WARNING") as logger:
            ec = entropy_of_observations(y_obs=yc / 10, y_pred=ones, se_pred=ones)
            self.assertEqual(len(logger.output), 1)
            self.assertEqual(logger.output[0], expected_warning)

    def test_contingency_table_construction(self) -> None:
        # Create a dummy set of observations and predictions
        y_obs = np.array([1, 3, 2, 5, 7, 3])
        y_pred = np.array([2, 4, 1, 6, 8, 2.5])
        se_pred = np.full(len(y_obs), np.nan)  # not used for fisher exact

        # Compute ground truth contingency table
        true_table = np.array([[2, 1], [1, 2]])

        scipy_result = fisher_exact(true_table, alternative="greater")[1]
        ax_result = _fisher_exact_test_p(y_obs, y_pred, se_pred=se_pred)

        self.assertEqual(scipy_result, ax_result)
