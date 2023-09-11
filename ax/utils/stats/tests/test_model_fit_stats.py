#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.stats.model_fit_stats import _fisher_exact_test_p
from scipy.stats import fisher_exact


class FisherExactTestTest(TestCase):
    def test_contingency_table_construction(self) -> None:
        # Create a dummy set of observations and predictions
        y_obs = np.array([1, 3, 2, 5, 7, 3])
        y_pred = np.array([2, 4, 1, 6, 8, 2.5])

        # Compute ground truth contingency table
        true_table = np.array([[2, 1], [1, 2]])

        scipy_result = fisher_exact(true_table, alternative="greater")[1]
        ax_result = _fisher_exact_test_p(y_obs, y_pred, se_pred=None)

        self.assertEqual(scipy_result, ax_result)
