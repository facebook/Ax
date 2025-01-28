# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.regression_analysis import RegressionAnalysis
from ax.core.data import Data
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment_with_multi_objective


class TestRegressionAnalysis(TestCase):
    def test_regression_analysis(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=True, with_status_quo=True
        )

        df = pd.DataFrame(
            {
                "metric_name": ["branin_a"] * 6 + ["branin_b"] * 6,
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"] * 2,
                "trial_index": [0] * 12,
                "mean": list(np.arange(7, 1, -1)) + list(np.arange(6, 1, -1)) + [10.0],
                "sem": [1.0] * 12,
            }
        )

        experiment.attach_data(Data(df=df))
        ra = RegressionAnalysis(prob_threshold=0.90)
        card = ra.compute(experiment=experiment, generation_strategy=None)
        self.assertEqual(card.name, "RegressionAnalysis")
        self.assertEqual(card.title, "Ax Regression Analysis Warning")
        self.assertTrue(
            "0_4" in card.subtitle
            and "branin_b" in card.subtitle
            and "Trial 0" in card.subtitle
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)

        df = pd.DataFrame(
            {
                "metric_name": ["branin_a"] * 6 + ["branin_b"] * 6,
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"] * 2,
                "trial_index": [0] * 12,
                "mean": list(np.arange(7, 1, -1)) * 2,
                "sem": [1.0] * 12,
            }
        )
        experiment.attach_data(Data(df=df))
        ra = RegressionAnalysis(prob_threshold=0.90)
        card = ra.compute(experiment=experiment, generation_strategy=None)
        self.assertEqual(card.name, "RegressionAnalysis")
        self.assertEqual(card.title, "Ax Regression Analysis Success")
        self.assertEqual(
            card.subtitle,
            "No metric regessions detected.",
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
