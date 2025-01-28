# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.analysis.healthcheck.regression_detection_utils import (
    detect_regressions_by_trial,
    detect_regressions_single_trial,
)
from ax.core.data import Data
from ax.modelbridge.factory import get_sobol
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_multi_objective,
    TEST_SOBOL_SEED,
)


class TestRegressionDetection(TestCase):
    def test_regression_detection_by_trial(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=True, with_status_quo=True
        )

        df0 = pd.DataFrame(
            {
                "metric_name": ["branin_a"] * 6 + ["branin_b"] * 6,
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"] * 2,
                "trial_index": [0] * 12,
                "mean": list(np.arange(7, 1, -1)) + list(np.arange(6, 1, -1)) + [10.0],
                "sem": [1.0] * 12,
            }
        )

        # zero size threshold and high probability threshold
        regressing_arms_metrics = detect_regressions_single_trial(
            experiment=experiment,
            thresholds={"branin_a": (0.0, 0.90), "branin_b": (0.0, 0.90)},
            data=Data(df=df0),
        )
        self.assertGreaterEqual(regressing_arms_metrics["0_4"]["branin_b"], 0.90)
        self.assertEqual(len(list(regressing_arms_metrics.keys())), 1)
        self.assertEqual(len(list(regressing_arms_metrics["0_4"].keys())), 1)
        self.assertGreaterEqual(regressing_arms_metrics["0_4"]["branin_b"], 0.90)

        # zero size threshold and different probability threshold
        regressing_arms_metrics = detect_regressions_single_trial(
            experiment=experiment,
            thresholds={"branin_a": (0, 0.10), "branin_b": (0, 0.95)},
            data=Data(df=df0),
        )
        self.assertGreaterEqual(regressing_arms_metrics["0_0"]["branin_a"], 0.10)
        self.assertEqual(len(list(regressing_arms_metrics.keys())), 1)
        self.assertEqual(len(list(regressing_arms_metrics["0_0"].keys())), 1)

        # different size thresholds
        regressing_arms_metrics = detect_regressions_single_trial(
            experiment=experiment,
            data=Data(df=df0),
            thresholds={"branin_a": (10.0, 0.50), "branin_b": (100, 0.50)},
        )
        self.assertEqual(len(regressing_arms_metrics), 0)

        # add one more trial
        sobol_generator = get_sobol(
            search_space=experiment.search_space, seed=TEST_SOBOL_SEED + 1
        )
        sobol_run = sobol_generator.gen(n=3)
        experiment.new_batch_trial(optimize_for_power=True).add_generator_run(sobol_run)

        df1 = pd.DataFrame(
            {
                "metric_name": ["branin_a"] * 4 + ["branin_b"] * 4,
                "arm_name": ["status_quo", "1_0", "1_1", "1_2"] * 2,
                "trial_index": [1] * 8,
                "mean": list(np.arange(5, 1, -1)) + list(np.arange(4, 1, -1)) + [20.0],
                "sem": [1.0] * 8,
            }
        )

        df = pd.concat([df0, df1], ignore_index=True)
        regressing_arms_metrics_by_trial = detect_regressions_by_trial(
            experiment=experiment,
            thresholds={"branin_a": (0.0, 0.90), "branin_b": (0.0, 0.90)},
            data=Data(df=df),
        )
        self.assertGreaterEqual(
            regressing_arms_metrics_by_trial[0]["0_4"]["branin_b"], 0.90
        )
        self.assertEqual(len(list(regressing_arms_metrics_by_trial[0].keys())), 1)
        self.assertEqual(
            len(list(regressing_arms_metrics_by_trial[0]["0_4"].keys())), 1
        )
        self.assertGreaterEqual(
            regressing_arms_metrics_by_trial[1]["1_2"]["branin_b"], 0.90
        )
        self.assertEqual(len(list(regressing_arms_metrics_by_trial[1].keys())), 1)
        self.assertEqual(
            len(list(regressing_arms_metrics_by_trial[1]["1_2"].keys())), 1
        )
