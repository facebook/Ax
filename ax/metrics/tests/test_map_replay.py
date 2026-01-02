#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.core.arm import Arm
from ax.core.data import Data, MAP_KEY
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.map_replay import MapDataReplayMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_test_map_data_experiment,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal


class MapDataReplayMetricTest(TestCase):
    def test_map_replay(self) -> None:
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        historical_data: Data = historical_experiment.lookup_data()
        replay_metric = MapDataReplayMetric(
            name="test_metric",
            map_data=historical_data,
            metric_name="branin_map",
            lower_is_better=True,
        )

        # Verify offset and scaling factor for uniform step data.
        # The test data has 2 trials, each with 2 fetches, resulting in steps 0 and 1.
        # offset = min(first step of each trial) = min(0, 0) = 0
        self.assertEqual(replay_metric.offset, 0)
        # scaling_factor = mean((final_step - offset) / num_points)
        #                = mean((1 - 0) / 2, (1 - 0) / 2) = mean(0.5, 0.5) = 0.5
        self.assertEqual(replay_metric.scaling_factor, 0.5)

        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=replay_metric,
                    minimize=True,
                )
            ),
            runner=SyntheticRunner(),
        )
        for i in range(0, 2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": float(i), "x2": 0.0}))
            trial.run()

        # fetch once for MAP_KEY = 0
        experiment.fetch_data()
        # the second fetch will be for MAP_KEY = 0 and MAP_KEY = 1
        data = experiment.fetch_data()
        metric_name = [replay_metric.name] * 4
        expected_df = DataFrame(
            {
                "trial_index": [0, 0, 1, 1],
                "arm_name": ["0_0", "0_0", "1_0", "1_0"],
                "metric_name": metric_name,
                "metric_signature": metric_name,
                "mean": [146.138620, 117.388086, 113.057480, 90.815154],
                "sem": [0.0, 0.0, 0.0, 0.0],
                MAP_KEY: [0.0, 1.0, 0.0, 1.0],
            }
        )
        assert_frame_equal(data.full_df, expected_df)

    def test_map_replay_non_uniform(self) -> None:
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        full_df = historical_experiment.lookup_data().full_df
        # The original data has 6 rows: 4 for branin_map and 2 for branin.
        # After assinging steps, we have following steps for branin_map:
        # Trial 0: steps [0.25, 0.95]
        # Trial 1: steps [0.25, 1.0]
        full_df[MAP_KEY] = pd.Series([0.25, 0.95, 0.0, 0.25, 1.0, 0.0])
        historical_data = Data(df=full_df)
        replay_metric = MapDataReplayMetric(
            name="test_metric",
            map_data=historical_data,
            metric_name="branin_map",
            lower_is_better=True,
        )
        # Verify offset: min(first step of each trial after sorting)
        self.assertEqual(replay_metric.offset, 0.25)
        # Verify scaling_factor: mean((final_step - offset) / num_points) across trials
        # Trial 0: (0.95 - 0.25) / 2 = 0.35
        # Trial 1: (1.0 - 0.25) / 2 = 0.375
        # scaling_factor = (0.35 + 0.375) / 2 = 0.3625
        self.assertEqual(replay_metric.scaling_factor, 0.3625)

        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=replay_metric,
                    minimize=True,
                )
            ),
            runner=SyntheticRunner(),
        )
        for i in range(0, 2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": float(i), "x2": 0.0}))
            trial.run()

        metric_name = [replay_metric.name] * 4
        full_expected_df = DataFrame(
            {
                "trial_index": [0, 0, 1, 1],
                "arm_name": ["0_0", "0_0", "1_0", "1_0"],
                "metric_name": metric_name,
                "metric_signature": metric_name,
                "mean": [146.138620, 117.388086, 113.057480, 90.815154],
                "sem": [0.0, 0.0, 0.0, 0.0],
                MAP_KEY: [0.25, 0.95, 0.25, 1.0],
            }
        )

        # Test that as we step through with steps of size 0.3625, we
        # first get both points at step 0.25.
        data = experiment.fetch_data()
        assert_frame_equal(
            data.full_df, full_expected_df.iloc[[0, 2]].reset_index(drop=True)
        )

        # Next, we add the point at step 0.95 of Trial 0.
        data = experiment.fetch_data()
        assert_frame_equal(data.full_df, full_expected_df.iloc[:3])

        # Finally, we get the point at step 1.0 of Trial 1.
        data = experiment.fetch_data()
        assert_frame_equal(data.full_df, full_expected_df.iloc[:4])
