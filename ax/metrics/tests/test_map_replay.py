#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.map_data import MAP_KEY, MapData
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.map_replay import MapDataReplayMetric

from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_test_map_data_experiment,
)
from pyre_extensions import assert_is_instance


class MapDataReplayMetricTest(TestCase):
    def test_map_replay(self) -> None:
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        historical_data: MapData = assert_is_instance(
            historical_experiment.lookup_data(), MapData
        )
        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=MapDataReplayMetric(
                        name="test_metric",
                        map_data=historical_data,
                        metric_name="branin_map",
                        lower_is_better=True,
                    ),
                    minimize=True,
                )
            ),
            runner=SyntheticRunner(),
        )
        for _ in range(0, 2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))
            trial.run()

        # fetch once for MAP_KEY = 0
        experiment.fetch_data()
        # the second fetch will be for MAP_KEY = 0 and MAP_KEY = 1
        data = experiment.fetch_data()
        self.assertTrue(
            np.allclose(
                # pyre-fixme[16]: `Data` has no attribute `map_df`.
                data.map_df["mean"].to_numpy(),
                np.array([146.138620, 117.388086, 113.057480, 90.815154]),
            )
        )

    def test_map_replay_non_uniform(self) -> None:
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        map_data: MapData = assert_is_instance(
            historical_experiment.lookup_data(), MapData
        )
        map_df = map_data.map_df
        map_df[MAP_KEY] = pd.Series([0.25, 0.95, 0.0, 0.25, 1.0, 0.0])
        historical_data = MapData(df=map_df)
        replay_metric = MapDataReplayMetric(
            name="test_metric",
            map_data=historical_data,
            metric_name="branin_map",
            lower_is_better=True,
        )
        self.assertEqual(replay_metric.offset, 0.25)
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
        for _ in range(0, 2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))
            trial.run()

        # Test that as we step through with steps of size 0.3625, we
        # first get both points at step 0.25.
        data: MapData = assert_is_instance(experiment.fetch_data(), MapData)
        self.assertEqual(len(data.map_df), 2)

        # Next, we add the point at step 0.95 of Trial 0.
        data: MapData = assert_is_instance(experiment.fetch_data(), MapData)
        self.assertEqual(len(data.map_df), 3)

        # Finally, we get the point at step 1.0 of Trial 1.
        data: MapData = assert_is_instance(experiment.fetch_data(), MapData)
        self.assertEqual(len(data.map_df), 4)
