#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial_status import TrialStatus
from ax.metrics.map_replay import MapDataReplayMetric
from ax.runners.map_replay import MapDataReplayRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_test_map_data_experiment,
)
from pyre_extensions import assert_is_instance


class MapReplayRunnerTest(TestCase):
    def test_map_replay(self) -> None:
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        historical_data: MapData = assert_is_instance(
            historical_experiment.lookup_data(), MapData
        )
        metric = MapDataReplayMetric(
            name="test_metric",
            map_data=historical_data,
            metric_name="branin_map",
            lower_is_better=True,
        )
        runner = MapDataReplayRunner(
            replay_metric=metric,
        )
        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=metric,
                    minimize=True,
                )
            ),
            runner=runner,
        )
        for _ in range(2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))
            trial.run()
            self.assertTrue(trial.run_metadata.get("replay_started"))

        # After 1 fetch, both trials should still be running since there is
        # still data available to replay
        experiment.fetch_data()
        trial_status = runner.poll_trial_status(trials=experiment.trials.values())
        self.assertEqual(trial_status[TrialStatus.RUNNING], {0, 1})

        # After 2 fetches, there is no data left to replay and both trials should
        # be completed
        experiment.fetch_data()
        trial_status = runner.poll_trial_status(trials=experiment.trials.values())
        self.assertEqual(trial_status[TrialStatus.COMPLETED], {0, 1})
