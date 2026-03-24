#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.arm import Arm
from ax.core.data import Data, MAP_KEY
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial_status import TrialStatus
from ax.metrics.map_replay import MapDataReplayMetric, MapDataReplayState
from ax.runners.map_replay import MapDataReplayRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_test_map_data_experiment,
)
from pandas import DataFrame


class MapReplayRunnerTest(TestCase):
    def test_trial_lifecycle(self) -> None:
        """Test CANDIDATE -> RUNNING -> COMPLETED transitions, ABANDONED for
        unknown trials, and cursor advancement during polling."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        historical_data: Data = historical_experiment.lookup_data()
        state = MapDataReplayState(
            map_data=historical_data,
            metric_signatures=["branin_map"],
            step_size=1.0,
        )
        metric = MapDataReplayMetric(
            name="test_metric",
            replay_state=state,
            metric_signature="branin_map",
            lower_is_better=True,
        )
        runner = MapDataReplayRunner(replay_state=state)
        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=metric, minimize=True)
            ),
            runner=runner,
            tracking_metrics=[metric],
        )

        # Create 3 trials: 2 with data (indices 0, 1), 1 without (index 2)
        for _ in range(3):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))

        with self.subTest("unstarted_trials_are_candidates"):
            trial_status = runner.poll_trial_status(trials=experiment.trials.values())
            self.assertEqual(trial_status[TrialStatus.CANDIDATE], {0, 1, 2})

        # Start all trials
        for t in experiment.trials.values():
            t.run()
            self.assertTrue(t.run_metadata.get("replay_started"))

        with self.subTest("first_poll_running_and_abandoned"):
            trial_status = runner.poll_trial_status(trials=experiment.trials.values())
            # Trials 0, 1 have data -> RUNNING; trial 2 has no data -> ABANDONED
            self.assertEqual(trial_status[TrialStatus.RUNNING], {0, 1})
            self.assertIn(2, trial_status[TrialStatus.ABANDONED])

        with self.subTest("second_poll_completed"):
            trial_status = runner.poll_trial_status(trials=experiment.trials.values())
            self.assertEqual(trial_status[TrialStatus.COMPLETED], {0, 1})

    def test_cursor_advances_during_poll(self) -> None:
        """Test that the runner advances the cursor for running trials on each
        poll cycle."""
        map_data = Data(
            df=DataFrame(
                {
                    "trial_index": [0, 0, 0],
                    "arm_name": ["0_0", "0_0", "0_0"],
                    "metric_name": ["m1", "m1", "m1"],
                    "metric_signature": ["m1", "m1", "m1"],
                    "mean": [1.0, 2.0, 3.0],
                    "sem": [0.0, 0.0, 0.0],
                    MAP_KEY: [0.0, 0.5, 1.0],
                }
            )
        )
        state = MapDataReplayState(
            map_data=map_data, metric_signatures=["m1"], step_size=0.25
        )
        runner = MapDataReplayRunner(replay_state=state)

        experiment = Experiment(
            name="dummy",
            search_space=get_branin_search_space(),
            runner=runner,
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))
        trial.run()

        self.assertAlmostEqual(state._trial_cursors[0], 0.0)
        runner.poll_trial_status(trials=experiment.trials.values())
        self.assertAlmostEqual(state._trial_cursors[0], 0.25)
        runner.poll_trial_status(trials=experiment.trials.values())
        self.assertAlmostEqual(state._trial_cursors[0], 0.50)
