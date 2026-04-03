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
from ax.metrics.map_replay import MapDataReplayMetric, MapDataReplayState
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_test_map_data_experiment,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pyre_extensions import none_throws


def _make_map_data(
    trial_metric_data: dict[int, dict[str, list[tuple[float, float, float]]]],
) -> Data:
    """Helper to build map data from a nested dict.

    Args:
        trial_metric_data:
            {trial_index: {metric_name: [(step, mean, sem), ...]}}
    """
    rows = []
    for trial_index, metrics in trial_metric_data.items():
        for metric_name, points in metrics.items():
            for step, mean, sem in points:
                rows.append(
                    {
                        "trial_index": trial_index,
                        "arm_name": f"{trial_index}_0",
                        "metric_name": metric_name,
                        "metric_signature": metric_name,
                        "mean": mean,
                        "sem": sem,
                        MAP_KEY: step,
                    }
                )
    return Data(df=DataFrame(rows))


class MapDataReplayStateTest(TestCase):
    def test_state_computation(self) -> None:
        """Test min_prog, max_prog, and per_trial_max_prog for various data shapes."""
        with self.subTest("uniform_steps"):
            map_data = _make_map_data(
                {
                    0: {"m1": [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0)]},
                    1: {"m1": [(0.0, 3.0, 0.0), (1.0, 4.0, 0.0)]},
                }
            )
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state.min_prog, 0.0)
            self.assertEqual(state.max_prog, 1.0)
            self.assertEqual(state._per_trial_max_prog, {0: 1.0, 1: 1.0})

        with self.subTest("non_uniform_steps"):
            map_data = _make_map_data(
                {
                    0: {"m1": [(0.25, 1.0, 0.0), (0.95, 2.0, 0.0)]},
                    1: {"m1": [(0.25, 3.0, 0.0), (1.0, 4.0, 0.0)]},
                }
            )
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state.min_prog, 0.25)
            self.assertEqual(state.max_prog, 1.0)
            self.assertEqual(state._per_trial_max_prog, {0: 0.95, 1: 1.0})

        with self.subTest("multi_metric"):
            map_data = _make_map_data(
                {
                    0: {
                        "m1": [(0.0, 1.0, 0.0), (5.0, 2.0, 0.0)],
                        "m2": [(1.0, 3.0, 0.0), (10.0, 4.0, 0.0)],
                    },
                }
            )
            state = MapDataReplayState(
                map_data=map_data, metric_signatures=["m1", "m2"]
            )
            self.assertEqual(state.min_prog, 0.0)
            self.assertEqual(state.max_prog, 10.0)
            self.assertEqual(state._per_trial_max_prog, {0: 10.0})

        with self.subTest("single_trial"):
            map_data = _make_map_data({0: {"m1": [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0)]}})
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state._trial_indices, {0})
            self.assertTrue(state.has_trial_data(trial_index=0))
            self.assertFalse(state.has_trial_data(trial_index=1))

        with self.subTest("non_contiguous_trial_indices"):
            map_data = _make_map_data(
                {
                    0: {"m1": [(0.0, 1.0, 0.0)]},
                    5: {"m1": [(0.0, 2.0, 0.0)]},
                    10: {"m1": [(0.0, 3.0, 0.0)]},
                }
            )
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state._trial_indices, {0, 5, 10})
            self.assertTrue(state.has_trial_data(trial_index=5))
            self.assertFalse(state.has_trial_data(trial_index=3))

        with self.subTest("min_equals_max_prog"):
            map_data = _make_map_data(
                {
                    0: {"m1": [(3.0, 1.0, 0.0)]},
                    1: {"m1": [(3.0, 2.0, 0.0)]},
                }
            )
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state.min_prog, 3.0)
            self.assertEqual(state.max_prog, 3.0)
            self.assertTrue(state.is_trial_complete(trial_index=0))
            self.assertTrue(state.is_trial_complete(trial_index=1))
            self.assertEqual(
                len(state.get_data(trial_index=0, metric_signature="m1")), 1
            )

        with self.subTest("empty_metric_data"):
            map_data = _make_map_data({0: {"m1": [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0)]}})
            # Request a metric signature that has no data
            state = MapDataReplayState(
                map_data=map_data, metric_signatures=["m1", "m_empty"]
            )
            # m1 should be present, m_empty should return empty
            self.assertTrue(state.has_trial_data(trial_index=0))
            self.assertTrue(
                state.get_data(trial_index=0, metric_signature="m_empty").empty
            )
            # min/max should be computed from m1 data only
            self.assertEqual(state.min_prog, 0.0)
            self.assertEqual(state.max_prog, 1.0)

        with self.subTest("different_num_points_per_trial"):
            map_data = _make_map_data(
                {
                    0: {"m1": [(0.0, 1.0, 0.0), (0.5, 2.0, 0.0), (1.0, 3.0, 0.0)]},
                    1: {"m1": [(0.0, 4.0, 0.0)]},
                }
            )
            state = MapDataReplayState(map_data=map_data, metric_signatures=["m1"])
            self.assertEqual(state._per_trial_max_prog, {0: 1.0, 1: 0.0})

    def test_cursor_advancement_and_data_serving(self) -> None:
        """Test cursor advancement, capping, progressive data serving,
        per-trial independence, and trial completion transitions."""
        map_data = _make_map_data(
            {
                0: {"m1": [(0.0, 1.0, 0.0), (0.5, 2.0, 0.0), (1.0, 3.0, 0.0)]},
                1: {"m1": [(0.0, 4.0, 0.0), (1.0, 5.0, 0.0)]},
            }
        )
        state = MapDataReplayState(
            map_data=map_data, metric_signatures=["m1"], step_size=0.5
        )

        with self.subTest("initial_cursor_is_zero"):
            self.assertEqual(state._trial_cursors[0], 0.0)
            self.assertEqual(state._trial_cursors[1], 0.0)

        with self.subTest("progressive_data_at_cursor_0"):
            self.assertEqual(
                len(state.get_data(trial_index=0, metric_signature="m1")), 1
            )
            self.assertEqual(
                len(state.get_data(trial_index=1, metric_signature="m1")), 1
            )

        with self.subTest("advance_and_check_independence"):
            state.advance_trial(trial_index=0)
            self.assertAlmostEqual(state._trial_cursors[0], 0.5)
            self.assertAlmostEqual(state._trial_cursors[1], 0.0)

        with self.subTest("progressive_data_at_cursor_0_5"):
            self.assertEqual(
                len(state.get_data(trial_index=0, metric_signature="m1")), 2
            )
            self.assertEqual(
                len(state.get_data(trial_index=1, metric_signature="m1")), 1
            )

        with self.subTest("advance_to_full"):
            state.advance_trial(trial_index=0)
            self.assertEqual(
                len(state.get_data(trial_index=0, metric_signature="m1")), 3
            )

        with self.subTest("cursor_caps_at_one"):
            state.advance_trial(trial_index=0)
            self.assertAlmostEqual(state._trial_cursors[0], 1.0)

        with self.subTest("trial_completion_transitions"):
            self.assertTrue(state.is_trial_complete(trial_index=0))
            self.assertFalse(state.is_trial_complete(trial_index=1))
            state.advance_trial(trial_index=1)
            state.advance_trial(trial_index=1)
            self.assertTrue(state.is_trial_complete(trial_index=1))

        with self.subTest("heterogeneous_max_prog_completion"):
            # Trial with lower max_prog completes before trial with higher
            map_data = _make_map_data(
                {
                    0: {"m1": [(0.0, 1.0, 0.0), (0.5, 2.0, 0.0)]},
                    1: {"m1": [(0.0, 3.0, 0.0), (1.0, 4.0, 0.0)]},
                }
            )
            state = MapDataReplayState(
                map_data=map_data, metric_signatures=["m1"], step_size=0.5
            )
            # Global range [0.0, 1.0]; trial 0 max=0.5, trial 1 max=1.0
            state.advance_trial(trial_index=0)
            state.advance_trial(trial_index=1)
            # cursor=0.5, curr_prog=0.5: trial 0 complete, trial 1 not
            self.assertTrue(state.is_trial_complete(trial_index=0))
            self.assertFalse(state.is_trial_complete(trial_index=1))

    def test_multi_metric_and_data_integrity(self) -> None:
        """Test multi-metric shared timeline, original MAP_KEY preservation,
        and get_data for nonexistent trial/metric."""
        map_data = _make_map_data(
            {
                0: {
                    "m1": [(10.0, 1.0, 0.0), (20.0, 2.0, 0.0)],
                    "m2": [(10.0, 10.0, 0.0), (20.0, 20.0, 0.0)],
                },
            }
        )
        state = MapDataReplayState(
            map_data=map_data, metric_signatures=["m1", "m2"], step_size=1.0
        )
        state.advance_trial(trial_index=0)

        with self.subTest("shared_timeline"):
            self.assertEqual(
                len(state.get_data(trial_index=0, metric_signature="m1")),
                len(state.get_data(trial_index=0, metric_signature="m2")),
            )

        with self.subTest("original_map_key_values"):
            self.assertListEqual(
                state.get_data(trial_index=0, metric_signature="m1")[MAP_KEY].tolist(),
                [10.0, 20.0],
            )

        with self.subTest("nonexistent_trial"):
            self.assertTrue(state.get_data(trial_index=99, metric_signature="m1").empty)

        with self.subTest("nonexistent_metric"):
            self.assertTrue(
                state.get_data(trial_index=0, metric_signature="m_missing").empty
            )


class MapDataReplayMetricTest(TestCase):
    def test_map_replay_uniform(self) -> None:
        """Test metric data fetching with uniform steps."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        historical_data: Data = historical_experiment.lookup_data()
        state = MapDataReplayState(
            map_data=historical_data,
            metric_signatures=["branin_map"],
            step_size=1.0,
        )
        replay_metric = MapDataReplayMetric(
            name="test_metric",
            replay_state=state,
            metric_signature="branin_map",
            lower_is_better=True,
        )

        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=replay_metric, minimize=True)
            ),
            tracking_metrics=[replay_metric],
            runner=SyntheticRunner(),
        )
        for i in range(2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": float(i), "x2": 0.0}))
            trial.run()

        state.advance_trial(trial_index=0)
        state.advance_trial(trial_index=1)

        data = experiment.fetch_data()
        metric_name = [replay_metric.name] * 4
        expected_df = Data(
            df=DataFrame(
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
        ).full_df
        assert_frame_equal(data.full_df, expected_df)

    def test_map_replay_non_uniform(self) -> None:
        """Test metric data fetching with non-uniform steps and progressive
        cursor advancement."""
        historical_experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=2, num_complete=2
        )
        full_df = historical_experiment.lookup_data().full_df
        full_df[MAP_KEY] = pd.Series([0.25, 0.0, 0.95, 0.25, 0.0, 1.0])
        historical_data = Data(df=full_df)
        state = MapDataReplayState(
            map_data=historical_data,
            metric_signatures=["branin_map"],
            step_size=0.5,
        )
        replay_metric = MapDataReplayMetric(
            name="test_metric",
            replay_state=state,
            metric_signature="branin_map",
            lower_is_better=True,
        )

        experiment = Experiment(
            name="dummy_experiment",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=replay_metric, minimize=True)
            ),
            tracking_metrics=[replay_metric],
            runner=SyntheticRunner(),
        )
        for i in range(2):
            trial = experiment.new_trial()
            trial.add_arm(Arm(parameters={"x1": float(i), "x2": 0.0}))
            trial.run()

        metric_name = [replay_metric.name] * 4
        full_expected_df = Data(
            df=DataFrame(
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
        ).full_df

        with self.subTest("cursor_0_only_first_points"):
            data = experiment.fetch_data()
            assert_frame_equal(
                data.full_df, full_expected_df.iloc[[0, 2]].reset_index(drop=True)
            )

        with self.subTest("cursor_0_5_intermediate"):
            state.advance_trial(trial_index=0)
            state.advance_trial(trial_index=1)
            data = experiment.fetch_data()
            assert_frame_equal(
                data.full_df, full_expected_df.iloc[[0, 2]].reset_index(drop=True)
            )

        with self.subTest("cursor_1_0_all_data"):
            state.advance_trial(trial_index=0)
            state.advance_trial(trial_index=1)
            data = experiment.fetch_data()
            assert_frame_equal(data.full_df, full_expected_df)

    def test_fetch_trial_data_multi_metric(self) -> None:
        """Test that fetch_trial_data returns data filtered by metric
        signature when multiple metrics share the same state."""
        map_data = _make_map_data(
            {
                0: {
                    "m1": [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0)],
                    "m2": [(0.0, 10.0, 0.0), (1.0, 20.0, 0.0)],
                },
            }
        )
        state = MapDataReplayState(
            map_data=map_data, metric_signatures=["m1", "m2"], step_size=1.0
        )
        metric_m1 = MapDataReplayMetric(
            name="replay_m1", replay_state=state, metric_signature="m1"
        )
        metric_m2 = MapDataReplayMetric(
            name="replay_m2", replay_state=state, metric_signature="m2"
        )

        experiment = Experiment(
            name="dummy",
            search_space=get_branin_search_space(),
            runner=SyntheticRunner(),
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0}))
        trial.run()
        state.advance_trial(trial_index=0)

        with self.subTest("m1_filtered"):
            result_m1 = metric_m1.fetch_trial_data(trial=trial)
            self.assertTrue(result_m1.is_ok())
            df_m1 = none_throws(result_m1.ok).full_df
            self.assertEqual(len(df_m1), 2)
            self.assertTrue((df_m1["metric_name"] == "replay_m1").all())
            self.assertListEqual(df_m1["mean"].tolist(), [1.0, 2.0])

        with self.subTest("m2_filtered"):
            result_m2 = metric_m2.fetch_trial_data(trial=trial)
            self.assertTrue(result_m2.is_ok())
            df_m2 = none_throws(result_m2.ok).full_df
            self.assertEqual(len(df_m2), 2)
            self.assertTrue((df_m2["metric_name"] == "replay_m2").all())
            self.assertListEqual(df_m2["mean"].tolist(), [10.0, 20.0])
