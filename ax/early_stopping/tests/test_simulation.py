#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
from ax.early_stopping.simulation import (
    _check_patience_window,
    _get_interval_progressions,
    best_trial_vulnerable,
    EarlyStoppingSimulationResult,
)
from ax.utils.common.testutils import TestCase


class TestEarlyStoppingSimulationResult(TestCase):
    def test_dataclass_fields(self) -> None:
        with self.subTest("basic creation"):
            result = EarlyStoppingSimulationResult(
                best_stopped=False,
                best_trial_index=0,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 0)
            self.assertIsNone(result.best_stop_progression)

        with self.subTest("with stop progression"):
            result = EarlyStoppingSimulationResult(
                best_stopped=True,
                best_trial_index=1,
                best_stop_progression=5.0,
            )
            self.assertTrue(result.best_stopped)
            self.assertEqual(result.best_trial_index, 1)
            self.assertEqual(result.best_stop_progression, 5.0)


class TestGetIntervalProgressions(TestCase):
    def test_interval_filtering(self) -> None:
        with self.subTest("basic interval filtering"):
            result = _get_interval_progressions(
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                min_progression=0.0,
                interval=3.0,
            )
            np.testing.assert_array_equal(result, [0.0, 3.0, 6.0, 9.0])

        with self.subTest("with min progression offset"):
            result = _get_interval_progressions(
                np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                min_progression=2.0,
                interval=2.0,
            )
            np.testing.assert_array_equal(result, [2.0, 4.0, 6.0, 8.0])

        with self.subTest("single interval"):
            result = _get_interval_progressions(
                np.array([0.0, 1.0, 2.0]),
                min_progression=0.0,
                interval=10.0,
            )
            np.testing.assert_array_equal(result, [0.0])

        with self.subTest("empty progressions"):
            result = _get_interval_progressions(
                np.array([]),
                min_progression=0.0,
                interval=5.0,
            )
            np.testing.assert_array_equal(result, [])


class TestCheckPatienceWindow(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.wide_df = pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0, 100.0],
                1: [50.0, 50.0, 50.0, 50.0],
                2: [30.0, 30.0, 30.0, 30.0],
            },
            index=[0.0, 1.0, 2.0, 3.0],
        )

    def test_patience_window_behavior(self) -> None:
        with self.subTest("consistent underperformer is stopped"):
            result = _check_patience_window(
                wide_df=self.wide_df,
                trial_indices={0, 1, 2},
                progression=2.0,
                patience=2.0,
                min_progression=0.0,
                quantile=0.75,
                minimize=True,
                n_best_trials_to_complete=None,
            )
            self.assertTrue(result[0])
            self.assertFalse(result[1])
            self.assertFalse(result[2])

        with self.subTest("inconsistent underperformer not stopped"):
            wide_df = pd.DataFrame(
                {
                    0: [30.0, 100.0, 100.0],
                    1: [50.0, 50.0, 50.0],
                    2: [100.0, 30.0, 30.0],
                },
                index=[0.0, 1.0, 2.0],
            )
            result = _check_patience_window(
                wide_df=wide_df,
                trial_indices={0, 1, 2},
                progression=2.0,
                patience=2.0,
                min_progression=0.0,
                quantile=0.5,
                minimize=True,
                n_best_trials_to_complete=None,
            )
            self.assertFalse(result[0])

        with self.subTest("n_best_trials protection"):
            result = _check_patience_window(
                wide_df=self.wide_df,
                trial_indices={0, 1, 2},
                progression=3.0,
                patience=1.0,
                min_progression=0.0,
                quantile=0.75,
                minimize=True,
                n_best_trials_to_complete=2,
            )
            self.assertFalse(result[1])
            self.assertFalse(result[2])

        with self.subTest("single progression in window"):
            result = _check_patience_window(
                wide_df=self.wide_df,
                trial_indices={0, 1, 2},
                progression=0.0,
                patience=1.0,
                min_progression=0.0,
                quantile=0.75,
                minimize=True,
                n_best_trials_to_complete=None,
            )
            self.assertTrue(result[0])
            self.assertFalse(result[1])
            self.assertFalse(result[2])

        with self.subTest("empty trial indices"):
            result = _check_patience_window(
                wide_df=self.wide_df,
                trial_indices=set(),
                progression=2.0,
                patience=1.0,
                min_progression=0.0,
                quantile=0.75,
                minimize=True,
                n_best_trials_to_complete=None,
            )
            self.assertEqual(len(result), 0)


class TestBestTrialVulnerable(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.wide_df = pd.DataFrame(
            {
                0: [100.0, 90.0, 80.0, 70.0, 60.0],
                1: [50.0, 45.0, 40.0, 35.0, 30.0],
                2: [30.0, 28.0, 25.0, 22.0, 20.0],
                3: [95.0, 85.0, 75.0, 65.0, 55.0],
                4: [60.0, 55.0, 50.0, 45.0, 40.0],
            },
            index=[0.0, 1.0, 2.0, 3.0, 4.0],
        )

    def test_best_trial_stopping(self) -> None:
        with self.subTest("best trial not stopped"):
            result = best_trial_vulnerable(
                wide_df=self.wide_df,
                minimize=True,
                completed_trials=[0, 1, 2, 3, 4],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=5.0,
                min_curves=2,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 2)

        with self.subTest("best trial stopped"):
            wide_df = pd.DataFrame(
                {
                    0: [100.0, 30.0, 20.0, 10.0],
                    1: [50.0, 45.0, 40.0, 35.0],
                    2: [60.0, 55.0, 50.0, 45.0],
                },
                index=[0.0, 1.0, 2.0, 3.0],
            )
            result = best_trial_vulnerable(
                wide_df=wide_df,
                minimize=True,
                completed_trials=[0, 1, 2],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=4.0,
                min_curves=1,
            )
            self.assertTrue(result.best_stopped)
            self.assertEqual(result.best_trial_index, 0)

    def test_patience_behavior(self) -> None:
        with self.subTest("patience prevents stopping"):
            wide_df = pd.DataFrame(
                {
                    0: [10.0, 100.0, 100.0, 100.0],
                    1: [50.0, 50.0, 50.0, 30.0],
                    2: [60.0, 60.0, 60.0, 60.0],
                },
                index=[0.0, 1.0, 2.0, 3.0],
            )
            result = best_trial_vulnerable(
                wide_df=wide_df,
                minimize=True,
                completed_trials=[0, 1, 2],
                percentile_threshold=50.0,
                min_progression=1.0,
                max_progression=4.0,
                min_curves=1,
                patience=2.0,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 1)

        with self.subTest("patience allows stopping"):
            wide_df = pd.DataFrame(
                {
                    0: [100.0, 100.0, 100.0, 10.0],
                    1: [50.0, 50.0, 50.0, 50.0],
                    2: [60.0, 60.0, 60.0, 60.0],
                },
                index=[0.0, 1.0, 2.0, 3.0],
            )
            result = best_trial_vulnerable(
                wide_df=wide_df,
                minimize=True,
                completed_trials=[0, 1, 2],
                percentile_threshold=50.0,
                min_progression=1.0,
                max_progression=4.0,
                min_curves=1,
                patience=2.0,
            )
            self.assertTrue(result.best_stopped)
            self.assertEqual(result.best_trial_index, 0)

    def test_configuration_options(self) -> None:
        with self.subTest("interval filtering"):
            result = best_trial_vulnerable(
                wide_df=self.wide_df,
                minimize=True,
                completed_trials=[0, 1, 2, 3, 4],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=5.0,
                min_curves=2,
                interval=2.0,
            )
            self.assertFalse(result.best_stopped)

        with self.subTest("n_best_trials protection"):
            wide_df = pd.DataFrame(
                {
                    0: [100.0, 90.0, 80.0, 70.0],
                    1: [50.0, 45.0, 40.0, 10.0],
                    2: [30.0, 28.0, 25.0, 22.0],
                },
                index=[0.0, 1.0, 2.0, 3.0],
            )
            result = best_trial_vulnerable(
                wide_df=wide_df,
                minimize=True,
                completed_trials=[0, 1, 2],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=4.0,
                min_curves=1,
                n_best_trials_to_complete=2,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 1)

        with self.subTest("maximization direction"):
            result = best_trial_vulnerable(
                wide_df=self.wide_df,
                minimize=False,
                completed_trials=[0, 1, 2, 3, 4],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=5.0,
                min_curves=2,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 0)

        with self.subTest("empty completed trials"):
            result = best_trial_vulnerable(
                wide_df=self.wide_df,
                minimize=True,
                completed_trials=[],
                percentile_threshold=50.0,
                min_progression=0.0,
                max_progression=5.0,
                min_curves=5,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 2)

        with self.subTest("low percentile threshold"):
            result = best_trial_vulnerable(
                wide_df=self.wide_df,
                minimize=True,
                completed_trials=[0, 1, 2, 3, 4],
                percentile_threshold=25.0,
                min_progression=0.0,
                max_progression=5.0,
                min_curves=0,
            )
            self.assertFalse(result.best_stopped)
            self.assertEqual(result.best_trial_index, 2)
