# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.core.data import Data
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.stats.no_effects import (
    check_experiment_effects,
    check_experiment_effects_per_metric,
    no_effect_test_welch,
)


def _data(rows: list[dict[str, object]]) -> Data:
    base = {"trial_index": 0, "metric_name": "m1", "metric_signature": "m1"}
    return Data(df=pd.DataFrame([{**base, **row} for row in rows]))


class TestNoEffects(TestCase):
    def test_effects_detected(self) -> None:
        # GIVEN two arms with clearly different means
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.01, "n": 1000},
                {"arm_name": "0_1", "mean": 2.0, "sem": 0.01, "n": 1000},
            ]
        )
        # WHEN we run the test of no effect
        df_tone = check_experiment_effects_per_metric(data=data, objective_names={"m1"})
        # THEN an effect is detected
        self.assertTrue(df_tone["has_effect"].all())

    def test_missing_sample_sizes(self) -> None:
        # GIVEN data without the optional `n` column (as produced by standard
        # trial evaluation, e.g. `Client.complete_trial`)
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.1},
                {"arm_name": "0_1", "mean": 2.0, "sem": 0.1},
            ]
        )
        # WHEN we run the test of no effect
        # THEN it raises a UserInputError instead of a KeyError
        with self.assertRaisesRegex(UserInputError, "per-arm sample sizes"):
            check_experiment_effects_per_metric(data=data, objective_names={"m1"})

    def test_check_experiment_effects_missing_sample_sizes(self) -> None:
        # GIVEN data without the optional `n` column
        data = _data(
            [
                {"arm_name": "status_quo", "mean": 1.0, "sem": 0.1},
                {"arm_name": "0_0", "mean": 2.0, "sem": 0.1},
            ]
        )
        # WHEN we run the overall (across-metric) test of no effect
        # THEN it raises a UserInputError instead of a KeyError
        with self.assertRaisesRegex(UserInputError, "per-arm sample sizes"):
            check_experiment_effects(data=data, objective_names={"m1"})

    def test_null_sample_sizes(self) -> None:
        # GIVEN data where some rows are missing a sample size
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.1, "n": 1000},
                {"arm_name": "0_1", "mean": 2.0, "sem": 0.1, "n": None},
            ]
        )
        # WHEN we run the test of no effect
        # THEN it raises a UserInputError
        with self.assertRaisesRegex(UserInputError, "per-arm sample sizes"):
            check_experiment_effects_per_metric(data=data, objective_names={"m1"})

    def test_zero_sem_with_different_means(self) -> None:
        # GIVEN deterministic data (sem == 0) with clearly different means
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.0, "n": 1000},
                {"arm_name": "0_1", "mean": 2.0, "sem": 0.0, "n": 1000},
            ]
        )
        # WHEN we run the test of no effect
        df_tone = check_experiment_effects_per_metric(data=data, objective_names={"m1"})
        # THEN the exact effect is detected (previously a NaN p-value silently
        # read as "no effect")
        self.assertEqual(df_tone["p_value"].item(), 0.0)
        self.assertTrue(df_tone["has_effect"].item())

    def test_zero_sem_with_equal_means(self) -> None:
        # GIVEN deterministic data (sem == 0) with identical means
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.0, "n": 1000},
                {"arm_name": "0_1", "mean": 1.0, "sem": 0.0, "n": 1000},
            ]
        )
        # WHEN we run the test of no effect
        df_tone = check_experiment_effects_per_metric(data=data, objective_names={"m1"})
        # THEN no effect is detected, with a well-defined p-value
        self.assertEqual(df_tone["p_value"].item(), 1.0)
        self.assertFalse(df_tone["has_effect"].item())

    def test_mixed_zero_and_positive_sems(self) -> None:
        # GIVEN one arm with sem == 0 and another with a positive sem
        # WHEN we run Welch's test directly
        # THEN it raises a UserInputError since the test is undefined
        with self.assertRaisesRegex(UserInputError, "positive sem"):
            no_effect_test_welch(means=[1.0, 1.0], sems=[0.0, 0.1], ns=[1000, 1000])

    def test_single_arm_groups_are_skipped(self) -> None:
        # GIVEN one single-arm trial and one two-arm trial
        data = _data(
            [
                {"arm_name": "0_0", "mean": 1.0, "sem": 0.1, "n": 1000},
                {
                    "arm_name": "1_0",
                    "mean": 1.0,
                    "sem": 0.1,
                    "n": 1000,
                    "trial_index": 1,
                },
                {
                    "arm_name": "1_1",
                    "mean": 2.0,
                    "sem": 0.1,
                    "n": 1000,
                    "trial_index": 1,
                },
            ]
        )
        # WHEN we run the test of no effect
        df_tone = check_experiment_effects_per_metric(data=data, objective_names={"m1"})
        # THEN the single-arm trial is skipped (previously it produced a NaN
        # p-value) and the two-arm trial is tested
        self.assertEqual(df_tone["trial_index"].tolist(), [1])
        self.assertTrue(df_tone["has_effect"].item())

    def test_welch_requires_at_least_two_arms(self) -> None:
        with self.assertRaisesRegex(UserInputError, "at least two arms"):
            no_effect_test_welch(means=[1.0], sems=[0.1], ns=[1000])

    def test_welch_requires_more_than_one_observation(self) -> None:
        with self.assertRaisesRegex(UserInputError, "more than one observation"):
            no_effect_test_welch(means=[1.0, 2.0], sems=[0.1, 0.1], ns=[1, 1000])
