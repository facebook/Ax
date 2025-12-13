# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest.mock import patch

import pandas as pd
from ax.analysis.healthcheck.early_stopping_healthcheck import EarlyStoppingAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.map_data import MapData
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_map_data, get_map_data


class TestEarlyStoppingAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.early_stopping_strategy = PercentileEarlyStoppingStrategy()
        self.experiment = get_experiment_with_map_data()

    def test_dataframe_output(self) -> None:
        experiment = self.experiment
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()
        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingAnalysis(
            early_stopping_strategy=self.early_stopping_strategy
        )

        # Test no savings case before adding early stopped trial
        with self.subTest("no_savings_when_no_stopped_trials"):
            card = healthcheck.compute(experiment=experiment)
            self.assertTrue(card.is_passing())
            self.assertIn("0 trials were early stopped", card.subtitle)
            self.assertIn("Capacity savings are not yet available", card.subtitle)

        # Now add early stopped trial for remaining tests
        experiment.new_trial()
        experiment.trials[1].mark_running(no_runner_required=True)
        experiment.attach_data(data=get_map_data(trial_index=1))
        experiment.trials[1].mark_early_stopped()

        card = healthcheck.compute(experiment=experiment)

        with self.subTest("savings_shown"):
            self.assertTrue(card.is_passing())
            self.assertIn("1 trials were early stopped", card.subtitle)

            df_dict = {row["Property"]: row["Value"] for _, row in card.df.iterrows()}
            self.assertEqual(df_dict["Early Stopped Trials"], "1")
            self.assertEqual(df_dict["Completed Trials"], "1")
            self.assertIn("Estimated Savings", df_dict)

        with self.subTest("contains_details_when_enabled"):
            df = card.df
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 7)

            properties = df["Property"].tolist()
            expected_properties = [
                "Early Stopped Trials",
                "Completed Trials",
                "Failed Trials",
                "Running Trials",
                "Total Trials",
                "Target Metric",
                "Estimated Savings",
            ]
            for prop in expected_properties:
                self.assertIn(prop, properties)

        with self.subTest("target_metric_shown"):
            df = card.df
            self.assertIsNotNone(df)

            target_metric_rows = df[df["Property"] == "Target Metric"]
            self.assertEqual(len(target_metric_rows), 1)
            self.assertIsNotNone(target_metric_rows.iloc[0]["Value"])

    def test_early_stopping_not_enabled(self) -> None:
        """Test behavior when early stopping is not enabled."""
        experiment = get_experiment_with_map_data()
        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingAnalysis(early_stopping_strategy=None)
        card = healthcheck.compute(experiment=experiment)

        with self.subTest("no_savings_detected"):
            # When replay doesn't find significant savings, should pass
            self.assertIn("Early stopping is not enabled", card.subtitle)

        with self.subTest("potential_savings_detected"):
            # Mock the replay to return an experiment with savings

            mock_savings = {"ax_test_metric": 25.0}
            with patch.object(
                healthcheck,
                "_estimate_hypothetical_savings_with_replay",
                return_value=mock_savings,
            ):
                card = healthcheck.compute(experiment=experiment)
                self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
                self.assertIn("compatible with early stopping", card.subtitle)
                self.assertIn("25%", card.subtitle)

    def test_validate_applicable_state(self) -> None:
        """Test that validate_applicable_state returns appropriate error messages."""
        healthcheck = EarlyStoppingAnalysis()

        with self.subTest("no_experiment"):
            validation_error = healthcheck.validate_applicable_state(
                experiment=None, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(validation_error)
            self.assertIn("experiment", validation_error.lower())

        with self.subTest("no_trials"):
            experiment = get_experiment_with_map_data()
            # Remove the trial that was created by get_experiment_with_map_data
            experiment._trials = {}
            validation_error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(validation_error)
            self.assertIn("no trials", validation_error.lower())

        with self.subTest("no_data"):
            experiment = get_experiment_with_map_data()
            # Clear all data from the experiment
            experiment._data_by_trial = {}
            validation_error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(validation_error)
            self.assertIn("no data", validation_error.lower())

        with self.subTest("valid_state"):
            experiment = get_experiment_with_map_data()
            experiment.attach_data(data=get_map_data())
            validation_error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNone(validation_error)

    def test_fail_no_metrics_found(self) -> None:
        """Test failure when early stopping is enabled but no metrics are found."""
        self.experiment.attach_data(data=get_map_data())

        # Use existing strategy with empty metric_signatures to test no metrics case.
        # Note: Cannot use self.early_stopping_strategy since it uses default
        # metric_signatures=None which falls back to objective metrics.
        no_metric_strategy = PercentileEarlyStoppingStrategy(metric_signatures=[])
        healthcheck = EarlyStoppingAnalysis(early_stopping_strategy=no_metric_strategy)
        card = healthcheck.compute(experiment=self.experiment)

        self.assertIn("no metrics were found", card.subtitle)
        self.assertIn("configuration issue", card.subtitle)

    def test_uses_official_savings_calculation(self) -> None:
        """Test that the healthcheck uses the official savings calculation
        from ax.early_stopping.utils."""
        experiment = self.experiment

        # Create multiple trials
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()
        for _ in range(2):
            experiment.new_trial()
        experiment.trials[1].mark_running(no_runner_required=True)
        experiment.trials[1].mark_completed()
        experiment.trials[2].mark_running(no_runner_required=True)

        # Create MapData with different progressions

        df = pd.DataFrame(
            [
                # Trial 0 completed at step 100
                {
                    "trial_index": 0,
                    "step": 100,
                    "arm_name": "0_0",
                    "metric_name": "ax_test_metric",
                    "mean": 1.0,
                    "sem": 0.1,
                    "metric_signature": "ax_test_metric",
                },
                # Trial 1 completed at step 100
                {
                    "trial_index": 1,
                    "step": 100,
                    "arm_name": "1_0",
                    "metric_name": "ax_test_metric",
                    "mean": 1.1,
                    "sem": 0.1,
                    "metric_signature": "ax_test_metric",
                },
                # Trial 2 early stopped at step 50
                {
                    "trial_index": 2,
                    "step": 50,
                    "arm_name": "2_0",
                    "metric_name": "ax_test_metric",
                    "mean": 2.0,
                    "sem": 0.1,
                    "metric_signature": "ax_test_metric",
                },
            ]
        )
        map_data = MapData(df=df)
        experiment.attach_data(data=map_data)

        # Now mark trial 2 as early stopped (must have data first)
        experiment.trials[2].mark_early_stopped()

        healthcheck = EarlyStoppingAnalysis(
            early_stopping_strategy=self.early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        # The savings should be calculated by the official method:
        # resources_used = {0: 100, 1: 100, 2: 50}
        # avg_completed = (100 + 100) / 2 = 100
        # resources_saved for trial 2 = 100 - 50 = 50
        # savings = resources_saved / (resources_saved + resources_used)
        #         = 50 / (50 + 250) = 50 / 300 ≈ 0.1667 = 17%
        df_dict = {row["Property"]: row["Value"] for _, row in card.df.iterrows()}
        savings_str = df_dict["Estimated Savings"]

        self.assertEqual(savings_str, "17%")
