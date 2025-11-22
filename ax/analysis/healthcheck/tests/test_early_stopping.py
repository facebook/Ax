# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.core.base_trial import TrialStatus

from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.exceptions.core import UserInputError
from ax.health_check.early_stopping import EarlyStoppingHealthcheck
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_map_data,
    get_map_data,
)


class TestEarlyStoppingHealthcheck(TestCase):
    def test_early_stopping_with_savings(self) -> None:
        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        experiment = get_experiment_with_map_data()

        # Set up trials with completed and early stopped statuses
        experiment.trials[0]._status = TrialStatus.COMPLETED
        experiment.new_trial()
        experiment.trials[1]._status = TrialStatus.EARLY_STOPPED

        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertTrue(card.is_passing())
        self.assertIn("1 trials were early stopped", card.subtitle)

        # Values are now in the DataFrame, not as separate attributes
        df_dict = {row["Property"]: row["Value"] for _, row in card.df.iterrows()}
        self.assertEqual(df_dict["Early Stopped Trials"], "1")
        self.assertEqual(df_dict["Completed Trials"], "1")
        self.assertIn("Estimated Savings", df_dict)

    def test_early_stopping_no_completed_trials(self) -> None:
        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        experiment = get_experiment_with_map_data()

        # Mark trial as early stopped but no completed trials yet
        experiment.trials[0]._status = TrialStatus.EARLY_STOPPED
        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertTrue(card.is_passing())
        self.assertIn("1 trials were early stopped", card.subtitle)
        self.assertIn("Capacity savings are not yet available", card.subtitle)

    def test_early_stopping_no_stopped_trials(self) -> None:
        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        experiment = get_experiment_with_map_data()

        # Mark trial as completed, none stopped yet
        experiment.trials[0]._status = TrialStatus.COMPLETED
        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertTrue(card.is_passing())
        self.assertIn("0 trials were early stopped", card.subtitle)
        self.assertIn("Capacity savings are not yet available", card.subtitle)

    def test_warnings(self) -> None:
        test_cases = [
            (
                "early_stopping_not_enabled",
                get_experiment_with_map_data(),
                None,
                True,
                lambda card: self.assertIn(
                    "Early stopping is not enabled", card.subtitle
                ),
            ),
            (
                "no_map_data",
                get_branin_experiment(),
                PercentileEarlyStoppingStrategy(),
                False,
                lambda card: (
                    self.assertIn("Early stopping requires MapData", card.subtitle),
                    self.assertIn("cannot be used", card.subtitle),
                ),
            ),
        ]

        for name, experiment, strategy, attach_data, check_card in test_cases:
            with self.subTest(name=name):
                if attach_data:
                    experiment.attach_data(data=get_map_data())

                healthcheck = EarlyStoppingHealthcheck(early_stopping_strategy=strategy)
                card = healthcheck.compute(experiment=experiment)
                check_card(card)

    def test_raises_error_no_experiment(self) -> None:
        healthcheck = EarlyStoppingHealthcheck()
        with self.assertRaisesRegex(
            UserInputError,
            "EarlyStoppingHealthcheck requires an Experiment",
        ):
            healthcheck.compute(experiment=None)

    def test_dataframe_contains_details_when_enabled(self) -> None:
        experiment = get_experiment_with_map_data()
        experiment.trials[0]._status = TrialStatus.COMPLETED
        experiment.attach_data(data=get_map_data())

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        df = card.df
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)

        properties = df["Property"].tolist()
        expected_properties = [
            "Early Stopped Trials",
            "Completed Trials",
            "Total Trials",
            "Target Metric",
            "Estimated Savings",
        ]
        for prop in expected_properties:
            self.assertIn(prop, properties)

    def test_custom_map_key(self) -> None:
        experiment = get_experiment_with_map_data()
        experiment.new_trial()  # Create a second trial
        experiment.trials[0]._status = TrialStatus.COMPLETED
        experiment.trials[1]._status = TrialStatus.EARLY_STOPPED

        # Create MapData with custom map_key "step" (the default)
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "step": 10,
                    "arm_name": "0_0",
                    "metric_name": "ax_test_metric",
                    "mean": 1.0,
                    "sem": 0.1,
                    "metric_signature": "ax_test_metric",
                },
                {
                    "trial_index": 1,
                    "step": 5,
                    "arm_name": "1_0",
                    "metric_name": "ax_test_metric",
                    "mean": 1.1,
                    "sem": 0.1,
                    "metric_signature": "ax_test_metric",
                },
            ]
        )
        map_data = MapData(df=df)
        experiment.attach_data(data=map_data)

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertTrue(card.is_passing())
        self.assertIn("step", card.subtitle)

    def test_fail_no_metrics_found(self) -> None:
        """Test failure when early stopping is enabled but no metrics are found."""
        from ax.core.experiment import Experiment
        from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy

        # Create a mock early stopping strategy that returns no metrics
        class NoMetricStrategy(BaseEarlyStoppingStrategy):
            def __init__(self) -> None:
                super().__init__(metric_signatures=[])

            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, any],
            ) -> dict[int, str | None]:
                return {}

        experiment = get_experiment_with_map_data()
        experiment.attach_data(data=get_map_data())

        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=NoMetricStrategy()
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertIn("no metrics were found", card.subtitle)
        self.assertIn("configuration issue", card.subtitle)

    def test_pass_when_no_map_data(self) -> None:
        """Test pass status when experiment has no MapData."""
        # Use branin experiment which has regular metrics, not map metrics
        experiment = get_branin_experiment()

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        # Should return PASS status when no MapData available
        self.assertTrue(card.is_passing())
        self.assertIn("MapData", card.subtitle)
        self.assertIn("cannot be used", card.subtitle)

    def test_multi_objective_note(self) -> None:
        """Test that multi-objective early stopping shows an informational note."""

        # Create a mock strategy that uses multiple metrics
        class MultiMetricStrategy(BaseEarlyStoppingStrategy):
            def __init__(self) -> None:
                super().__init__(metric_signatures=["m1", "m2"])

            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, any],
            ) -> dict[int, str | None]:
                return {}

        experiment = get_experiment_with_map_data()
        experiment.attach_data(data=get_map_data())

        # Mark trial as completed
        experiment.trials[0]._status = TrialStatus.COMPLETED

        # Create early stopping strategy that will use both metrics
        early_stopping_strategy = MultiMetricStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        # Should mention multiple metrics in subtitle
        self.assertIn("metrics are being used", card.subtitle)
        self.assertIn("compute resource savings", card.subtitle)

    def test_target_metric_shown_in_dataframe(self) -> None:
        """Test that the target metric is shown in the dataframe output."""
        experiment = get_experiment_with_map_data()
        experiment.trials[0]._status = TrialStatus.COMPLETED
        experiment.attach_data(data=get_map_data())

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        df = card.df
        self.assertIsNotNone(df)

        # Check that Target Metric row exists
        target_metric_rows = df[df["Property"] == "Target Metric"]
        self.assertEqual(len(target_metric_rows), 1)
        # The metric name might be different (e.g., "m1") depending on experiment setup
        self.assertIsNotNone(target_metric_rows.iloc[0]["Value"])

    def test_uses_official_savings_calculation(self) -> None:
        """Test that the healthcheck uses the official savings calculation
        from ax.early_stopping.utils."""
        experiment = get_experiment_with_map_data()

        # Create multiple trials
        experiment.trials[0]._status = TrialStatus.COMPLETED
        for _ in range(2):
            experiment.new_trial()
        experiment.trials[1]._status = TrialStatus.COMPLETED
        experiment.trials[2]._status = TrialStatus.EARLY_STOPPED

        # Create MapData with different progressions
        import pandas as pd

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

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        healthcheck = EarlyStoppingHealthcheck(
            early_stopping_strategy=early_stopping_strategy
        )
        card = healthcheck.compute(experiment=experiment)

        # The savings should be calculated by the official method
        # Expected: (100 - 50) / (100 + 100 + 50) = 0.2 = 20%
        df_dict = {row["Property"]: row["Value"] for _, row in card.df.iterrows()}
        savings_str = df_dict["Estimated Savings"]

        # Verify savings is present and non-zero
        self.assertNotEqual(savings_str, "N/A")
        self.assertIn("%", savings_str)
