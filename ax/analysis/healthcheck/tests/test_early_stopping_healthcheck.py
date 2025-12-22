# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest.mock import patch

import pandas as pd
from ax.analysis.healthcheck.early_stopping_healthcheck import EarlyStoppingAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_metric,
    get_experiment,
    get_experiment_with_map_data,
)


class TestEarlyStoppingAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.early_stopping_strategy = PercentileEarlyStoppingStrategy()
        self.experiment = get_experiment_with_map_data()
        self.healthcheck = EarlyStoppingAnalysis(
            early_stopping_strategy=self.early_stopping_strategy
        )

    def _get_df_dict(self, card: object) -> dict[str, str]:
        """Extract Property -> Value dict from card dataframe."""
        # pyre-ignore[16]: card has df attribute
        return {row["Property"]: row["Value"] for _, row in card.df.iterrows()}

    def _create_map_data(
        self,
        trial_steps: list[tuple[int, int]],
        metric_name: str = "ax_test_metric",
    ) -> MapData:
        """Create MapData with specified trial progressions.

        Args:
            trial_steps: List of (trial_index, step) tuples.
            metric_name: Name of the metric
        """
        rows = []
        for trial_index, step in trial_steps:
            rows.append(
                {
                    "trial_index": trial_index,
                    "step": step,
                    "arm_name": f"{trial_index}_0",
                    "metric_name": metric_name,
                    "mean": 1.0 + trial_index * 0.1,
                    "sem": 0.1,
                    "metric_signature": metric_name,
                }
            )
        return MapData(df=pd.DataFrame(rows))

    def _mark_trial_completed(
        self, experiment: Experiment | None = None, trial_index: int = 0
    ) -> None:
        """Mark a trial as completed.

        Args:
            experiment: The experiment to mark trial completed on.
                Defaults to self.experiment.
            trial_index: Index of the trial to mark completed.
        """
        exp = experiment if experiment is not None else self.experiment
        exp.trials[trial_index].mark_running(no_runner_required=True)
        exp.trials[trial_index].mark_completed()

    def _mark_trial_early_stopped(
        self, experiment: Experiment | None = None, trial_index: int = 0
    ) -> None:
        """Mark a trial as early stopped.

        Args:
            experiment: The experiment to mark trial early stopped on.
                Defaults to self.experiment.
            trial_index: Index of the trial to mark early stopped.
        """
        exp = experiment if experiment is not None else self.experiment
        exp.trials[trial_index].mark_running(no_runner_required=True)
        exp.trials[trial_index].mark_early_stopped()

    def _fresh_experiment(self) -> Experiment:
        """Create a fresh experiment with map data (for tests that modify state)."""
        return get_experiment_with_map_data()

    def test_dataframe_output(self) -> None:
        """Test the dataframe output contains expected properties and values."""
        self._mark_trial_completed()

        with self.subTest("savings_unavailable_without_stopped_trials"):
            card = self.healthcheck.compute(experiment=self.experiment)
            self.assertTrue(card.is_passing())
            self.assertIn("Capacity savings are not yet available", card.subtitle)

        # Add early stopped trial for remaining subtests
        self.experiment.new_trial()
        self.experiment.attach_data(data=self._create_map_data([(1, 1)]))
        self._mark_trial_early_stopped(trial_index=1)
        card = self.healthcheck.compute(experiment=self.experiment)

        with self.subTest("savings_shown_with_stopped_trials"):
            self.assertTrue(card.is_passing())
            df_dict = self._get_df_dict(card)
            self.assertEqual(df_dict["Early Stopped Trials"], "1")

        with self.subTest("contains_all_expected_properties"):
            expected_properties = [
                "Early Stopped Trials",
                "Completed Trials",
                "Failed Trials",
                "Running Trials",
                "Total Trials",
                "Target Metric",
                "Estimated Savings",
            ]
            properties = card.df["Property"].tolist()
            for prop in expected_properties:
                self.assertIn(prop, properties)

    def test_early_stopping_not_enabled(self) -> None:
        """Test behavior when early stopping is not enabled."""
        healthcheck = EarlyStoppingAnalysis(early_stopping_strategy=None)

        with self.subTest("no_savings_detected"):
            card = healthcheck.compute(experiment=self.experiment)
            self.assertIn("Early stopping is not enabled", card.subtitle)

        with self.subTest("potential_savings_detected"):
            mock_savings = {"ax_test_metric": 25.0}
            with patch.object(
                healthcheck,
                "_estimate_hypothetical_savings_with_replay",
                return_value=mock_savings,
            ):
                card = healthcheck.compute(experiment=self.experiment)
                self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
                self.assertIn("25%", card.subtitle)

    def test_validate_applicable_state(self) -> None:
        """Test that validate_applicable_state returns appropriate error messages."""
        healthcheck = EarlyStoppingAnalysis()

        with self.subTest("no_experiment"):
            error = healthcheck.validate_applicable_state(
                experiment=None, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(error)
            self.assertIn("experiment", error.lower())

        with self.subTest("valid_state"):
            error = healthcheck.validate_applicable_state(
                experiment=self.experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNone(error)

        with self.subTest("no_trials"):
            self.experiment._trials = {}
            error = healthcheck.validate_applicable_state(
                experiment=self.experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(error)
            self.assertIn("no trials", error.lower())

        with self.subTest("no_data"):
            experiment = self._fresh_experiment()
            experiment._data_by_trial = {}
            error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=None, adapter=None
            )
            self.assertIsNotNone(error)
            self.assertIn("no data", error.lower())

    def test_fail_no_metrics_found(self) -> None:
        """Test failure when early stopping is enabled but no metrics are found."""
        no_metric_strategy = PercentileEarlyStoppingStrategy(metric_signatures=[])
        healthcheck = EarlyStoppingAnalysis(early_stopping_strategy=no_metric_strategy)

        card = healthcheck.compute(experiment=self.experiment)

        self.assertIn("no metrics were found", card.subtitle)

    def test_fail_metrics_not_map_metrics(self) -> None:
        """Test failure when early stopping metrics are not MapMetrics."""
        # Use get_experiment() which has regular Metrics (not MapMetrics)
        experiment = get_experiment()
        experiment.new_trial()
        self._mark_trial_completed(experiment=experiment, trial_index=0)

        card = self.healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("are MapMetrics with map data", card.subtitle)

    def test_uses_official_savings_calculation(self) -> None:
        """Test that the healthcheck uses the official savings calculation."""
        # Setup: trial 0 completed at step 100, trial 1 completed at step 100,
        # trial 2 early stopped at step 50
        for _ in range(1, 3):
            self.experiment.new_trial()
        self.experiment.attach_data(
            data=self._create_map_data([(0, 100), (1, 100), (2, 50)])
        )
        self._mark_trial_completed(trial_index=0)
        self._mark_trial_completed(trial_index=1)
        self._mark_trial_early_stopped(trial_index=2)

        card = self.healthcheck.compute(experiment=self.experiment)

        # Expected: resources_saved = 50, resources_used = 250
        # savings = 50 / 300 ≈ 0.1667 = 17%
        df_dict = self._get_df_dict(card)
        self.assertEqual(df_dict["Estimated Savings"], "17%")

    def test_auto_early_stopping_config(self) -> None:
        """Test behavior of auto_early_stopping_config parameter."""
        with self.subTest("disabled"):
            healthcheck = EarlyStoppingAnalysis(auto_early_stopping_config="disabled")
            card = healthcheck.compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertIn("auto_early_stopping_config='disabled'", card.subtitle)

        with self.subTest("standard"):
            self._mark_trial_completed()
            healthcheck = EarlyStoppingAnalysis(auto_early_stopping_config="standard")
            card = healthcheck.compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertIn("0 trials were early stopped", card.subtitle)

        with self.subTest("strategy_override"):
            custom_strategy = PercentileEarlyStoppingStrategy(percentile_threshold=30)
            healthcheck = EarlyStoppingAnalysis(
                early_stopping_strategy=custom_strategy,
                auto_early_stopping_config="standard",
            )
            card = healthcheck.compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertEqual(healthcheck.early_stopping_strategy, custom_strategy)

    def test_multiple_metrics_note_in_subtitle(self) -> None:
        """Test that a note is added when multiple metrics are used for ESS."""
        self._mark_trial_completed()

        with patch(
            "ax.analysis.healthcheck.early_stopping_healthcheck"
            ".get_early_stopping_metrics",
            return_value=["ax_test_metric", "other_metric"],
        ):
            card = self.healthcheck.compute(experiment=self.experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertIn("2 metrics are", card.subtitle)

    def test_get_problem_type_via_disabled_config(self) -> None:
        """Test _get_problem_type is correctly reported when ESS is disabled."""
        healthcheck = EarlyStoppingAnalysis(auto_early_stopping_config="disabled")

        with self.subTest("single_objective_unconstrained"):
            card = healthcheck.compute(experiment=self.experiment)
            df_dict = self._get_df_dict(card)
            self.assertEqual(df_dict["Problem Type"], "Single-objective unconstrained")

        # Remaining subtests need modified optimization configs,
        # so use fresh experiments
        with self.subTest("no_optimization_config"):
            experiment = self._fresh_experiment()
            experiment._optimization_config = None
            card = healthcheck.compute(experiment=experiment)
            df_dict = self._get_df_dict(card)
            self.assertEqual(df_dict["Problem Type"], "No optimization config")

        with self.subTest("multi_objective"):
            experiment = self._fresh_experiment()
            metric1 = get_branin_metric(name="m1")
            metric2 = get_branin_metric(name="m2")
            experiment._optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[Objective(metric=metric1), Objective(metric=metric2)]
                )
            )
            card = healthcheck.compute(experiment=experiment)
            df_dict = self._get_df_dict(card)
            self.assertEqual(df_dict["Problem Type"], "Multi-objective")

        with self.subTest("constrained"):
            experiment = self._fresh_experiment()
            metric = get_branin_metric(name="m1")
            constraint_metric = get_branin_metric(name="constraint_metric")
            experiment._optimization_config = OptimizationConfig(
                objective=Objective(metric=metric),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=constraint_metric, op=ComparisonOp.LEQ, bound=10.0
                    )
                ],
            )
            card = healthcheck.compute(experiment=experiment)
            df_dict = self._get_df_dict(card)
            self.assertEqual(df_dict["Problem Type"], "Constrained (1 constraints)")

    def test_hypothetical_savings_nudge(self) -> None:
        """Test hypothetical savings reporting via the nudge path."""
        healthcheck = EarlyStoppingAnalysis(early_stopping_strategy=None)

        with self.subTest("basic_nudge"):
            with patch(
                "ax.analysis.healthcheck.early_stopping_healthcheck.replay_experiment",
                return_value=object(),
            ), patch(
                "ax.analysis.healthcheck.early_stopping_healthcheck"
                ".estimate_early_stopping_savings",
                return_value=0.25,
            ):
                card = healthcheck.compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("25%", card.subtitle)

        with self.subTest("with_additional_info"):
            nudge_info = (
                "See [this tutorial](https://www.example.com/?id=123) "
                "for instructions on how to turn on early stopping."
            )
            healthcheck_with_info = EarlyStoppingAnalysis(
                early_stopping_strategy=None, nudge_additional_info=nudge_info
            )

            mock_savings = {"ax_test_metric": 25.0}
            with patch.object(
                healthcheck_with_info,
                "_estimate_hypothetical_savings_with_replay",
                return_value=mock_savings,
            ):
                card = healthcheck_with_info.compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn(nudge_info, card.subtitle)
