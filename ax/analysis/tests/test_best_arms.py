# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.best_arms import BestArms
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.base_trial import TrialStatus
from ax.exceptions.core import DataRequiredError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node


class TestBestArms(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = Client()
        self.client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        self.client.configure_optimization(objective="foo")
        self.experiment = self.client._experiment

    def test_compute_soo(self) -> None:
        """Test BestArms for single-objective optimization."""
        client = self.client
        # Setup: Create multiple trials with different objective values
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 3.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=2, raw_data={"foo": 2.0})

        # Execute: Compute BestArms analysis
        analysis = BestArms()

        card = analysis.compute(
            experiment=self.experiment,
            generation_strategy=client._generation_strategy,
        )

        # Assert: Verify only the best trial is returned
        self.assertEqual(card.name, "BestTrials")
        self.assertEqual(card.title, "Best Trial for Experiment")
        self.assertIn("the best objective value", card.subtitle.lower())
        self.assertIsNotNone(card.blob)

        # Assert: Should only have 1 row for the best trial
        self.assertEqual(len(card.df), 1)
        # The best trial should have foo=3.0 (maximizing objective, default)
        self.assertEqual(card.df["foo"].iloc[0], 3.0)

        # Assert: Verify dataframe structure
        self.assertEqual(
            {*card.df.columns},
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "foo",
                "x1",
                "x2",
            },
        )

    def test_compute_moo(self) -> None:
        """Test BestArms for multi-objective optimization."""
        client = self.client
        # Reconfigure as multi-objective
        client.configure_optimization(
            objective="foo, bar",
        )

        # Setup: Create trials that form a Pareto frontier
        client.get_next_trials(max_trials=3)
        client.complete_trial(
            trial_index=0, raw_data={"foo": 1.0, "bar": 3.0}
        )  # Pareto optimal
        client.complete_trial(
            trial_index=1, raw_data={"foo": 0.5, "bar": 2.5}
        )  # Dominated by trial 0
        client.complete_trial(
            trial_index=2, raw_data={"foo": 3.0, "bar": 1.0}
        )  # Pareto optimal

        # Execute: Compute BestArms analysis
        analysis = BestArms()

        card = analysis.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        )

        # Assert: Verify Pareto frontier trials are returned
        self.assertEqual(card.name, "BestTrials")
        self.assertEqual(card.title, "Pareto Frontier Trials for Experiment")
        self.assertIn("pareto", card.subtitle.lower())

        # Assert: Should have exactly 2 Pareto optimal trials (0 and 2)
        self.assertEqual(len(card.df), 2)
        pareto_indices = set(card.df["trial_index"].values)
        self.assertEqual(pareto_indices, {0, 2})

    def test_no_eligible_trials_returns_validation_error(self) -> None:
        """Test that BestArms returns validation error when no eligible trials."""
        client = self.client
        # Setup: Create and complete a trial, then filter by a different status
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        # Execute: Attempt to validate BestArms with FAILED status filter
        # (no trials are FAILED, so this should return an error)
        analysis = BestArms(trial_statuses=[TrialStatus.FAILED])

        # Assert: Should return error string when no trials match the status filter
        error = analysis.validate_applicable_state(
            experiment=self.experiment,
            generation_strategy=client._generation_strategy,
        )
        self.assertIsNotNone(error)
        self.assertIn("No trials found with status in", error)

    def test_generation_strategy_requirements(self) -> None:
        """Test GenerationStrategy requirements based on use_model_predictions."""
        client = self.client
        # Setup: Create and complete a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        with self.subTest(msg="GS not required for raw observations"):
            # Execute & Assert: Should succeed without generation_strategy
            # when using raw observations
            analysis = BestArms(use_model_predictions=False)
            card = analysis.compute(
                experiment=self.experiment, generation_strategy=None
            )

            # Assert: Should return a valid result
            self.assertEqual(card.name, "BestTrials")
            self.assertIsNotNone(card.df)
            self.assertEqual(len(card.df), 1)

        with self.subTest(msg="GS required for model predictions"):
            # Execute & Assert: Should return error from validation
            # when generation_strategy is None with model predictions
            analysis = BestArms(use_model_predictions=True)
            error = analysis.validate_applicable_state(
                experiment=self.experiment, generation_strategy=None
            )
            self.assertIsNotNone(error)
            self.assertIn("requires a `GenerationStrategy`", error)

    def test_trial_status_filter(self) -> None:
        """Test that trial_statuses parameter filters correctly."""
        client = self.client
        # Setup: Create trials with different statuses
        # The best trial (trial 2 with foo=3.0) is NOT completed,
        # so filtering by COMPLETED should return trial 1 (foo=2.0) instead
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 2.0})
        # Mark trial 2 as failed
        self.experiment.trials[2].mark_failed()

        # Execute: Compute BestArms with only COMPLETED status filter
        analysis = BestArms(trial_statuses=[TrialStatus.COMPLETED])
        card = analysis.compute(
            experiment=self.experiment,
            generation_strategy=client._generation_strategy,
        )

        # Assert: Only COMPLETED trials should be in results
        self.assertTrue(all(card.df["trial_status"] == "COMPLETED"))
        # Best completed trial should have foo=2.0, not foo=3.0
        # (trial 2 with foo=3.0 is filtered out because it's FAILED)
        self.assertEqual(card.df["foo"].iloc[0], 2.0)
        self.assertEqual(card.df["trial_index"].iloc[0], 1)

    def test_use_model_predictions_insufficient_data(self) -> None:
        """Test use_model_predictions=True raises error with insufficient data."""
        client = self.client
        # Setup: Create trials with limited data (insufficient for model)
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 2.0})
        client.complete_trial(trial_index=2, raw_data={"foo": 3.0})

        # Execute & Assert: Should raise error when model cannot make predictions
        analysis = BestArms(use_model_predictions=True)
        with self.assertRaisesRegex(
            DataRequiredError, "No best arm.*could be identified"
        ):
            analysis.compute(
                experiment=self.experiment,
                generation_strategy=client._generation_strategy,
            )

    def test_compute_soo_multi_batch(self) -> None:
        """Test SOO with batch trials: card.name is 'BestArm' and output contains
        all arms from the winning batch."""
        exp = get_branin_experiment(
            with_completed_batch=True, num_batch_trial=2, num_arms_per_trial=3
        )

        card = BestArms().compute(experiment=exp)

        # Batch trials produce "BestArm" display name
        self.assertEqual(card.name, "BestArm")
        self.assertEqual(card.title, "Best Trial for Experiment")
        # Output should contain all arms from the winning batch, not just one
        self.assertGreater(len(card.df), 1)
        # All returned arms should be from the same trial
        self.assertEqual(len(card.df["trial_index"].unique()), 1)

    def test_compute_moo_multi_batch(self) -> None:
        """Test MOO Pareto frontier across multiple batch trials."""
        exp = get_branin_experiment_with_multi_objective(
            with_completed_batch=True,
            with_status_quo=True,
            has_objective_thresholds=True,
        )
        gs = get_default_generation_strategy_at_MBM_node(experiment=exp)

        card = BestArms().compute(experiment=exp, generation_strategy=gs)

        self.assertEqual(card.name, "BestArm")
        self.assertEqual(card.title, "Pareto Frontier Trials for Experiment")
        self.assertIn("pareto", card.subtitle.lower())
        # Pareto frontier should return at least one trial
        self.assertGreater(len(card.df), 0)
