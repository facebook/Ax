# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.best_trials import BestTrials
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.base_trial import TrialStatus
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestBestTrials(TestCase):
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

    def test_compute_soo(self) -> None:
        """Test BestTrials for single-objective optimization."""
        client = self.client

        # Setup: Create multiple trials with different objective values
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 3.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=2, raw_data={"foo": 2.0})

        # Execute: Compute BestTrials analysis
        analysis = BestTrials()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        generation_strategy = client._generation_strategy

        card = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Assert: Verify only the best trial is returned
        self.assertEqual(card.name, "BestTrials")
        self.assertEqual(card.title, "Best Trial for test_experiment")
        self.assertIn("best trial", card.subtitle.lower())
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
        """Test BestTrials for multi-objective optimization."""
        client = Client()
        client.configure_experiment(
            name="test_moo_experiment",
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
        # Configure as multi-objective
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

        # Execute: Compute BestTrials analysis
        analysis = BestTrials()
        experiment = client._experiment
        generation_strategy = client._generation_strategy

        card = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Assert: Verify Pareto frontier trials are returned
        self.assertEqual(card.name, "BestTrials")
        self.assertEqual(card.title, "Pareto Frontier Trials for test_moo_experiment")
        self.assertIn("pareto frontier", card.subtitle.lower())

        # Assert: Should have exactly 2 Pareto optimal trials (0 and 2)
        # Trial 0 (foo=1.0, bar=3.0) and Trial 2 (foo=3.0, bar=1.0) are Pareto optimal
        # Trial 1 (foo=0.5, bar=2.5) is dominated by Trial 0
        # (since 1.0 > 0.5 and 3.0 > 2.5, Trial 0 dominates Trial 1)
        self.assertEqual(len(card.df), 2)
        pareto_indices = set(card.df["trial_index"].values)
        self.assertEqual(pareto_indices, {0, 2})

    def test_no_best_trial_raises_error(self) -> None:
        """Test that BestTrials raises error when no best trial can be identified."""
        client = self.client

        # Setup: Create trials but don't complete them (no data)
        client.get_next_trials(max_trials=1)

        # Execute: Attempt to compute BestTrials
        analysis = BestTrials()
        experiment = client._experiment
        generation_strategy = client._generation_strategy

        # Assert: Should raise error when no best trial found
        with self.assertRaisesRegex(
            UserInputError, "No best trial.*could be identified"
        ):
            analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

    def test_requires_generation_strategy(self) -> None:
        """Test that BestTrials requires a GenerationStrategy."""
        client = self.client

        # Setup: Create and complete a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        # Execute & Assert: Should raise error when generation_strategy is None
        analysis = BestTrials()
        with self.assertRaisesRegex(UserInputError, "requires a `GenerationStrategy`"):
            analysis.compute(experiment=client._experiment, generation_strategy=None)

    def test_omit_empty_columns(self) -> None:
        """Test that omit_empty_columns parameter works correctly."""
        client = self.client

        # Setup: Create trials with best trial
        client.get_next_trials(max_trials=2)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 2.0})

        experiment = client._experiment
        generation_strategy = client._generation_strategy

        # Execute: Compute with omit_empty_columns=False
        analysis_no_omit = BestTrials(omit_empty_columns=False)
        card_no_omit = analysis_no_omit.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Execute: Compute with omit_empty_columns=True (default)
        analysis_omit = BestTrials(omit_empty_columns=True)
        card_omit = analysis_omit.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Assert: no_omit should have more or equal columns (e.g., fail_reason)
        self.assertGreaterEqual(len(card_no_omit.df.columns), len(card_omit.df.columns))

    def test_trial_status_filter(self) -> None:
        """Test that trial_statuses parameter filters correctly."""
        client = self.client

        # Setup: Create trials with different statuses
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 2.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 1.0})
        client.mark_trial_failed(trial_index=2)

        experiment = client._experiment
        generation_strategy = client._generation_strategy

        # Execute: Compute BestTrials with only COMPLETED status filter
        analysis = BestTrials(trial_statuses=[TrialStatus.COMPLETED])
        card = analysis.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # Assert: Only COMPLETED trials should be in results
        self.assertTrue(all(card.df["trial_status"] == "COMPLETED"))
        # Best completed trial should have foo=2.0 (maximizing, default)
        self.assertEqual(card.df["foo"].iloc[0], 2.0)

    def test_use_model_predictions_parameter(self) -> None:
        """Test that use_model_predictions parameter works for both True and False."""
        client = self.client

        # Setup: Create trials with data
        client.get_next_trials(max_trials=3)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})
        client.complete_trial(trial_index=1, raw_data={"foo": 2.0})
        client.complete_trial(trial_index=2, raw_data={"foo": 3.0})

        experiment = client._experiment
        generation_strategy = client._generation_strategy

        with self.subTest(msg="Using raw observations", use_model_predictions=False):
            analysis = BestTrials(use_model_predictions=False)
            card = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

            # Assert: Should return a valid result using raw observations
            self.assertEqual(card.name, "BestTrials")
            self.assertIsNotNone(card.df)
            self.assertGreater(len(card.df), 0)
            # Should select the best trial (foo=3.0 for maximization)
            self.assertEqual(card.df["foo"].iloc[0], 3.0)

        # Note: Model predictions may not be available with limited data,
        # so we just verify the parameter is accepted without errors
        with self.subTest(msg="Using model predictions", use_model_predictions=True):
            analysis = BestTrials(use_model_predictions=True)
            # Should either succeed or raise a specific error about insufficient data
            try:
                card = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )
                # If successful, verify basic structure
                self.assertEqual(card.name, "BestTrials")
                self.assertIsNotNone(card.df)
                self.assertGreater(len(card.df), 0)
            except UserInputError as e:
                # Expected when model predictions cannot be generated
                self.assertIn("No best trial", str(e))
