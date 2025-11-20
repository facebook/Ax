# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.analysis.plotly.utility_progression import UtilityProgressionAnalysis
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


class TestUtilityProgressionAnalysis(TestCase):
    def test_utility_progression_soo(self) -> None:
        """Test that UtilityProgressionAnalysis works for SOO experiments."""
        # Setup: Create SOO experiment with completed trials
        experiment = get_branin_experiment(with_completed_trial=True, num_trial=3)

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate that analysis is applicable
        validation_result = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Analysis should be applicable (no error message)
        self.assertIsNone(validation_result)

        # Execute: Compute the analysis
        card = analysis.compute(experiment=experiment)

        # Assert: Check that we got a valid card with correct structure
        self.assertIsInstance(card, PlotlyAnalysisCard)
        self.assertEqual(card.name, "UtilityProgressionAnalysis")
        self.assertIn("Best", card.title)
        self.assertIn("branin", card.title)  # Metric name
        self.assertIn("trial_index", card.df.columns)
        self.assertIn("utility", card.df.columns)
        self.assertEqual(len(card.df), 3)  # 3 completed trials

    def test_utility_progression_moo(self) -> None:
        """Test that UtilityProgressionAnalysis works for MOO experiments."""
        # Setup: Create MOO experiment with completed trials and data
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=True,
            with_completed_batch=True,
            with_status_quo=False,
        )

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate that analysis is applicable
        validation_result = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Analysis should be applicable (no error message)
        self.assertIsNone(validation_result)

        # Execute: Compute the analysis
        card = analysis.compute(experiment=experiment)

        # Assert: Check that we got a valid card with correct structure
        self.assertIsInstance(card, PlotlyAnalysisCard)
        self.assertEqual(card.name, "UtilityProgressionAnalysis")
        self.assertIn("Hypervolume", card.title)
        self.assertIn("Pareto", card.title)
        self.assertIn("trial_index", card.df.columns)
        self.assertIn("utility", card.df.columns)
        self.assertGreater(len(card.df), 0)

        # Assert: Check that subtitle is the generic utility progression subtitle
        self.assertIn("utility progression", card.subtitle.lower())
        self.assertIn("multi-objective", card.subtitle.lower())

        # Assert: Hypervolume should be non-decreasing
        utility_values = card.df["utility"].tolist()
        for i in range(1, len(utility_values)):
            self.assertGreaterEqual(
                utility_values[i],
                utility_values[i - 1],
                msg="Hypervolume should be non-decreasing",
            )

    def test_validation_no_optimization_config(self) -> None:
        """Test that validation fails when no optimization config is present."""
        # Setup: Create experiment with trials but without optimization config
        experiment = get_branin_experiment(with_completed_trial=True, num_trial=1)
        experiment._optimization_config = None

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate the analysis
        error_message = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail with appropriate error
        self.assertIsNotNone(error_message)
        self.assertIn("optimization config", error_message.lower())

    def test_validation_moo_no_reference_point(self) -> None:
        """Test that validation requires reference point for MOO."""
        # Setup: Create basic MOO experiment
        experiment = get_branin_experiment_with_multi_objective(
            has_optimization_config=True,
            has_objective_thresholds=False,
            with_batch=False,
            with_status_quo=False,
        )

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate the analysis
        error_message = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail if no trials/data,
        # or require reference point if data exists
        self.assertIsNotNone(error_message)

    def test_validation_no_trials(self) -> None:
        """Test that validation fails when experiment has no trials."""
        # Setup: Create experiment with no trials
        experiment = get_branin_experiment(with_trial=False)

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate the analysis
        error_message = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail
        self.assertIsNotNone(error_message)

    def test_validation_no_data(self) -> None:
        """Test that validation fails when experiment has trials but no data."""
        # Setup: Create experiment with trial but without attaching data
        experiment = get_branin_experiment(with_trial=True)
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()
        # Don't call fetch_data(), so no data is available

        analysis = UtilityProgressionAnalysis()

        # Execute: Validate the analysis
        error_message = analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail due to no data
        self.assertIsNotNone(error_message)
