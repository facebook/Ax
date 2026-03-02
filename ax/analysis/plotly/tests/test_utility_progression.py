# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from unittest.mock import patch

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.analysis.plotly.utility_progression import UtilityProgressionAnalysis
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import PreferenceOptimizationConfig
from ax.exceptions.core import ExperimentNotReadyError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_experiment_with_custom_runner_and_metric,
)
from ax.utils.testing.preference_stubs import get_pbo_experiment


class TestUtilityProgressionAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.analysis = UtilityProgressionAnalysis()

    def _assert_valid_utility_card(
        self,
        card: PlotlyAnalysisCard,
    ) -> None:
        """Assert that a card has valid structure for utility progression."""
        self.assertIsInstance(card, PlotlyAnalysisCard)
        self.assertEqual(card.name, "UtilityProgressionAnalysis")
        self.assertIn("trace_index", card.df.columns)
        self.assertIn("utility", card.df.columns)

    def test_utility_progression_soo(self) -> None:
        """Test that UtilityProgressionAnalysis works for SOO experiments."""
        # Setup: Create SOO experiment with completed trials
        experiment = get_branin_experiment(with_completed_trial=True, num_trial=3)

        # Execute: Validate that analysis is applicable
        validation_result = self.analysis.validate_applicable_state(
            experiment=experiment
        )

        # Assert: Analysis should be applicable (no error message)
        self.assertIsNone(validation_result)

        # Execute: Compute the analysis
        card = self.analysis.compute(experiment=experiment)

        # Assert: Check that we got a valid card with correct structure
        self._assert_valid_utility_card(card)
        self.assertEqual(len(card.df), 3)  # 3 completed trials

    def test_utility_progression_moo(self) -> None:
        """Test that UtilityProgressionAnalysis works for MOO experiments."""
        # Setup: Create MOO experiment with completed trials and data
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=True,
            with_completed_batch=True,
            with_status_quo=True,
        )
        # Execute: Validate that analysis is applicable
        validation_result = self.analysis.validate_applicable_state(
            experiment=experiment
        )
        # Assert: Analysis should be applicable (no error message)
        self.assertIsNone(validation_result)

        # Execute: Compute the analysis
        card = self.analysis.compute(experiment=experiment)

        # Assert: Check that we got a valid card with correct structure
        self._assert_valid_utility_card(card)
        self.assertGreater(len(card.df), 0)

        # Assert: Check that subtitle contains expected content for MOO
        self.assertIn("hypervolume", card.subtitle.lower())
        self.assertIn("pareto frontier", card.subtitle.lower())

        # Assert: Hypervolume should be non-decreasing
        utility_values = card.df["utility"].tolist()
        for i in range(1, len(utility_values)):
            self.assertGreaterEqual(
                utility_values[i],
                utility_values[i - 1],
                msg="Hypervolume should be non-decreasing",
            )

    def test_utility_progression_bope(self) -> None:
        """Test UtilityProgressionAnalysis for BOPE experiments."""
        metric_names = ["branin_a", "branin_b"]

        # Setup: Create main BO experiment with metric data
        bope_experiment = get_pbo_experiment(
            num_experimental_metrics=len(metric_names),
            tracking_metric_names=metric_names,
            num_experimental_trials=3,
            num_preference_trials=0,
        )

        # Setup: Create and attach PE_EXPERIMENT with preference data
        pe_experiment = get_pbo_experiment(
            parameter_names=metric_names,
            num_experimental_metrics=0,
            num_experimental_trials=0,
            num_preference_trials=2,
            unbounded_search_space=True,
            experiment_name="test_profile",
        )

        # Setup: Attach PE_EXPERIMENT with preference data to main experiment
        bope_experiment.add_auxiliary_experiment(
            auxiliary_experiment=AuxiliaryExperiment(
                experiment=pe_experiment, data=pe_experiment.lookup_data()
            ),
            purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
        )

        # Setup: Set PreferenceOptimizationConfig
        bope_experiment.optimization_config = PreferenceOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric(name=m), minimize=False)
                    for m in metric_names
                ]
            ),
            preference_profile_name="test_profile",
        )

        # Execute & Assert
        self.assertIsNone(
            self.analysis.validate_applicable_state(experiment=bope_experiment)
        )

        # Execute: Compute the analysis
        card = self.analysis.compute(experiment=bope_experiment)

        # Assert: Check that we got a valid card with correct structure
        self._assert_valid_utility_card(card)
        self.assertGreater(len(card.df), 0)

        # Assert: Check that subtitle mentions preference/utility-based trace
        self.assertIn("preference", card.subtitle.lower())

    def test_validation_no_optimization_config(self) -> None:
        """Test that validation fails when no optimization config is present."""
        # Setup: Create experiment with trials but without optimization config
        experiment = get_branin_experiment(with_completed_trial=True, num_trial=1)
        experiment._optimization_config = None

        # Execute: Validate the analysis
        error_message = self.analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail with appropriate error
        self.assertIsNotNone(error_message)
        self.assertIn("optimization config", error_message.lower())

    def test_validation_no_trials(self) -> None:
        """Test that validation fails when experiment has no trials."""
        # Setup: Create experiment with no trials
        experiment = get_branin_experiment(with_trial=False)

        # Execute: Validate the analysis
        error_message = self.analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail due to no trials
        self.assertEqual(error_message, "Experiment has no trials.")

    def test_validation_no_data(self) -> None:
        """Test that validation fails when experiment has trials but no data."""
        # Setup: Create experiment with trial but without attaching data
        experiment = get_branin_experiment(with_trial=True)
        experiment.trials[0].mark_running(no_runner_required=True)

        # Execute: Validate the analysis
        error_message = self.analysis.validate_applicable_state(experiment=experiment)

        # Assert: Validation should fail due to no data
        self.assertEqual(error_message, "Experiment has no data.")

    def test_scalarized_objective_support(self) -> None:
        """Test that ScalarizedObjective works with UtilityProgressionAnalysis."""
        # Setup: Create experiment with ScalarizedObjective
        experiment = get_experiment_with_custom_runner_and_metric(
            scalarized_objective=True,
            num_trials=3,
        )

        # Execute: Compute the analysis
        card = self.analysis.compute(experiment=experiment)

        # Assert: Check that we got a valid card
        self._assert_valid_utility_card(card)
        self.assertEqual(len(card.df), 3)

        # Assert: Check that title/subtitle show the formula
        self.assertIn("formula:", card.subtitle)

    def test_all_infeasible_points_raises_error(self) -> None:
        """Test that an error is raised when all points are infeasible."""
        experiment = get_branin_experiment(with_completed_trial=True)

        with (
            patch(
                "ax.analysis.plotly.utility_progression.get_trace",
                return_value=[math.inf, -math.inf, math.inf],
            ),
            self.assertRaises(ExperimentNotReadyError) as cm,
        ):
            self.analysis.compute(experiment=experiment)

        self.assertIn("infeasible", str(cm.exception).lower())
