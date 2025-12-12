#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import OptimizationNotConfiguredError, UserInputError
from ax.service.orchestrator import OrchestratorOptions
from ax.utils.common.complexity_utils import (
    ADVANCED_TIER_MESSAGE,
    check_if_in_wheelhouse,
    format_tier_message,
    OptimizationSummary,
    summarize_ax_optimization_complexity,
    UNSUPPORTED_TIER_MESSAGE,
    WHEELHOUSE_TIER_MESSAGE,
)

from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_experiment_with_multi_objective,
)


class TestSummarizeAxOptimizationComplexity(TestCase):
    """Tests for the summarize_ax_optimization_complexity function."""

    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_experiment()
        self.options = OrchestratorOptions()
        self.tier_metadata: dict[str, object] = {}

    def test_basic_experiment_summary(self) -> None:
        # GIVEN a basic experiment with single objective (from setUp)

        # WHEN we summarize the experiment
        summary = summarize_ax_optimization_complexity(
            experiment=self.experiment,
            options=self.options,
            tier_metadata=self.tier_metadata,
        )

        # THEN the summary should be an OptimizationSummary with correct values
        self.assertIsInstance(summary, OptimizationSummary)

        # Validate specific values for single-objective experiment
        self.assertEqual(summary.num_objectives, 1)
        self.assertFalse(summary.uses_early_stopping)
        self.assertFalse(summary.uses_global_stopping)

    def test_multi_objective_experiment(self) -> None:
        # GIVEN a multi-objective experiment
        experiment = get_experiment_with_multi_objective()

        # WHEN we summarize the experiment
        summary = summarize_ax_optimization_complexity(
            experiment=experiment,
            options=self.options,
            tier_metadata=self.tier_metadata,
        )

        # THEN num_objectives should be greater than 1
        self.assertGreater(summary.num_objectives, 1)

    def test_experiment_without_optimization_config_raises(self) -> None:
        # GIVEN an experiment without optimization config
        self.experiment._optimization_config = None

        # WHEN/THEN summarizing should raise OptimizationNotConfiguredError
        with self.assertRaisesRegex(
            OptimizationNotConfiguredError,
            "Experiment must have an optimization_config",
        ):
            summarize_ax_optimization_complexity(
                experiment=self.experiment,
                options=self.options,
                tier_metadata=self.tier_metadata,
            )

    def test_tier_metadata_extraction(self) -> None:
        # Test that tier_metadata values are correctly extracted
        test_cases = [
            (
                "with_values",
                {"user_supplied_max_trials": 50, "all_inputs_are_configs": True},
                50,
                True,
            ),
            (
                "empty_defaults",
                {},
                None,
                False,
            ),
        ]

        for (
            name,
            tier_metadata,
            expected_max_trials,
            expected_all_configs,
        ) in test_cases:
            with self.subTest(name=name):
                # WHEN we summarize the experiment
                summary = summarize_ax_optimization_complexity(
                    experiment=self.experiment,
                    options=self.options,
                    tier_metadata=tier_metadata,
                )

                # THEN the summary should reflect tier metadata values
                self.assertEqual(summary.max_trials, expected_max_trials)
                self.assertEqual(summary.all_inputs_are_configs, expected_all_configs)

    def test_orchestrator_options_extraction(self) -> None:
        # GIVEN custom orchestrator options
        options = OrchestratorOptions(
            tolerated_trial_failure_rate=0.25,
            max_pending_trials=5,
            min_failed_trials_for_failure_rate_check=10,
        )

        # WHEN we summarize the experiment
        summary = summarize_ax_optimization_complexity(
            experiment=self.experiment,
            options=options,
            tier_metadata=self.tier_metadata,
        )

        # THEN the summary should reflect orchestrator options
        self.assertEqual(summary.tolerated_trial_failure_rate, 0.25)
        self.assertEqual(summary.max_pending_trials, 5)
        self.assertEqual(summary.min_failed_trials_for_failure_rate_check, 10)

    def test_parameter_constraints_counted(self) -> None:
        # GIVEN an experiment with parameter constraints
        experiment = get_experiment(constrain_search_space=True)

        # WHEN we summarize the experiment
        summary = summarize_ax_optimization_complexity(
            experiment=experiment,
            options=self.options,
            tier_metadata=self.tier_metadata,
        )

        # THEN num_parameter_constraints should be greater than 0
        self.assertGreater(summary.num_parameter_constraints, 0)


class TestFormatTierMessage(TestCase):
    """Tests for format_tier_message."""

    def test_tier_messages(self) -> None:
        """Test formatting of tier messages for all tiers."""
        test_cases: list[
            tuple[
                str,
                list[str] | None,
                list[str] | None,
                str,
                list[str],
            ]
        ] = [
            (
                "Wheelhouse",
                None,
                None,
                WHEELHOUSE_TIER_MESSAGE,
                ["tier 'Wheelhouse'"],
            ),
            (
                "Advanced",
                ["51 tunable parameters", "Early stopping is enabled"],
                None,
                ADVANCED_TIER_MESSAGE,
                [
                    "tier 'Advanced'",
                    "Why this experiment is not in the 'Wheelhouse' tier:",
                    "51 tunable parameters",
                    "Early stopping is enabled",
                ],
            ),
            (
                "Unsupported",
                ["51 tunable parameters"],
                ["201 tunable parameters"],
                UNSUPPORTED_TIER_MESSAGE,
                [
                    "tier 'Unsupported'",
                    "Why this experiment is not in the 'Wheelhouse' tier:",
                    "51 tunable parameters",
                    "Why this experiment is not in the 'Advanced' tier:",
                    "201 tunable parameters",
                ],
            ),
        ]

        for (
            tier,
            why_not_wheelhouse,
            why_not_supported,
            expected_message,
            expected_contents,
        ) in test_cases:
            with self.subTest(tier=tier):
                msg = format_tier_message(
                    tier=tier,
                    why_not_is_in_wheelhouse=why_not_wheelhouse,
                    why_not_supported=why_not_supported,
                )
                self.assertIn(expected_message, msg)
                for content in expected_contents:
                    self.assertIn(content, msg)

    def test_unknown_tier_raises_error(self) -> None:
        """Test that unknown tier raises ValueError."""
        with self.assertRaisesRegex(ValueError, 'Got unexpected tier "BadTier"'):
            format_tier_message(
                tier="BadTier",
                why_not_is_in_wheelhouse=None,
                why_not_supported=None,
            )


def get_experiment_summary(
    max_trials: int | None = 100,
    num_params: int = 10,
    num_binary: int = 0,
    num_categorical_3_5: int = 0,
    num_categorical_6_inf: int = 0,
    num_parameter_constraints: int = 0,
    num_objectives: int = 1,
    num_outcome_constraints: int = 0,
    uses_early_stopping: bool = False,
    uses_global_stopping: bool = False,
    all_inputs_are_configs: bool = True,
    tolerated_trial_failure_rate: float | None = 0.5,
    max_pending_trials: int | None = 5,
    min_failed_trials_for_failure_rate_check: int | None = 5,
    non_default_advanced_options: bool | None = None,
    uses_merge_multiple_curves: bool | None = None,
) -> OptimizationSummary:
    """Create an OptimizationSummary for testing."""
    return OptimizationSummary(
        max_trials=max_trials,
        num_params=num_params,
        num_binary=num_binary,
        num_categorical_3_5=num_categorical_3_5,
        num_categorical_6_inf=num_categorical_6_inf,
        num_parameter_constraints=num_parameter_constraints,
        num_objectives=num_objectives,
        num_outcome_constraints=num_outcome_constraints,
        uses_early_stopping=uses_early_stopping,
        uses_global_stopping=uses_global_stopping,
        all_inputs_are_configs=all_inputs_are_configs,
        tolerated_trial_failure_rate=tolerated_trial_failure_rate,
        max_pending_trials=max_pending_trials,
        min_failed_trials_for_failure_rate_check=(
            min_failed_trials_for_failure_rate_check
        ),
        non_default_advanced_options=non_default_advanced_options,
        uses_merge_multiple_curves=uses_merge_multiple_curves,
    )


class TestCheckIfInWheelhouse(TestCase):
    """Tests for check_if_in_wheelhouse."""

    def setUp(self) -> None:
        super().setUp()
        self.base_summary = get_experiment_summary()

    def test_wheelhouse_tier_for_simple_experiment(self) -> None:
        """Test that a simple experiment is classified as Wheelhouse tier."""
        tier, why_not_wheelhouse, why_not_supported = check_if_in_wheelhouse(
            self.base_summary
        )

        self.assertEqual(tier, "Wheelhouse")
        self.assertEqual(why_not_wheelhouse, None)
        self.assertEqual(why_not_supported, None)

    def test_advanced_tier_conditions(self) -> None:
        """Test conditions that result in Advanced tier."""
        test_cases: list[tuple[OptimizationSummary, str]] = [
            (get_experiment_summary(max_trials=250), "250 total trials"),
            (get_experiment_summary(num_params=60), "60 tunable parameter(s)"),
            (get_experiment_summary(num_binary=75), "75 binary tunable parameter(s)"),
            (
                get_experiment_summary(num_categorical_3_5=1),
                "1 unordered choice parameter(s)",
            ),
            (
                get_experiment_summary(num_parameter_constraints=4),
                "4 parameter constraints",
            ),
            (get_experiment_summary(num_objectives=3), "3 objectives"),
            (
                get_experiment_summary(num_outcome_constraints=3),
                "3 outcome constraints",
            ),
            (
                get_experiment_summary(uses_early_stopping=True),
                "Early stopping is enabled",
            ),
            (
                get_experiment_summary(uses_global_stopping=True),
                "Global stopping is enabled",
            ),
        ]

        for summary, expected_msg in test_cases:
            with self.subTest(expected_msg=expected_msg):
                tier, why_not_wheelhouse, why_not_supported = check_if_in_wheelhouse(
                    summary
                )

                self.assertEqual(tier, "Advanced")
                self.assertIsNotNone(why_not_wheelhouse)
                self.assertIn(expected_msg, why_not_wheelhouse[0])
                self.assertEqual(why_not_supported, None)

    def test_unsupported_tier_conditions(self) -> None:
        """Test conditions that result in Unsupported tier."""
        test_cases: list[tuple[OptimizationSummary, str]] = [
            (get_experiment_summary(max_trials=510), "510 total trials"),
            (get_experiment_summary(num_params=201), "201 tunable parameter(s)"),
            (get_experiment_summary(num_binary=101), "101 binary tunable parameter(s)"),
            (
                get_experiment_summary(num_categorical_3_5=6),
                "unordered choice parameters with more than 3 options",
            ),
            (
                get_experiment_summary(num_categorical_6_inf=2),
                "unordered choice parameters with more than 5 options",
            ),
            (
                get_experiment_summary(num_parameter_constraints=6),
                "6 parameter constraints",
            ),
            (get_experiment_summary(num_objectives=5), "5 objectives"),
            (
                get_experiment_summary(num_outcome_constraints=6),
                "6 outcome constraints",
            ),
            (
                get_experiment_summary(all_inputs_are_configs=False),
                "uses_standard_api=False",
            ),
            (
                get_experiment_summary(tolerated_trial_failure_rate=0.99),
                "tolerated_trial_failure_rate=0.99",
            ),
            (
                get_experiment_summary(non_default_advanced_options=True),
                "Non-default advanced_options",
            ),
            (
                get_experiment_summary(uses_merge_multiple_curves=True),
                "merge_multiple_curves=True",
            ),
            (
                get_experiment_summary(
                    max_pending_trials=3, min_failed_trials_for_failure_rate_check=7
                ),
                "min_failed_trials_for_failure_rate_check=7",
            ),
        ]

        for summary, expected_msg in test_cases:
            with self.subTest(expected_msg=expected_msg):
                tier, _, why_not_supported = check_if_in_wheelhouse(summary)

                self.assertEqual(tier, "Unsupported")
                self.assertIsNotNone(why_not_supported)
                self.assertIn(expected_msg, why_not_supported[0])

    def test_max_trials_none_raises(self) -> None:
        """Test max_trials=None with all_inputs_are_configs=True raises error."""
        summary = get_experiment_summary(all_inputs_are_configs=True, max_trials=None)

        with self.assertRaisesRegex(UserInputError, "`max_trials` should not be None!"):
            check_if_in_wheelhouse(summary)
