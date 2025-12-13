#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import OptimizationNotConfiguredError
from ax.service.orchestrator import OrchestratorOptions
from ax.utils.common.complexity_utils import summarize_ax_optimization_complexity
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

        # THEN the summary should contain all expected keys with correct values
        expected_keys = [
            "max_trials",
            "num_params",
            "num_binary",
            "num_categorical_3_5",
            "num_categorical_6_inf",
            "num_parameter_constraints",
            "num_objectives",
            "num_outcome_constraints",
            "uses_early_stopping",
            "uses_global_stopping",
            "uses_merge_multiple_curves",
            "all_inputs_are_configs",
            "tolerated_trial_failure_rate",
            "max_pending_trials",
            "min_failed_trials_for_failure_rate_check",
        ]
        for key in expected_keys:
            self.assertIn(key, summary)

        # Validate specific values for single-objective experiment
        self.assertEqual(summary["num_objectives"], 1)
        self.assertFalse(summary["uses_early_stopping"])
        self.assertFalse(summary["uses_global_stopping"])

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
        self.assertGreater(summary["num_objectives"], 1)

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
                self.assertEqual(summary["max_trials"], expected_max_trials)
                self.assertEqual(
                    summary["all_inputs_are_configs"], expected_all_configs
                )

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
        self.assertEqual(summary["tolerated_trial_failure_rate"], 0.25)
        self.assertEqual(summary["max_pending_trials"], 5)
        self.assertEqual(summary["min_failed_trials_for_failure_rate_check"], 10)

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
        self.assertGreater(summary["num_parameter_constraints"], 0)
