# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override


@final
class SupportTierHealthcheck(Analysis):
    """
    Healthcheck that evaluates whether the experiment configuration is in the
    "wheelhouse" (fully supported), supported but not wheelhouse, or unsupported.

    This analysis is particularly useful for AxSweep experiments, but can be adapted
    for general Ax experiments as well.

    Status Logic:
    - PASS: Configuration is in wheelhouse (fully supported)
    - WARNING: Configuration is supported but not in wheelhouse (advanced features)
    - FAIL: Configuration is unsupported

    The healthcheck evaluates:
    - Search space complexity (number of parameters, parameter constraints, etc.)
    - Optimization config (number of objectives and outcome constraints)
    - Other settings (early stopping, global stopping, trial limits, etc.)
    """

    def __init__(
        self,
        max_trials: int | None = None,
        uses_early_stopping: bool = False,
        uses_global_stopping: bool = False,
        tolerated_trial_failure_rate: float | None = None,
        max_pending_trials: int | None = None,
        min_failed_trials_for_failure_rate_check: int | None = None,
        all_inputs_are_configs: bool = True,
        uses_merge_multiple_curves: bool = False,
        non_default_advanced_options: bool = False,
    ) -> None:
        """
        Args:
            max_trials: Maximum number of trials for the experiment
            uses_early_stopping: Whether early stopping is enabled
            uses_global_stopping: Whether global stopping is enabled
            tolerated_trial_failure_rate: Maximum tolerated trial failure rate
            max_pending_trials: Maximum number of pending trials
            min_failed_trials_for_failure_rate_check: Minimum failed trials
                before checking failure rate
            all_inputs_are_configs: Whether all inputs use config objects
                (vs raw Experiment/GenerationStrategy abstractions)
            uses_merge_multiple_curves: Whether merge_multiple_curves is used
            non_default_advanced_options: Whether non-default advanced options
                are set on GenerationStrategyConfig
        """
        self.max_trials = max_trials
        self.uses_early_stopping = uses_early_stopping
        self.uses_global_stopping = uses_global_stopping
        self.tolerated_trial_failure_rate = tolerated_trial_failure_rate
        self.max_pending_trials = max_pending_trials
        self.min_failed_trials_for_failure_rate_check = (
            min_failed_trials_for_failure_rate_check
        )
        self.all_inputs_are_configs = all_inputs_are_configs
        self.uses_merge_multiple_curves = uses_merge_multiple_curves
        self.non_default_advanced_options = non_default_advanced_options

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        if experiment is None:
            raise UserInputError("SupportTierHealthcheck requires an Experiment.")

        # Gather experiment summary
        experiment_summary = self._summarize_experiment(experiment)

        # Determine tier
        tier, why_not_wheelhouse, why_not_supported = self._check_tier(
            experiment_summary
        )

        # Determine status
        if tier == "Wheelhouse":
            status = HealthcheckStatus.PASS
            subtitle = (
                "✓ This experiment configuration is in the 'Wheelhouse' tier.\n\n"
                "Experiments in this tier are fully supported and should not run into "
                "any problems. If an issue does occur, please reach out to the support team."
            )
        elif tier == "Advanced":
            status = HealthcheckStatus.WARNING
            subtitle = (
                "⚠ This experiment configuration is in the 'Advanced' tier.\n\n"
                "This experiment should technically run, but uses advanced features that "
                "may not be well-tested and/or may not be compatible with other advanced features."
            )
            if why_not_wheelhouse:
                subtitle += "\n\n**Why not in 'Wheelhouse' tier:**\n"
                for reason in why_not_wheelhouse:
                    subtitle += f"- {reason}\n"
        else:  # Unsupported
            status = HealthcheckStatus.FAIL
            subtitle = (
                "✗ This experiment configuration is in the 'Unsupported' tier.\n\n"
                "This configuration pushes beyond supported limits. Please consider "
                "simplifying your experiment configuration."
            )
            if why_not_wheelhouse:
                subtitle += "\n\n**Why not in 'Wheelhouse' tier:**\n"
                for reason in why_not_wheelhouse:
                    subtitle += f"- {reason}\n"
            if why_not_supported:
                subtitle += "\n\n**Why not in 'Advanced' tier:**\n"
                for reason in why_not_supported:
                    subtitle += f"- {reason}\n"

        # Create dataframe with experiment summary
        df = pd.DataFrame(
            [
                {
                    "Metric": "Support Tier",
                    "Value": tier,
                },
                {
                    "Metric": "Total Parameters",
                    "Value": str(experiment_summary["num_params"]),
                },
                {
                    "Metric": "Objectives",
                    "Value": str(experiment_summary["num_objectives"]),
                },
                {
                    "Metric": "Outcome Constraints",
                    "Value": str(experiment_summary["num_outcome_constraints"]),
                },
                {
                    "Metric": "Parameter Constraints",
                    "Value": str(experiment_summary["num_parameter_constraints"]),
                },
            ]
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Support Tier Healthcheck",
            subtitle=subtitle,
            df=df,
            status=status,
            tier=tier,
        )

    def _summarize_experiment(self, experiment: Experiment) -> dict[str, Any]:
        """Summarize the experiment configuration."""
        from ax.fb.adapter.utils import can_map_to_binary, is_unordered_choice

        search_space = experiment.search_space
        optimization_config = none_throws(experiment.optimization_config)
        params = search_space.tunable_parameters.values()

        num_params = len(search_space.tunable_parameters)
        num_binary = sum(can_map_to_binary(p) for p in params)
        num_categorical_3_5 = sum(
            is_unordered_choice(p, min_choices=3, max_choices=5) for p in params
        )
        num_categorical_6_inf = sum(
            is_unordered_choice(p, min_choices=6) for p in params
        )
        num_parameter_constraints = len(search_space.parameter_constraints)
        num_objectives = (
            len(optimization_config.objective.objectives)
            if isinstance(optimization_config.objective, MultiObjective)
            else 1
        )
        num_outcome_constraints = len(optimization_config.outcome_constraints)

        return {
            "max_trials": self.max_trials,
            "num_params": num_params,
            "num_binary": num_binary,
            "num_categorical_3_5": num_categorical_3_5,
            "num_categorical_6_inf": num_categorical_6_inf,
            "num_parameter_constraints": num_parameter_constraints,
            "num_objectives": num_objectives,
            "num_outcome_constraints": num_outcome_constraints,
            "uses_early_stopping": self.uses_early_stopping,
            "uses_global_stopping": self.uses_global_stopping,
            "tolerated_trial_failure_rate": self.tolerated_trial_failure_rate,
            "max_pending_trials": self.max_pending_trials,
            "min_failed_trials_for_failure_rate_check": self.min_failed_trials_for_failure_rate_check,
            "all_inputs_are_configs": self.all_inputs_are_configs,
            "uses_merge_multiple_curves": self.uses_merge_multiple_curves,
            "non_default_advanced_options": self.non_default_advanced_options,
        }

    def _check_tier(
        self, experiment_summary: dict[str, Any]
    ) -> tuple[str, list[str], list[str]]:
        """
        Check the support tier of the experiment.

        Returns:
            A tuple containing:
            - The tier string ("Wheelhouse", "Advanced", or "Unsupported")
            - A list of reasons for not being in the Wheelhouse
            - A list of reasons for not being Supported
        """
        is_in_wheelhouse = True
        is_supported = True
        why_not_wheelhouse: list[str] = []
        why_not_supported: list[str] = []

        # Check search space
        is_in_wheelhouse, is_supported = self._check_search_space(
            experiment_summary, why_not_wheelhouse, why_not_supported
        )

        # Check optimization config
        wh, sup = self._check_optimization_config(
            experiment_summary, why_not_wheelhouse, why_not_supported
        )
        is_in_wheelhouse &= wh
        is_supported &= sup

        # Check other settings
        wh, sup = self._check_other_settings(
            experiment_summary, why_not_wheelhouse, why_not_supported
        )
        is_in_wheelhouse &= wh
        is_supported &= sup

        # Return tier and messages
        if is_in_wheelhouse:
            return "Wheelhouse", [], []
        if is_supported:
            return "Advanced", why_not_wheelhouse, []
        return "Unsupported", why_not_wheelhouse, why_not_supported

    def _check_search_space(
        self,
        experiment_summary: dict[str, Any],
        why_not_wheelhouse: list[str],
        why_not_supported: list[str],
    ) -> tuple[bool, bool]:
        """Check the search space constraints."""
        is_in_wheelhouse = True
        is_supported = True

        num_params = experiment_summary["num_params"]
        num_binary = experiment_summary["num_binary"]
        num_categorical_3_5 = experiment_summary["num_categorical_3_5"]
        num_categorical_6_inf = experiment_summary["num_categorical_6_inf"]
        num_parameter_constraints = experiment_summary["num_parameter_constraints"]

        # Total tunable parameters
        if num_params > 50:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_params} tunable parameters (max in-wheelhouse is 50)"
            )
            if num_params > 200:
                is_supported = False
                why_not_supported.append(
                    f"{num_params} tunable parameters (max supported is 200)"
                )

        # Binary parameters
        if num_binary > 50:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_binary} binary parameters (max in-wheelhouse is 50)"
            )
            if num_binary > 100:
                is_supported = False
                why_not_supported.append(
                    f"{num_binary} binary parameters (max supported is 100)"
                )

        # Unordered choice parameters (3-5 options)
        if num_categorical_3_5 > 0:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_categorical_3_5} unordered choice parameters with 3-5 options "
                "(max in-wheelhouse is 0)"
            )
            if num_categorical_3_5 > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_categorical_3_5} unordered choice parameters with 3-5 options "
                    "(max supported is 5)"
                )

        # Unordered choice parameters (6+ options)
        if num_categorical_6_inf > 0:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_categorical_6_inf} unordered choice parameters with 6+ options "
                "(max in-wheelhouse is 0)"
            )
            if num_categorical_6_inf > 1:
                is_supported = False
                why_not_supported.append(
                    f"{num_categorical_6_inf} unordered choice parameters with 6+ options "
                    "(max supported is 1)"
                )

        # Parameter constraints
        if num_parameter_constraints > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_parameter_constraints} parameter constraints "
                "(max in-wheelhouse is 2)"
            )
            if num_parameter_constraints > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_parameter_constraints} parameter constraints "
                    "(max supported is 5)"
                )

        return is_in_wheelhouse, is_supported

    def _check_optimization_config(
        self,
        experiment_summary: dict[str, Any],
        why_not_wheelhouse: list[str],
        why_not_supported: list[str],
    ) -> tuple[bool, bool]:
        """Check the optimization config constraints."""
        is_in_wheelhouse = True
        is_supported = True

        num_objectives = experiment_summary["num_objectives"]
        num_outcome_constraints = experiment_summary["num_outcome_constraints"]

        # Objectives
        if num_objectives > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_objectives} objectives (max in-wheelhouse is 2)"
            )
            if num_objectives > 4:
                is_supported = False
                why_not_supported.append(
                    f"{num_objectives} objectives (max supported is 4)"
                )

        # Outcome constraints
        if num_outcome_constraints > 2:
            is_in_wheelhouse = False
            why_not_wheelhouse.append(
                f"{num_outcome_constraints} outcome constraints "
                "(max in-wheelhouse is 2)"
            )
            if num_outcome_constraints > 5:
                is_supported = False
                why_not_supported.append(
                    f"{num_outcome_constraints} outcome constraints "
                    "(max supported is 5)"
                )

        return is_in_wheelhouse, is_supported

    def _check_other_settings(
        self,
        experiment_summary: dict[str, Any],
        why_not_wheelhouse: list[str],
        why_not_supported: list[str],
    ) -> tuple[bool, bool]:
        """Check other settings like early stopping, trial limits, etc."""
        is_in_wheelhouse = True
        is_supported = True

        # Check if using simple inputs
        if not experiment_summary["all_inputs_are_configs"]:
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                "Using Ax abstractions (e.g., Experiment, GenerationStrategy) as inputs "
                "instead of simple config objects"
            )

        # Check max trials
        max_trials = experiment_summary.get("max_trials")
        if max_trials is not None:
            if max_trials > 200:
                is_in_wheelhouse = False
                why_not_wheelhouse.append(
                    f"{max_trials} total trials (max in-wheelhouse is 200)"
                )
                if max_trials > 500:
                    is_supported = False
                    why_not_supported.append(
                        f"{max_trials} total trials (max supported is 500)"
                    )

        # Check early stopping
        if experiment_summary["uses_early_stopping"]:
            is_in_wheelhouse = False
            why_not_wheelhouse.append("Early stopping is enabled")

        # Check global stopping
        if experiment_summary["uses_global_stopping"]:
            is_in_wheelhouse = False
            why_not_wheelhouse.append("Global stopping is enabled")

        # Check failure rate
        tolerated_trial_failure_rate = experiment_summary.get(
            "tolerated_trial_failure_rate"
        )
        if (
            tolerated_trial_failure_rate is not None
            and tolerated_trial_failure_rate > 0.9
        ):
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                f"Tolerated trial failure rate of {tolerated_trial_failure_rate} "
                "is larger than 0.9"
            )

        # Check failure rate check configuration
        max_pending_trials = experiment_summary.get("max_pending_trials")
        min_failed_trials_for_failure_rate_check = experiment_summary.get(
            "min_failed_trials_for_failure_rate_check"
        )
        if (
            max_pending_trials is not None
            and min_failed_trials_for_failure_rate_check is not None
            and max(2 * max_pending_trials, 5)
            < min_failed_trials_for_failure_rate_check
        ):
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                f"min_failed_trials_for_failure_rate_check "
                f"({min_failed_trials_for_failure_rate_check}) exceeds "
                f"{max(2 * max_pending_trials, 5)}. Please reduce "
                "min_failed_trials_for_failure_rate_check below the stated threshold."
            )

        # Check advanced options
        if experiment_summary["non_default_advanced_options"]:
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                "Non-default advanced_options are set on GenerationStrategyConfig"
            )

        # Check merge_multiple_curves
        if experiment_summary["uses_merge_multiple_curves"]:
            is_in_wheelhouse = False
            is_supported = False
            why_not_supported.append(
                "merge_multiple_curves=True is not supported (experimental feature)"
            )

        return is_in_wheelhouse, is_supported
