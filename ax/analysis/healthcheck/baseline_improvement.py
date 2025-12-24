# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.utils import validate_experiment
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point_utils import (
    get_best_trial_indices,
    select_baseline_name_default_first_trial,
)
from ax.service.utils.report_utils import (
    construct_comparison_message,
    maybe_extract_baseline_comparison_values,
)
from pyre_extensions import none_throws, override


@final
class BaselineImprovementAnalysis(Analysis):
    """
    Healthcheck that evaluates whether the optimization has improved over a
    baseline.

    This analysis compares the best performing arms to a baseline arm for each
    objective metric and determines if there is improvement.

    Status Logic:
        - PASS: All objectives improved over baseline
        - WARNING: Some objectives improved over baseline
        - FAIL: No objectives improved over baseline

    The healthcheck evaluates improvement by:
        1. Identifying a baseline arm (explicit, status quo, or first trial)
        2. Comparing best performing arms for each objective to baseline values
        3. Computing percent improvement for metrics that beat baseline
    """

    def __init__(
        self,
        comparison_arm_names: Sequence[str] | None = None,
        baseline_arm_name: str | None = None,
        documentation_link: str | None = None,
        no_improvement_message: str | None = None,
    ) -> None:
        """
        Args:
            comparison_arm_names: Names of arms to compare to baseline. If None,
                will automatically use all non-baseline arms from the experiment.
                The analysis will then select the best performing arm(s) for each
                objective from this list.
            baseline_arm_name: Name of baseline arm. If None, will be selected
                using the default selection logic (explicit -> status quo ->
                first trial).
            documentation_link: Optional link to documentation about baseline
                configuration. If provided, will be appended to help messages
                with appropriate context depending on whether the baseline was
                explicitly provided or auto-selected.
            no_improvement_message: Optional custom message to display when no
                objectives improved over baseline. This can be used for directing users
                to support channels or sites.
        """
        self.comparison_arm_names: list[str] | None = (
            list(comparison_arm_names) if comparison_arm_names is not None else None
        )
        self.baseline_arm_name = baseline_arm_name
        self.documentation_link = documentation_link
        self.no_improvement_message = no_improvement_message

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        Validates that the experiment is in an applicable state for baseline
        improvement analysis.

        Requires:
            - A valid experiment
            - An optimization config with objectives
            - At least one completed trial with data
        """
        experiment_validation = validate_experiment(experiment=experiment)
        if experiment_validation is not None:
            return experiment_validation

        experiment = none_throws(experiment)

        if experiment.optimization_config is None:
            return (
                "Experiment does not have an `OptimizationConfig`. "
                "BaselineImprovementAnalysis requires defined objectives to compare "
                "against for proper evaluation."
            )

        try:
            baseline_arm_name, _ = select_baseline_name_default_first_trial(
                experiment=experiment,
                baseline_arm_name=self.baseline_arm_name,
            )
        except Exception as e:
            return (
                f"Could not select baseline arm: {e}. Please ensure the "
                "experiment has at least one trial with data, or specify a "
                "valid baseline_arm_name."
            )

        comparison_arm_names = self._get_comparison_arm_names(
            experiment=experiment,
            baseline_arm_name=baseline_arm_name,
            generation_strategy=generation_strategy,
        )

        if len(comparison_arm_names) == 0:
            return (
                "No comparison arms available. The experiment needs at least one "
                "arm besides the baseline to compare against."
            )

        comparison_list = maybe_extract_baseline_comparison_values(
            experiment=experiment,
            optimization_config=experiment.optimization_config,
            comparison_arm_names=comparison_arm_names,
            baseline_arm_name=baseline_arm_name,
        )

        if comparison_list is None or len(comparison_list) == 0:
            return (
                "Could not extract baseline comparison values. Ensure the experiment "
                "has completed trials with data for the objective metrics. Also verify "
                "that the objective metric names match those in your trial data."
            )

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        """Compute the baseline improvement healthcheck."""
        experiment = none_throws(experiment)

        # Select baseline and get comparison arms
        # Validated in validate_applicable_state
        baseline_arm_name, auto_selected_from_first_arm = (
            select_baseline_name_default_first_trial(
                experiment=experiment,
                baseline_arm_name=self.baseline_arm_name,
            )
        )

        # Validated in validate_applicable_state
        comparison_arm_names = self._get_comparison_arm_names(
            experiment=experiment,
            baseline_arm_name=baseline_arm_name,
            generation_strategy=generation_strategy,
        )

        # Extract comparison values using existing utility
        # Validated in validate_applicable_state
        comparison_list = none_throws(
            maybe_extract_baseline_comparison_values(
                experiment=experiment,
                optimization_config=experiment.optimization_config,
                comparison_arm_names=comparison_arm_names,
                baseline_arm_name=baseline_arm_name,
            )
        )

        # Calculate improvements
        improved: list[str] = []
        not_improved: list[str] = []
        details: list[dict[str, str]] = []

        for metric, minimize, bl_arm, bl_val, comp_arm, comp_val in comparison_list:
            is_better = (minimize and bl_val > comp_val) or (
                not minimize and bl_val < comp_val
            )

            if is_better:
                improved.append(metric)
                msg = construct_comparison_message(
                    objective_name=metric,
                    objective_minimize=minimize,
                    baseline_arm_name=bl_arm,
                    baseline_value=bl_val,
                    comparison_arm_name=comp_arm,
                    comparison_value=comp_val,
                )
                details.append(
                    {
                        "Metric": metric,
                        "Status": "Improved",
                        "Details": (msg or "").strip(),
                    }
                )
            else:
                not_improved.append(metric)
                details.append(
                    {
                        "Metric": metric,
                        "Status": "Not Improved",
                        "Details": (
                            f"Metric `{metric}` did not improve. "
                            f"Baseline: `{bl_val:.4g}`, Comparison: `{comp_val:.4g}`."
                        ),
                    }
                )

        # Determine status
        num_improved = len(improved)
        num_total = len(comparison_list)

        if num_improved == num_total:
            status = HealthcheckStatus.PASS
        elif num_improved > 0:
            status = HealthcheckStatus.WARNING
        else:
            status = HealthcheckStatus.FAIL

        # Build subtitle
        subtitle = self._build_subtitle(
            num_improved=num_improved,
            num_total=num_total,
            not_improved=not_improved,
            details=details,
            status=status,
            baseline_arm_name=baseline_arm_name,
            auto_selected_from_first_arm=auto_selected_from_first_arm,
            baseline_in_design=experiment.search_space.check_membership(
                parameterization=experiment.arms_by_name[baseline_arm_name].parameters,
                raise_error=False,
                check_all_parameters_present=True,
            ),
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Baseline Improvement Healthcheck",
            subtitle=subtitle,
            df=pd.DataFrame(details),
            status=status,
            num_objectives_improved=num_improved,
            num_objectives_total=num_total,
            baseline_arm_name=baseline_arm_name,
        )

    def _get_comparison_arm_names(
        self,
        experiment: Experiment,
        baseline_arm_name: str,
        generation_strategy: GenerationStrategy | None = None,
    ) -> list[str]:
        """Get the list of arm names to compare against the baseline.

        Note: This method assumes optimization_config is not None, as validated
        by validate_applicable_state before this method is called.
        """
        if self.comparison_arm_names is not None:
            return self.comparison_arm_names

        # Try to auto-detect best arms
        best_arms = self._try_get_best_arms(experiment, generation_strategy)
        if best_arms:
            return best_arms

        return [n for n in experiment.arms_by_name.keys() if n != baseline_arm_name]

    def _try_get_best_arms(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy | None,
    ) -> list[str] | None:
        """Try to automatically detect the best arms for comparison.

        Uses AutoML-geared logic to find best trials.

        Note: This method assumes optimization_config is not None, as validated
        by validate_applicable_state before this method is called.
        """
        optimization_config = none_throws(experiment.optimization_config)

        try:
            best_trial_indices = get_best_trial_indices(
                experiment=experiment,
                optimization_config=optimization_config,
                generation_strategy=generation_strategy,
                trial_indices=None,
                use_model_predictions=False,
            )
            if not best_trial_indices:
                return None

            return [
                arm.name
                for idx in best_trial_indices
                for arm in experiment.trials[idx].arms
            ] or None
        except Exception:
            return None

    def _build_subtitle(
        self,
        num_improved: int,
        num_total: int,
        not_improved: list[str],
        details: list[dict[str, str]],
        status: HealthcheckStatus,
        baseline_arm_name: str,
        auto_selected_from_first_arm: bool,
        baseline_in_design: bool,
    ) -> str:
        """Build the subtitle text for the healthcheck card."""
        parts: list[str] = []

        # Status summary
        if status == HealthcheckStatus.PASS:
            parts.append(f"All {num_total} objective(s) improved over baseline.")
        elif status == HealthcheckStatus.WARNING:
            parts.append(
                f"{num_improved} out of {num_total} objective(s) "
                "improved over baseline. The following metrics were not improved: "
                f"{not_improved}."
            )
        elif self.no_improvement_message:
            parts.append(self.no_improvement_message)
        else:
            parts.append(
                f"None of the {num_total} objective(s) improved over baseline."
            )

        # Baseline info
        if not auto_selected_from_first_arm:
            parts.append(f"**Baseline arm:** '{baseline_arm_name}'")

        # Objective breakdown
        if num_total > 1:
            lines = ["**Objective-by-objective breakdown:**"]
            lines.extend(f"- {d['Details']}" for d in details)
            parts.append("\n".join(lines))
        elif num_total == 1:
            parts.append(details[0]["Details"])

        # Documentation link
        if self.documentation_link:
            if auto_selected_from_first_arm:
                parts.append(
                    "To manually set the baseline, and for more information on "
                    "performance measurement, "
                    f"please see {self.documentation_link}"
                )
            else:
                parts.append(
                    f"For more information on performance measurement, "
                    f"please see {self.documentation_link}"
                )

        # Notes
        if auto_selected_from_first_arm:
            parts.append(
                f"**Note:** Using the first trial's first arm ('{baseline_arm_name}') "
                "as the baseline since no explicit baseline was provided."
            )
        if not baseline_in_design:
            parts.append("**Note:** The baseline arm is out-of-design.")

        return "\n\n".join(parts)
