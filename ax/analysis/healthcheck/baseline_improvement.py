# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point_utils import select_baseline_name_default_first_trial
from ax.service.utils.report_utils import (
    _construct_comparison_message,
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
    - Identifying a baseline arm (explicit, status quo, or first trial)
    - Comparing best performing arms for each objective to baseline values
    - Computing percent improvement for metrics that beat baseline
    """

    def __init__(
        self,
        comparison_arm_names: list[str] | None = None,
        baseline_arm_name: str | None = None,
        footer_notes: str | None = None,
    ) -> None:
        """
        Args:
            comparison_arm_names: Names of arms to compare to baseline. If None,
                will attempt to use the best arms for each objective.
            baseline_arm_name: Name of baseline arm. If None, will be selected
                using the default selection logic (explicit -> status quo ->
                first trial).
            footer_notes: Optional footer text to append to the subtitle.
                This can be used to add context-specific information like
                documentation links or additional instructions.
        """
        self.comparison_arm_names = comparison_arm_names
        self.baseline_arm_name = baseline_arm_name
        self.footer_notes = footer_notes

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
        # Validate experiment exists
        experiment_validation_str = validate_experiment(experiment=experiment)
        if experiment_validation_str is not None:
            return experiment_validation_str

        experiment = none_throws(experiment)

        # Validate optimization config exists
        if experiment.optimization_config is None:
            return (
                "Experiment does not have an optimization_config. "
                "BaselineImprovementAnalysis requires objectives to compare against."
            )

        # Validate we can select a baseline arm
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

        # If comparison_arm_names not provided, analysis is not applicable
        # because we need explicit arms to compare
        if self.comparison_arm_names is None:
            return (
                "BaselineImprovementAnalysis requires explicit comparison_arm_names "
                "to be provided. This analysis cannot automatically determine which "
                "arms to compare to the baseline."
            )

        # Validate we can extract comparison values
        comparison_list = maybe_extract_baseline_comparison_values(
            experiment=experiment,
            optimization_config=experiment.optimization_config,
            comparison_arm_names=self.comparison_arm_names,
            baseline_arm_name=baseline_arm_name,
        )

        if comparison_list is None or len(comparison_list) == 0:
            return (
                "Could not extract baseline comparison values. Ensure the experiment "
                "has completed trials with data for the objective metrics."
            )

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        # Experiment is validated in validate_applicable_state
        experiment = none_throws(experiment)

        # If comparison_arm_names not provided, analysis is not applicable
        # Return early with a warning card instead of raising an error
        if self.comparison_arm_names is None:
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Baseline Improvement Healthcheck",
                subtitle=(
                    "Analysis requires explicit comparison_arm_names to be provided. "
                    "This analysis cannot automatically determine which arms to "
                    "compare to the baseline."
                ),
                df=pd.DataFrame(),
                status=HealthcheckStatus.WARNING,
            )

        # Select baseline arm
        try:
            baseline_arm_name, selected_from_first_arm = (
                select_baseline_name_default_first_trial(
                    experiment=experiment,
                    baseline_arm_name=self.baseline_arm_name,
                )
            )
        except Exception as e:
            raise UserInputError(
                f"Could not select baseline arm: {e}. Please specify a valid "
                "baseline_arm_name."
            )

        # Check if baseline arm is within the search space
        baseline_arm = experiment.arms_by_name[baseline_arm_name]
        baseline_arm_in_design = experiment.search_space.check_membership(
            parameterization=baseline_arm.parameters,
            raise_error=False,
            check_all_parameters_present=True,
        )

        # Extract comparison values
        comparison_list = maybe_extract_baseline_comparison_values(
            experiment=experiment,
            optimization_config=experiment.optimization_config,
            comparison_arm_names=self.comparison_arm_names,
            baseline_arm_name=baseline_arm_name,
        )

        if comparison_list is None or len(comparison_list) == 0:
            raise UserInputError(
                "Could not extract baseline comparison values. "
                "Ensure the experiment has completed trials with data."
            )

        # Determine which objectives improved
        improved_objectives: list[str] = []
        not_improved_objectives: list[str] = []
        improvement_details: list[dict[str, str]] = []

        for (
            metric_name,
            minimize,
            baseline_arm,
            baseline_val,
            comparison_arm,
            comp_val,
        ) in comparison_list:
            # Check if improved
            improved = (minimize and baseline_val > comp_val) or (
                not minimize and baseline_val < comp_val
            )

            if improved:
                improved_objectives.append(metric_name)
                # Get formatted improvement message
                message = _construct_comparison_message(
                    objective_name=metric_name,
                    objective_minimize=minimize,
                    baseline_arm_name=baseline_arm,
                    baseline_value=baseline_val,
                    comparison_arm_name=comparison_arm,
                    comparison_value=comp_val,
                )
                if message:
                    improvement_details.append(
                        {
                            "Metric": metric_name,
                            "Status": "✓ Improved",
                            "Details": message.strip(),
                        }
                    )
            else:
                not_improved_objectives.append(metric_name)
                direction = "decreased" if minimize else "increased"
                improvement_details.append(
                    {
                        "Metric": metric_name,
                        "Status": "✗ Not Improved",
                        "Details": (
                            f"Metric `{metric_name}` {direction} from "
                            f"`{baseline_val:.4g}` (baseline) to `{comp_val:.4g}` "
                            f"(comparison), no improvement."
                        ),
                    }
                )

        # Determine status
        num_objectives = len(comparison_list)
        num_improved = len(improved_objectives)

        if num_improved == num_objectives:
            status = HealthcheckStatus.PASS
            subtitle = f"All {num_objectives} objective(s) improved over baseline.\n\n"
        elif num_improved > 0:
            status = HealthcheckStatus.WARNING
            subtitle = (
                f"{num_improved} out of {num_objectives} objective(s) "
                "improved over baseline.\n\n"
            )
        else:
            status = HealthcheckStatus.FAIL
            subtitle = (
                f"None of the {num_objectives} objective(s) improved over baseline.\n\n"
            )

        # Add baseline info
        if selected_from_first_arm:
            subtitle += (
                f"**Note:** Using the first trial's first arm ('{baseline_arm_name}') "
                "as the baseline since no explicit baseline was provided.\n\n"
            )
        else:
            subtitle += f"**Baseline arm:** '{baseline_arm_name}'\n\n"

        # Add note if baseline arm is out-of-design
        if not baseline_arm_in_design:
            subtitle += "**Note:** The baseline arm is out-of-design.\n\n"

        # Add improvement details to subtitle
        if num_objectives > 1:
            subtitle += "**Objective-by-objective breakdown:**\n"
            for detail in improvement_details:
                subtitle += f"\n- {detail['Details']}"
        elif num_objectives == 1:
            subtitle += improvement_details[0]["Details"]

        # Add footer notes if provided
        if self.footer_notes:
            subtitle += f"\n\n{self.footer_notes}"

        # Create dataframe
        df = pd.DataFrame(improvement_details)

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Baseline Improvement Healthcheck",
            subtitle=subtitle,
            df=df,
            status=status,
            num_objectives_improved=num_improved,
            num_objectives_total=num_objectives,
            baseline_arm_name=baseline_arm_name,
        )
