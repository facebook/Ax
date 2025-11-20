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
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
)
from ax.core.experiment import Experiment
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override


RESTRICTIVE_P_FEAS_THRESHOLD: float = 0.5
RESTRICTIVE_ARM_FRACTION: float = 0.9


def _format_constraint_label(
    outcome_constraint: OutcomeConstraint | ScalarizedOutcomeConstraint,
    constraint_name: str,
) -> str:
    """
    Format an outcome constraint as a string for display.

    Args:
        outcome_constraint: The outcome constraint to format.
        constraint_name: The constraint name (metric name or string representation).

    Returns:
        A formatted string like "capacity >= 3" or "latency <= 100".
    """
    if isinstance(outcome_constraint, ScalarizedOutcomeConstraint):
        # For scalarized constraints, use the string representation
        return str(outcome_constraint)

    # Format as "metric_name op bound" with optional % for relative constraints
    op_symbol = ">=" if outcome_constraint.op.name == "GEQ" else "<="
    bound_str = f"{outcome_constraint.bound}"

    # Add % suffix for relative constraints
    if outcome_constraint.relative:
        bound_str += "%"

    return f"{constraint_name} {op_symbol} {bound_str}"


@final
class IndividualConstraintsFeasibilityAnalysis(Analysis):
    """
    Analysis for checking the feasibility of individual constraints on the experiment.

    Unlike ConstraintsFeasibilityAnalysis which checks joint feasibility across all
    constraints, this analysis evaluates each constraint independently to identify
    which specific constraints are overly restrictive.

    A constraint is considered overly restrictive if at least half of the arms on the
    experiment have a probability of satisfying that constraint below the threshold
    (default: 0.5).
    """

    def __init__(
        self,
        restrictive_threshold: float = RESTRICTIVE_P_FEAS_THRESHOLD,
        fraction_arms_threshold: float = RESTRICTIVE_ARM_FRACTION,
    ) -> None:
        r"""
        Args:
            restrictive_threshold: The p(feasible) threshold below which an arm is
                considered to have difficulty satisfying a constraint. Default is 0.5.
            fraction_arms_threshold: The fraction of arms that must fall below
                restrictive_threshold for a constraint to be flagged as overly
                restrictive. Default is 0.9 (i.e., 90% of arms).
        """
        self.restrictive_threshold = restrictive_threshold
        self.fraction_arms_threshold = fraction_arms_threshold

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        IndividualConstraintsFeasibilityAnalysis requires an experiment. If the
        experiment has outcome constraints, it also requires an adapter that can
        predict the constraint metrics.
        """
        experiment_validation_str = validate_experiment(experiment=experiment)
        if experiment_validation_str is not None:
            return experiment_validation_str

        experiment = none_throws(experiment)

        # Just validate adapter. Edge cases like
        # No opt config / no constraints are handled in compute
        if (
            experiment.optimization_config is not None
            and experiment.optimization_config.outcome_constraints
        ):
            optimization_config = none_throws(experiment.optimization_config)
            constraint_metric_names = [
                outcome_constraint.metric.name
                for outcome_constraint in optimization_config.outcome_constraints
            ]

            adapter_can_predict_validation_str = validate_adapter_can_predict(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                required_metric_names=constraint_metric_names,
            )
            if adapter_can_predict_validation_str is not None:
                return adapter_can_predict_validation_str

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        r"""
        Compute the feasibility of individual constraints for the experiment.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.
            adapter: Adaptor to be used for predictions.

        Returns:
            A HealthcheckAnalysisCard object with information on overly restrictive
            constraints, i.e., constraints for which a significant fraction of arms
            have low probability of satisfaction.
        """
        experiment = none_throws(experiment)

        # If no optimization config or constraints, return early with PASS
        # Note, a bit of duplication here since validate_applicable_state already
        # does some checking on opt config
        if (
            experiment.optimization_config is None
            or not experiment.optimization_config.outcome_constraints
        ):
            subtitle = (
                "No optimization config is specified."
                if experiment.optimization_config is None
                else "No constraints are specified."
            )
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Ax Individual Constraints Feasibility Success",
                subtitle=subtitle,
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        optimization_config = experiment.optimization_config

        # adapter validated in validate_applicable_state
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        arm_data = prepare_arm_data(
            experiment=experiment,
            metric_names=[*optimization_config.metrics.keys()],
            use_model_predictions=True,
            adapter=relevant_adapter,
            trial_statuses=[TrialStatus.COMPLETED, TrialStatus.RUNNING],
            compute_p_feasible_per_constraint=True,
        )

        # Analyze each constraint individually
        restrictive_constraints = []
        constraint_stats = []

        # Get individual constraint p_feasible columns
        p_feasible_cols = [
            col
            for col in arm_data.columns
            if col.startswith("p_feasible_")
            and col not in ["p_feasible_mean", "p_feasible_sem"]
        ]

        # Create a mapping from metric names to outcome constraints for formatting
        constraint_map = {}
        for oc in optimization_config.outcome_constraints:
            if isinstance(oc, ScalarizedOutcomeConstraint):
                constraint_map[str(oc)] = oc
            else:
                constraint_map[oc.metric.name] = oc

        for col in p_feasible_cols:
            constraint_name = col.replace("p_feasible_", "")

            # Get the formatted constraint label
            if constraint_name in constraint_map:
                formatted_constraint = _format_constraint_label(
                    constraint_map[constraint_name], constraint_name
                )
            else:
                formatted_constraint = constraint_name

            # Count how many arms fall below the threshold
            num_arms = len(arm_data)
            num_below_threshold = (arm_data[col] < self.restrictive_threshold).sum()
            fraction_below = num_below_threshold / num_arms if num_arms > 0 else 0

            constraint_stats.append(
                {
                    "constraint": formatted_constraint,
                    "num_arms_below_threshold": num_below_threshold,
                    "total_arms": num_arms,
                    "fraction_below_threshold": fraction_below,
                }
            )

            # Check if this constraint is overly restrictive
            if fraction_below >= self.fraction_arms_threshold:
                restrictive_constraints.append(formatted_constraint)

        df = pd.DataFrame(constraint_stats)

        if restrictive_constraints:
            status = HealthcheckStatus.WARNING
            num_restrictive = len(restrictive_constraints)
            constraint_list = ", ".join(
                [f"<b>{c}</b>" for c in restrictive_constraints]
            )

            if num_restrictive == 1:
                subtitle = (
                    f"Found 1 overly restrictive constraint: {constraint_list}. "
                    "For this constraint, at least "
                    f"{self.fraction_arms_threshold * 100:.0f}% "
                    "of arms have a probability of satisfaction below "
                    f"{self.restrictive_threshold}. Consider relaxing the bounds for "
                    "this constraint to improve optimization performance."
                )
            else:
                subtitle = (
                    f"Found {num_restrictive} overly restrictive constraints: "
                    f"{constraint_list}. "
                    "For these constraints, at least "
                    f" {self.fraction_arms_threshold * 100:.0f}% "
                    "of arms have a probability of satisfaction below "
                    f"{self.restrictive_threshold}. Consider relaxing the bounds for "
                    "these constraints to improve optimization performance."
                )
            title_status = "Warning"
        else:
            subtitle = "All constraints are individually feasible."
            title_status = "Success"
            status = HealthcheckStatus.PASS

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=f"Ax Individual Constraints Feasibility {title_status}",
            subtitle=subtitle,
            df=df,
            status=status,
        )
