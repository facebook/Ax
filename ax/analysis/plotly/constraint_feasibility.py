# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from logging import Logger
from typing import Any, final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import BOTORCH_COLOR_SCALE
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.utils import truncate_label
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
    validate_outcome_constraints,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from plotly import graph_objects as go
from pyre_extensions import none_throws, override

logger: Logger = get_logger(__name__)


@final
class ConstraintFeasibilityPlot(Analysis):
    """
    Plotly parallel coordinates plot showing the probability of satisfying each
    individual constraint, with one line per arm and dimensions for each constraint
    in the optimization config. This plot is useful for understanding which constraints
    are most difficult to satisfy and how arms perform across different constraints.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index during which the arm was run
        - arm_name: The name of the arm
        - trial_status: The status of the trial
        - generation_node: The name of the GenerationNode that generated the arm
        - overall p(feasible): The joint probability that the arm satisfies
            all constraints on the Ax Experiment
        - p_feasible_{CONSTRAINT_NAME}: The probability that the arm satisfies the
          constraint, for each constraint
    """

    def __init__(
        self,
        use_model_predictions: bool = True,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
    ) -> None:
        """
        Args:
            use_model_predictions: Whether to use model predictions or raw data.
            trial_index: If specified, only include arms from this trial.
            trial_statuses: If specified, only include arms from trials with these
                statuses.
            additional_arms: Additional arms to include in the plot (requires
                use_model_predictions=True).
        """
        self.use_model_predictions = use_model_predictions
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        ConstraintFeasibilityPlot requires an experiment with trials,
        data, and at least one outcome constraint.
        """
        experiment_validation_str = validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=True,
        )

        if experiment_validation_str is not None:
            return experiment_validation_str

        experiment = none_throws(experiment)

        # Validate outcome constraints for feasibility calculation
        outcome_constraints_validation_str = validate_outcome_constraints(
            experiment=experiment,
        )
        if outcome_constraints_validation_str is not None:
            return outcome_constraints_validation_str

        outcome_constraint_metrics = [
            outcome_constraint.metric.name
            for outcome_constraint in none_throws(
                experiment.optimization_config
            ).outcome_constraints
        ]

        # Ensure that we either can predict the outcome constraint metrics or that we
        # have observations for them.
        if self.use_model_predictions:
            adapter_can_predict_validation_str = validate_adapter_can_predict(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                required_metric_names=outcome_constraint_metrics,
            )
            if adapter_can_predict_validation_str is not None:
                return adapter_can_predict_validation_str
        else:
            if self.additional_arms is not None:
                return (
                    "Cannot provide additional arms when use_model_predictions=False."
                )

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        experiment = none_throws(experiment)
        optimization_config = none_throws(experiment.optimization_config)

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
        else:
            relevant_adapter = None

        # Collect all metrics and compute
        constraint_metric_names = [
            outcome_constraint.metric.name
            for outcome_constraint in optimization_config.outcome_constraints
        ]
        objective_metric_names = [
            metric.name for metric in optimization_config.objective.metrics
        ]
        all_metric_names = list(set(constraint_metric_names + objective_metric_names))

        arm_data = prepare_arm_data(
            experiment=experiment,
            metric_names=all_metric_names,
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=False,
            compute_p_feasible_per_constraint=True,
        )

        constraint_names = _get_constraint_names(
            outcome_constraints=optimization_config.outcome_constraints
        )

        fig = _prepare_plot(
            df=arm_data,
            outcome_constraints=optimization_config.outcome_constraints,
            constraint_names=constraint_names,
        )

        # Add p_feasible columns to df
        output_columns = ["trial_index", "arm_name", "trial_status", "generation_node"]
        output_df = arm_data.loc[:, output_columns].copy()
        output_df["overall p(feasible)"] = arm_data["p_feasible_mean"]
        p_feasible_columns = [
            col for col in arm_data.columns if col.startswith("p_feasible_")
        ]
        for col in p_feasible_columns:
            output_df[col] = arm_data[col]

        # Add objective metrics to df
        objective_metric_columns = [
            col
            for col in arm_data.columns
            if any(
                col.startswith(f"{metric_name}_")
                for metric_name in objective_metric_names
            )
        ]
        for col in objective_metric_columns:
            output_df[col] = arm_data[col]

        subtitle = (
            "The parallel coordinates plot displays the probability of satisfying "
            "each individual constraint along with the joint probability of satisfying "
            "all constraints (last dimension). Each line represents an arm, "
            "and each dimension represents a constraint, with the overall probability "
            "shown last. This visualization helps identify which constraints are "
            "most difficult to satisfy and how different arms perform across the "
            "constraint landscape. Lines colored by trial_index."
        )

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=(
                ("Predicted" if self.use_model_predictions else "Observed")
                + " Constraint Feasibility by Constraint"
            ),
            subtitle=subtitle,
            df=output_df,
            fig=fig,
        )


def _get_constraint_names(
    outcome_constraints: Sequence[Any],
) -> list[str]:
    """
    Extract constraint names from outcome constraints.

    Args:
        outcome_constraints: The outcome constraints from the optimization config.

    Returns:
        A list of constraint names.
    """
    constraint_names = []
    for oc in outcome_constraints:
        if isinstance(oc, ScalarizedOutcomeConstraint):
            constraint_names.append(str(oc))
        else:
            constraint_names.append(oc.metric.name)
    return constraint_names


def _format_constraint_label(
    outcome_constraint: Any,
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

    # Format label, handling relative constraitns as needed
    op_symbol = ">=" if outcome_constraint.op.name == "GEQ" else "<="
    bound_str = f"{outcome_constraint.bound}"
    if outcome_constraint.relative:
        bound_str += "%"

    return f"{constraint_name} {op_symbol} {bound_str}"


def _prepare_plot(
    df: pd.DataFrame,
    outcome_constraints: Sequence[Any],
    constraint_names: list[str],
) -> go.Figure:
    """
    Prepare a parallel coordinates plot showing p_feasible for each constraint.

    Args:
        df: DataFrame from prepare_arm_data with p_feasible_{constraint_name} columns.
        outcome_constraints: The outcome constraints from the optimization config.
        constraint_names: List of constraint names (in order).

    Returns:
        A Plotly figure.
    """

    color_values = df["trial_index"].tolist()
    dimensions = []

    # Add p_feasible dimensions for each constraint
    for oc, constraint_name in zip(outcome_constraints, constraint_names):
        col_name = f"p_feasible_{constraint_name}"
        if col_name in df.columns:
            constraint_label = _format_constraint_label(oc, constraint_name)
            dimensions.append(
                {
                    "label": truncate_label(label=constraint_label),
                    "values": df[col_name].tolist(),
                    "range": [0, 1],
                }
            )

    # Add overall p_feasible as the last dimension
    if "p_feasible_mean" in df.columns:
        dimensions.append(
            {
                "label": "Overall p(feasible)",
                "values": df["p_feasible_mean"].tolist(),
                "range": [0, 1],
            }
        )

    colorscale = BOTORCH_COLOR_SCALE

    line_config = {
        "color": color_values,
        "colorscale": colorscale,
        "showscale": True,
        "colorbar": {
            "title": "trial_index",
            "dtick": 1,
        },
    }

    fig = go.Figure(
        go.Parcoords(
            line=line_config,
            dimensions=dimensions,
            labelangle=-25,
        )
    )

    fig.update_layout(
        annotations=[
            {
                "text": "Experiment Constraints",
                "x": 0.5,
                "y": -0.15,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 14},
            },
            {
                "text": "Probability of Constraint Satisfaction",
                "x": -0.05,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "middle",
                "showarrow": False,
                "textangle": -90,
                "font": {"size": 14},
            },
        ],
        margin={"l": 120, "b": 80, "t": 150},
    )

    return fig
