# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any

import numpy as np

import pandas as pd
from ax.analysis.plotly.utils import (
    format_constraint_violated_probabilities,
    get_constraint_violated_probabilities,
)
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import TParameterization
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.prediction_utils import predict_at_point
from plotly import express as px, graph_objects as go
from pyre_extensions import none_throws


def format_parameters_for_effects_by_arm_plot(
    parameters: TParameterization, max_num_params: int = 5
) -> str:
    """Format the parameters for tooltips in the predicted or insample
    effects plot."""
    parameter_items = list(parameters.items())[:max_num_params]
    string = "<br />  " + "<br />  ".join([f"{k}: {v}" for k, v in parameter_items])
    if len(parameter_items) < len(parameters):
        string += "<br />  ..."
    return string


def prepare_arm_effects_plot(
    df: pd.DataFrame, metric_name: str, outcome_constraints: list[OutcomeConstraint]
) -> go.Figure:
    """Prepare a plotly figure for the predicted effects based on the data in df.

    Args:
        metric_name: The name of the metric to plot.
        outcome_constraints: The outcome constraints for the experiment used to
            determine if the metric is a constraint, and if so, what the bound is
            so the bound can be rendered in the plot.
        df: A dataframe of data to plot with the following columns:
            - source: In-sample or model key that geneerated the candidate
            - arm_name: The name of the arm
            - mean: The observed or predicted mean of the metric specified
            - sem: The observed or predicted sem of the metric specified
            - error_margin: The 95% CI of the metric specified for the arm
            - size_column: The size of the circle in the plot, which represents
                the probability that the arm is feasible (does not violate any
                constraints).
            - parameters: A string representation of the parameters for the arm
                to be viewed in the tooltip.
            - constraints_violated: A string representation of the probability
                each constraint is violated for the arm, to be viewed in the tooltip.
    """
    fig = px.scatter(
        df,
        x="arm_name",
        y="mean",
        error_y="error_margin",
        color="source",
        # TODO: can we format this by callable or string template?
        hover_data=_get_parameter_columns(df),
        # avoid red because it will match the constraint violation indicator
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    dot_size = 8
    # set all dots to size 8 in plots
    fig.update_traces(marker={"line": {"width": 2}, "size": dot_size})

    # Manually create each constraint violation indicator
    # as a red outline around the dot, with alpha based on the
    # probability of constraint violation.
    for trace in fig.data:
        # there is a trace per source, so get the rows of df
        # pertaining to this trace
        indices = df["source"] == trace.name
        trace.marker.line.color = [
            # raising the alpha to a power < 1 makes the colors more
            # visible when there is a lower chance of constraint violation
            f"rgba(255, 0, 0, {(alpha) ** .75})"
            for alpha in df.loc[indices, "overall_probability_constraints_violated"]
            if not np.isnan(alpha)
        ]
        # Create a separate trace for the legend, otherwise the legend
        # will have the constraint violation indicator of the first arm
        # in the source group
        legend_trace = go.Scatter(
            # (None, None) is a hack to get a legend item without
            # appearing on the plot
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": dot_size,
                "color": trace.marker.color,
            },
            name=trace.name,
        )
        fig.add_trace(legend_trace)
        trace.showlegend = False

    # Add an item to the legend for the constraint violation indicator
    if df["overall_probability_constraints_violated"].notna().any():
        legend_trace = go.Scatter(
            # (None, None) is a hack to get a legend item without
            # appearing on the plot
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": dot_size,
                "color": "white",
                "line": {"width": 2, "color": "red"},
            },
            name="Constraint Violation",
        )
        fig.add_trace(legend_trace)

    _add_style_to_effects_by_arm_plot(
        fig=fig, df=df, metric_name=metric_name, outcome_constraints=outcome_constraints
    )
    return fig


def _get_parameter_columns(df: pd.DataFrame) -> dict[str, bool]:
    """Get the names of the columns that represent parameters in df."""
    return {
        col: (col not in ["source", "error_margin", "size_column"])
        for col in df.columns
    }


def _add_style_to_effects_by_arm_plot(
    fig: go.Figure,
    df: pd.DataFrame,
    metric_name: str,
    outcome_constraints: list[OutcomeConstraint],
) -> None:
    """Add style to a plotly figure for predicted or insample effects.

    - If we have a status quo, we add a solid red line at the status quo mean.
    - If the metric is a constraint, we add a dashed red line at the constraint
        bound.
    - Make the x-axis (arm name) tick angle 45 degrees.
    """
    if "status_quo" in df["arm_name"].values:
        fig.add_hline(
            y=df[df["arm_name"] == "status_quo"]["mean"].iloc[0],
            line_width=1,
            line_color="red",
            showlegend=True,
            name="Status Quo Mean",
        )
        # Add the status quo mean to the legend
        fig.add_trace(
            go.Scatter(
                # (None, None) is a hack to get a legend item without
                # appearing on the plot
                x=[None],
                y=[None],
                mode="lines",
                line={"color": "red", "width": 1},
                name="Status Quo Mean",
            )
        )
    for constraint in outcome_constraints:
        if constraint.metric.name == metric_name:
            assert not constraint.relative
            fig.add_hline(
                y=constraint.bound,
                line_width=1,
                line_color="red",
                line_dash="dash",
            )
            # Add the constraint bound to the legend
            fig.add_trace(
                go.Scatter(
                    # (None, None) is a hack to get a legend item without
                    # appearing on the plot
                    x=[None],
                    y=[None],
                    mode="lines",
                    line={"color": "red", "width": 1, "dash": "dash"},
                    name="Constraint Bound",
                )
            )
    fig.update_layout(
        xaxis={
            "tickangle": 45,
        },
        legend={
            "title": None,
        },
    )


def _get_trial_index_for_predictions(model: ModelBridge) -> int | None:
    """Returns status quo features index if defined on the model.  Otherwise, returns
    the max observed trial index to appease multitask models for prediction
    by giving fixed features. The max index is not necessarily accurate and should
    eventually come from the generation strategy, but at least gives consistent
    predictions accross trials.
    """
    if model.status_quo is None:
        observed_trial_indices = [
            obs.features.trial_index
            for obs in model.get_training_data()
            if obs.features.trial_index is not None
        ]
        if len(observed_trial_indices) == 0:
            return None
        return max(observed_trial_indices)

    return model.status_quo.features.trial_index


def get_predictions_by_arm(
    model: ModelBridge,
    metric_name: str,
    outcome_constraints: list[OutcomeConstraint],
    gr: GeneratorRun | None = None,
    abandoned_arms: set[str] | None = None,
) -> list[dict[str, Any]]:
    trial_index = _get_trial_index_for_predictions(model)
    if gr is None:
        if abandoned_arms:
            raise UserInputError(
                "`abandoned_arms` should only be specified if a generator run is "
                "provided."
            )
        observations = model.get_training_data()
        features = [o.features for o in observations]
        arm_names = [o.arm_name for o in observations]
        for feature in features:
            feature.trial_index = trial_index
    else:
        abandoned_arms = set() if abandoned_arms is None else abandoned_arms
        features = [
            ObservationFeatures(parameters=arm.parameters, trial_index=trial_index)
            for arm in gr.arms
            if arm.name not in abandoned_arms
        ]
        arm_names = [a.name for a in gr.arms if a.name not in abandoned_arms]
    try:
        predictions = [
            predict_at_point(
                model=model,
                obsf=obsf,
                metric_names={metric_name}.union(
                    {constraint.metric.name for constraint in outcome_constraints}
                ),
            )
            for obsf in features
        ]
    except NotImplementedError:
        raise UserInputError(
            "This plot requires a GenerationStrategy which is "
            "in a state where the current model supports prediction.  The current "
            f"model is {model._model_key} and does not support prediction."
        )
    constraints_violated_by_constraint = get_constraint_violated_probabilities(
        predictions=predictions,
        outcome_constraints=outcome_constraints,
    )
    probabilities_not_feasible = constraints_violated_by_constraint.pop(
        "any_constraint_violated"
    )
    constraints_violated = [
        {
            c: constraints_violated_by_constraint[c][i]
            for c in constraints_violated_by_constraint
        }
        for i in range(len(features))
    ]

    for i in range(len(features)):
        if (
            model.status_quo is not None
            and features[i].parameters
            == none_throws(model.status_quo).features.parameters
        ):
            probabilities_not_feasible[i] = 0
            constraints_violated[i] = {}
    return [
        {
            "source": "In-sample" if gr is None else gr._model_key,
            "arm_name": arm_names[i],
            "mean": predictions[i][0][metric_name],
            "sem": predictions[i][1][metric_name],
            "error_margin": 1.96 * predictions[i][1][metric_name],
            "constraints_violated": format_constraint_violated_probabilities(
                constraints_violated[i]
            ),
            # used for constraint violation indicator
            "overall_probability_constraints_violated": round(
                probabilities_not_feasible[i], ndigits=2
            ),
            "parameters": format_parameters_for_effects_by_arm_plot(
                parameters=features[i].parameters
            ),
        }
        for i in range(len(features))
    ]
