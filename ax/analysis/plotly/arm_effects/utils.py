# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

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
        size="size_column",
        size_max=10,
    )
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
    fig.update_layout(
        xaxis={
            "tickangle": 45,
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
) -> list[dict[str, Any]]:
    trial_index = _get_trial_index_for_predictions(model)
    if gr is None:
        observations = model.get_training_data()
        features = [o.features for o in observations]
        arm_names = [o.arm_name for o in observations]
        for feature in features:
            feature.trial_index = trial_index
    else:
        features = [
            ObservationFeatures(parameters=arm.parameters, trial_index=trial_index)
            for arm in gr.arms
        ]
        arm_names = [a.name for a in gr.arms]
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
            "size_column": 100 - probabilities_not_feasible[i] * 100,
            "parameters": format_parameters_for_effects_by_arm_plot(
                parameters=features[i].parameters
            ),
        }
        for i in range(len(features))
    ]
