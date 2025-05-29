# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import re
from typing import Any, Sequence, Union

import numpy as np

import pandas as pd
import torch
from ax.adapter.base import Adapter
from ax.adapter.prediction_utils import predict_at_point
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from botorch.utils.probability.utils import compute_log_prob_feas_from_bounds
from numpy.typing import NDArray
from plotly import express as px, graph_objects as go
from pyre_extensions import none_throws

MAX_LABEL_LEN: int = 50

# Because normal distributions have long tails, every arm has a non-zero
# probability of violating the constraint. But below a certain threshold, we
# consider probability of violation to be negligible.
MINIMUM_CONTRAINT_VIOLATION_THRESHOLD = 0.01

# Z-score for 95% confidence interval
Z_SCORE_95_CI = 1.96

# Plotting style constants
CANDIDATE_RED = "rgba(220, 20, 60, 0.3)"
CANDIDATE_CI_RED = "rgba(220, 20, 60, 0.2)"
CONSTRAINT_VIOLATION_RED = "red"

# Colors sampled from Botorch flame logo
BOTORCH_COLOR_SCALE = [
    "#f7931e",  # Botorch orange
    "#eb882d",
    "#df7d3c",
    "#d3724b",
    "#c7685a",
    "#bb5d69",
    "#af5278",
    "#a34887",
    "#973d96",
    "#8b32a5",
    "#7f28b5",  # Botorch purple
    "#792fbb",
    "#7436c1",
    "#6f3dc7",
    "#6a44cd",
    "#654bd4",
    "#6052da",
    "#5b59e0",
    "#5660e6",
    "#5167ec",
    "#4c6ef3",  # Botorch blue
]
AX_BLUE = "#5078f9"  # rgb code: rgb(80, 120, 249)
LIGHT_AX_BLUE = "#adc0fd"  # rgb(173, 192, 253)


# Splat this into a go.Scatter initializer when drawing a line that represents the
# cummulative best, Pareto frontier, etc. for a unified look and feel.
BEST_LINE_SETTINGS: dict[str, str | dict[str, str] | bool] = {
    "mode": "lines",
    "line": {
        "color": px.colors.qualitative.Plotly[9],  # Gold
        "dash": "dash",
        # This gives us the "stepped" line effect we want
        "shape": "hv",
    },
    # Do not show this line in the legend or in hover tooltips.
    "showlegend": False,
    "hoverinfo": "skip",
}

# Use the same continuous sequential color scale for all plots. PRGn uses purples for
# low values and transitions to greens for high values.
METRIC_CONTINUOUS_COLOR_SCALE: list[str] = px.colors.colorbrewer.PRGn
COLOR_FOR_INCREASES: str = METRIC_CONTINUOUS_COLOR_SCALE[8]  # lighter green
COLOR_FOR_DECREASES: str = METRIC_CONTINUOUS_COLOR_SCALE[2]  # lighter purple

# Move the legened to the bottom, and make horizontal
LEGEND_POSITION: dict[str, Union[float, str]] = {
    "orientation": "h",
    "yanchor": "top",
    "y": -0.2,
    "xanchor": "center",
    "x": 0.5,
    "title_text": "",  # remove title
}

MARGIN_REDUCUTION: dict[str, int] = {"t": 50}

# Always use the same transparency factor for CI colors to improve legibility when many
# scatter points are plotted on the same plot.
CI_ALPHA: float = 0.5


def get_scatter_point_color(
    hex_color: str,
    ci_transparency: bool = False,
) -> str:
    """
    Convert a hex color (like those in px.colors.qualitative) to an rgba string.

    Always use transparency for CI colors to improve legibility.
    """
    red, green, blue = px.colors.hex_to_rgb(hex_color)
    alpha = CI_ALPHA if ci_transparency else 1

    return f"rgba({red}, {green}, {blue}, {alpha})"


def trial_index_to_color(
    trial_df: pd.DataFrame,
    completed_trials_list: list[int],
    trial_index: int,
    transparent: bool,
) -> str:
    """
    Determines the color for a trial based on its index and status.

    If the trial is a candidate, it returns LIGHT_AX_BLUE. Otherwise,
    it calculates a color from the BOTORCH_COLOR_SCALE based on the trial's
    normalized index (relative to all completed trials).

    """
    max_trial_index = len(completed_trials_list) - 1

    if trial_df["trial_status"].iloc[0] == TrialStatus.CANDIDATE.name:
        return get_scatter_point_color(
            hex_color=LIGHT_AX_BLUE, ci_transparency=transparent
        )

    adj_trial_index = completed_trials_list.index(trial_index)
    normalized_index = 0 if max_trial_index == 0 else adj_trial_index / max_trial_index
    color_index = int(normalized_index * (len(BOTORCH_COLOR_SCALE) - 1))
    hex_color = BOTORCH_COLOR_SCALE[color_index]
    return get_scatter_point_color(hex_color=hex_color, ci_transparency=transparent)


def get_arm_tooltip(
    row: pd.Series,
    metric_names: Sequence[str],
) -> str:
    """
    Given a row from ax.analysis.utils.prepare_arm_data return a tooltip. This should
    be used in every Plotly analysis where we source data from prepare_arm_data.
    """

    trial_str = f"Trial: {row['trial_index']}"
    arm_str = f"Arm: {row['arm_name']}"
    status_str = f"Status: {row['trial_status']}"
    generation_node_str = f"Generation Node: {row['generation_node']}"

    metric_strs = [
        (
            (f"{metric_name}: {row[f'{metric_name}_mean']:.5f}")
            + f"±{Z_SCORE_95_CI * row[f'{metric_name}_sem']:.5f}"
            if not math.isnan(row[f"{metric_name}_sem"])
            else ""
        )
        for metric_name in metric_names
    ]

    if row["p_feasible"] < MINIMUM_CONTRAINT_VIOLATION_THRESHOLD:
        constraints_warning_str = "[Warning] This arm is likely infeasible"
    else:
        constraints_warning_str = ""

    return "<br />".join(
        [
            trial_str,
            arm_str,
            status_str,
            generation_node_str,
            *metric_strs,
            constraints_warning_str,
        ]
    )


def truncate_label(label: str, n: int = MAX_LABEL_LEN) -> str:
    if len(label) <= n:
        return label
    # Remove suffix with uppercase letters and underscores
    label = re.sub(r"_([A-Z_]+)$", "", label)
    if len(label) <= n:
        return label
    # Remove prefix up to the first colon
    label = re.sub(r"^[^:]*:", "", label)
    if len(label) <= n:
        return label
    # Filter out empty segments and those that seem too generic
    segments = re.split(r"[:]", label)
    filtered_segments = [
        s for s in segments if s and s.lower() not in ["overall", "v2"]
    ]
    # Build the shortened label by adding segments from the end
    shortened_label = ""
    for segment in reversed(filtered_segments):
        # Check if a single segment is longer than max_length
        if len(segment) > n:
            return segment[: n - 3] + "..."
        # Check if adding the next segment would exceed the max length
        if len(segment) + (1 if shortened_label else 0) + len(shortened_label) <= n:
            if shortened_label:
                shortened_label = ":" + shortened_label
            shortened_label = segment + shortened_label
        else:
            break
    return shortened_label


def get_constraint_violated_probabilities(
    predictions: list[tuple[dict[str, float], dict[str, float]]],
    outcome_constraints: list[OutcomeConstraint],
) -> dict[str, list[float]]:
    """Get the probability that each arm violates the outcome constraints.

    Args:
        predictions: List of predictions for each observation feature
            generated by predict_at_point.  It should include predictions
            for all outcome constraint metrics.
        outcome_constraints: List of outcome constraints to check.

    Returns:
        A dict of probabilities that each arm violates the outcome
        constraint provided, and for "any_constraint_violated" the probability that
        the arm violates *any* outcome constraint provided.
    """
    if len(outcome_constraints) == 0:
        return {"any_constraint_violated": [0.0] * len(predictions)}
    if any(constraint.relative for constraint in outcome_constraints):
        raise UserInputError(
            "`get_constraint_violated_probabilities()` does not support relative "
            "outcome constraints. Use `Derelativize().transform_optimization_config()` "
            "before passing constraints to this method."
        )

    metrics = [constraint.metric.name for constraint in outcome_constraints]
    means = torch.as_tensor(
        [
            [prediction[0][metric_name] for metric_name in metrics]
            for prediction in predictions
        ]
    )
    sigmas = torch.as_tensor(
        [
            [prediction[1][metric_name] for metric_name in metrics]
            for prediction in predictions
        ]
    )
    feasibility_probabilities: dict[str, NDArray] = {}
    for constraint in outcome_constraints:
        if constraint.op == ComparisonOp.GEQ:
            con_lower_inds = torch.tensor([metrics.index(constraint.metric.name)])
            con_lower = torch.tensor([constraint.bound])
            con_upper_inds = torch.as_tensor([])
            con_upper = torch.as_tensor([])
        else:
            con_lower_inds = torch.as_tensor([])
            con_lower = torch.as_tensor([])
            con_upper_inds = torch.tensor([metrics.index(constraint.metric.name)])
            con_upper = torch.tensor([constraint.bound])

        feasibility_probabilities[constraint.metric.name] = (
            compute_log_prob_feas_from_bounds(
                means=means,
                sigmas=sigmas,
                con_lower_inds=con_lower_inds,
                con_upper_inds=con_upper_inds,
                con_lower=con_lower,
                con_upper=con_upper,
                # "both" can also be expressed by 2 separate constraints...
                con_both_inds=torch.as_tensor([]),
                con_both=torch.as_tensor([]),
            )
            .exp()
            .numpy()
        )

    feasibility_probabilities["any_constraint_violated"] = np.prod(
        list(feasibility_probabilities.values()), axis=0
    )

    return {
        metric_name: (1 - feasibility_probabilities[metric_name]).tolist()
        for metric_name in feasibility_probabilities
    }


def format_constraint_violated_probabilities(
    constraints_violated: dict[str, float],
) -> str:
    """Format the constraints violated for the tooltip."""
    max_metric_length = 70
    constraints_violated = {
        k: v
        for k, v in constraints_violated.items()
        if v > MINIMUM_CONTRAINT_VIOLATION_THRESHOLD
    }
    constraints_violated_str = "<br />  ".join(
        [
            (
                f"{k[:max_metric_length]}{'...' if len(k) > max_metric_length else ''}"
                f": {v * 100:.1f}% chance violated"
            )
            for k, v in constraints_violated.items()
        ]
    )
    if len(constraints_violated_str) == 0:
        return "No constraints violated"
    else:
        constraints_violated_str = "<br />  " + constraints_violated_str

    return constraints_violated_str


def get_nudge_value(
    metric_name: str,
    experiment: Experiment | None = None,
    use_modeled_effects: bool = False,
) -> int:
    """Get the amount to nudge the level of the plot. Deteremined by metric
    importance and whether modeled effects are used.
    """
    # without an experiment or optimization config, we can't tell if this plot is
    # relatively more important
    if experiment is None or experiment.optimization_config is None:
        return 0

    nudge = 0
    # More important metrics have a higher nudge
    if metric_name in experiment.optimization_config.objective.metric_names:
        nudge += 2
    elif metric_name in experiment.optimization_config.metrics:
        nudge += 1

    # Relevant for plots where observed effects and modeled effects can both be shown
    if use_modeled_effects:
        nudge += 1

    return nudge


def is_predictive(adapter: Adapter) -> bool:
    # TODO: Improve this logic and move it to base adapter class
    """Check if an adapter is predictive.  Basically, we're checking if
    predict() is implemented.

    NOTE: This does not mean it's capable of out of sample prediction.
    """
    try:
        adapter.predict(observation_features=[])
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def select_metric(experiment: Experiment) -> str:
    """Select the most relevant metric to plot from an Experiment."""
    if experiment.optimization_config is None:
        raise ValueError(
            "Cannot infer metric to plot from Experiment without OptimizationConfig"
        )
    objective = experiment.optimization_config.objective
    if isinstance(objective, MultiObjective):
        raise UnsupportedError(
            "Cannot infer metric to plot from MultiObjective, please "
            "specify a metric"
        )
    if isinstance(objective, ScalarizedObjective):
        raise UnsupportedError(
            "Cannot infer metric to plot from ScalarizedObjective, please "
            "specify a metric"
        )
    return experiment.optimization_config.objective.metric.name


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
            - source: In-sample or model key that generated the candidate
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
                "line": {"width": 2, "color": CONSTRAINT_VIOLATION_RED},
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


def _get_trial_index_for_predictions(model: Adapter) -> int | None:
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
    model: Adapter,
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
            "error_margin": Z_SCORE_95_CI * predictions[i][1][metric_name],
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
