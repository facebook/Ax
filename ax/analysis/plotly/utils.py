# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import re
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.base import Adapter

from botorch.utils.probability.utils import compute_log_prob_feas_from_bounds
from numpy.typing import NDArray
from plotly import express as px

MAX_LABEL_LEN: int = 50

# Because normal distributions have long tails, every arm has a non-zero
# probability of violating the constraint. But below a certain threshold, we
# consider probability of violation to be negligible.
MINIMUM_CONTRAINT_VIOLATION_THRESHOLD = 0.01

# Plotting style constants
CONFIDENCE_INTERVAL_BLUE = "rgba(0, 0, 255, 0.2)"
MARKER_BLUE = "rgba(0, 0, 255, 0.3)"  # slightly more opaque than the CI blue
CANDIDATE_RED = "rgba(220, 20, 60, 0.3)"
CANDIDATE_CI_RED = "rgba(220, 20, 60, 0.2)"

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

# Use the same continuous sequential color scale for all plots. Plasma uses purples for
# low values and transitions to yellows for high values.
METRIC_CONTINUOUS_COLOR_SCALE: list[str] = px.colors.sequential.Plasma


# Use a consistent color for each TrialStatus name, sourced from
# the default Plotly color palette. See https://plotly.com/python/discrete-color/
# for more details and swatches.
TRIAL_STATUS_TO_PLOTLY_COLOR: dict[str, str] = {
    TrialStatus.CANDIDATE.name: px.colors.qualitative.Plotly[8],  # Pink
    TrialStatus.STAGED.name: px.colors.qualitative.Plotly[3],  # Purple
    TrialStatus.FAILED.name: px.colors.qualitative.Plotly[4],  # Orange
    TrialStatus.COMPLETED.name: px.colors.qualitative.Plotly[0],  # Blue
    TrialStatus.RUNNING.name: px.colors.qualitative.Plotly[2],  # Green
    TrialStatus.ABANDONED.name: px.colors.qualitative.Plotly[1],  # Red
    TrialStatus.EARLY_STOPPED.name: px.colors.qualitative.Plotly[5],  # Teal
}

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


def trial_status_to_plotly_color(
    trial_status: str,
    ci_transparency: bool = False,
) -> str:
    """
    Standardize the colors which correspond to a TrialStatus name across the Plotly
    analyses.

    Always use transparency for CI colors to improve legibility.
    """
    hex_color = TRIAL_STATUS_TO_PLOTLY_COLOR.get(
        trial_status,
        # Default to pink, treating unknown trial status as CANDIDATE
        px.colors.qualitative.Plotly[8],
    )

    return get_scatter_point_color(hex_color=hex_color, ci_transparency=ci_transparency)


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
            + f"±{1.96 * row[f'{metric_name}_sem']:.5f}"
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
    """Check if a adapter is predictive.  Basically, we're checking if
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
