# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import re
from typing import Sequence, Union

import pandas as pd
from ax.analysis.plotly.color_constants import BOTORCH_COLOR_SCALE, LIGHT_AX_BLUE
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UnsupportedError
from plotly import express as px

MAX_LABEL_LEN: int = 50

# Because normal distributions have long tails, every arm has a non-zero
# probability of violating the constraint. But below a certain threshold, we
# consider probability of violation to be negligible.
MINIMUM_CONTRAINT_VIOLATION_THRESHOLD = 0.01

# Z-score for 95% confidence interval
Z_SCORE_95_CI = 1.96

STALE_FAIL_REASON = "Newer candidates generated."

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


# Move the legened to the bottom, and make horizontal
LEGEND_POSITION: dict[str, Union[float, str]] = {
    "orientation": "h",
    "yanchor": "top",
    "y": -0.2,
    "xanchor": "center",
    "x": 0.5,
    "title_text": "",  # remove title
}

# The Base y-offset (in normalized coordinates) to place the legend
# below the plot area.
LEGEND_BASE_OFFSET: float = -0.1

# This scaling factor controls how much additional space is added
# based on the max tick label length
X_TICKER_SCALING_FACTOR: int = 40

MARGIN_REDUCUTION: dict[str, int] = {"t": 50}

# Always use the same transparency factor for CI colors to improve legibility when many
# scatter points are plotted on the same plot.
CI_ALPHA: float = 0.5

# The max length of a hover label to prevent overflow making the hover unreadable
MAX_HOVER_LABEL_LEN: int = 300

SINGLE_CANDIDATE_TRIAL_LEGEND: str = "Candidate Trial"
MULTIPLE_CANDIDATE_TRIALS_LEGEND: str = "Candidate Trials"


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
    trials_list: list[int],
    trial_index: int,
    transparent: bool,
) -> str:
    """
    Determines the color for a trial based on its index and status.

    If the trial is a candidate, it returns LIGHT_AX_BLUE. Otherwise,
    it calculates a color from the BOTORCH_COLOR_SCALE based on the trial's
    normalized index (relative to all completed trials).

    Note, we are calculating normalized_index here by using the length of the
    trials list and the index of each trial_index in that list rather than the
    trial_index associated with each trial. This is done to ensure trial colors
    are evenly spaced out, even in cases where there are many FAILED trials.
    """
    max_trial_index = len(trials_list) - 1

    if trial_df["trial_status"].iloc[0] == TrialStatus.CANDIDATE.name:
        return get_scatter_point_color(
            hex_color=LIGHT_AX_BLUE, ci_transparency=transparent
        )

    adj_trial_index = trials_list.index(trial_index)
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
    tooltip_strs = []
    trial_index = row["trial_index"]
    if trial_index != -1:
        # omit the trial tooltip for additional arms
        tooltip_strs.append(f"Trial: {trial_index}")

    tooltip_strs.append(f"Arm: {row['arm_name']}")
    tooltip_strs.append(f"Status: {row['trial_status']}")
    tooltip_strs.append(f"Generation Node: {row['generation_node']}")

    tooltip_strs.extend(
        [
            (
                (f"{metric_name}: {row[f'{metric_name}_mean']:.5f}")
                + f"±{Z_SCORE_95_CI * row[f'{metric_name}_sem']:.5f}"
                if not math.isnan(row[f"{metric_name}_sem"])
                else ""
            )
            for metric_name in metric_names
        ]
    )

    if row["p_feasible_mean"] < MINIMUM_CONTRAINT_VIOLATION_THRESHOLD:
        constraints_warning_str = "[Warning] This arm is likely infeasible"
    else:
        constraints_warning_str = ""
    tooltip_strs.append(constraints_warning_str)

    return "<br />".join(tooltip_strs)


def get_trial_trace_name(trial_index: int) -> str:
    """Get a trace name for a trial index."""
    return "Additional Arms" if trial_index == -1 else f"Trial {trial_index}"


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


def get_trial_statuses_with_fallback(
    trial_statuses: Sequence[TrialStatus] | None, trial_index: int | None
) -> list[TrialStatus] | None:
    """Get the default trial statuses to plot.

    By default, include all trials except those that are abandoned, stale, or failed.
    If trial_index is provided, then we only filter based on trial_index,
    and therefore this function returns None.
    """
    if trial_index is not None:
        return None
    elif trial_statuses is not None:
        return [*trial_statuses]
    return [
        *{*TrialStatus} - {TrialStatus.ABANDONED, TrialStatus.STALE, TrialStatus.FAILED}
    ]
