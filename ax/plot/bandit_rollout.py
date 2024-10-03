#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import plotly.graph_objs as go
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import MIXED_SCALE, rgba


def plot_bandit_rollout(experiment: Experiment) -> AxPlotConfig:
    """Plot bandit rollout from ane experiement."""

    categories: list[str] = []
    arms: dict[str, dict[str, Any]] = {}

    data = []

    index = 0
    for trial in sorted(experiment.trials.values(), key=lambda trial: trial.index):
        if not isinstance(trial, BatchTrial):
            raise ValueError("Bandit rollout graph is not supported for BaseTrial.")

        category = f"Round {trial.index}"
        categories.append(category)

        for arm, weight in trial.normalized_arm_weights(total=100).items():
            if arm.name not in arms:
                arms[arm.name] = {
                    "index": index,
                    "name": arm.name,
                    "x": [],
                    "y": [],
                    "text": [],
                }
                index += 1

            arms[arm.name]["x"].append(category)
            arms[arm.name]["y"].append(weight)
            arms[arm.name]["text"].append(f"{weight:.2f}%")

    for key in arms.keys():
        data.append(arms[key])

    colors = [rgba(c) for c in MIXED_SCALE]

    layout = go.Layout(
        title="Rollout Process<br>Bandit Weight Graph",
        xaxis={
            "title": "Rounds",
            "zeroline": False,
            "categoryorder": "array",
            "categoryarray": categories,
        },
        yaxis={"title": "Percent", "showline": False},
        barmode="stack",
        showlegend=False,
        margin={"r": 40},
    )

    bandit_config = {"type": "bar", "hoverinfo": "name+text", "width": 0.5}

    bandits = [
        dict(bandit_config, marker={"color": colors[d["index"] % len(colors)]}, **d)
        for d in data
    ]
    for bandit in bandits:
        del bandit["index"]  # Have to delete index or figure creation causes error
    fig = go.Figure(data=bandits, layout=layout)

    # pyre-fixme[6]: For 1st argument expected `Dict[str, typing.Any]` but got `Figure`.
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)
