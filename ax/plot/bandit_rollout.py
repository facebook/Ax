#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Dict, List

from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.color import MIXED_SCALE, rgba


def plot_bandit_rollout(experiment: Experiment) -> AxPlotConfig:
    """Plot bandit rollout from ane experiement."""

    categories: List[str] = []
    arms: Dict[str, Dict[str, Any]] = {}

    data = []

    index = 0
    for trial in sorted(experiment.trials.values(), key=lambda trial: trial.index):
        if not isinstance(trial, BatchTrial):
            raise ValueError(
                "Bandit rollout graph is not supported for BaseTrial."
            )  # pragma: no cover

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
            arms[arm.name]["text"].append("{:.2f}%".format(weight))

    for key in arms.keys():
        data.append(arms[key])

    # pyre-fixme[6]: Expected `typing.Tuple[...g.Tuple[int, int, int]`.
    colors = [rgba(c) for c in MIXED_SCALE]
    config = {"data": data, "categories": categories, "colors": colors}

    return AxPlotConfig(config, plot_type=AxPlotTypes.BANDIT_ROLLOUT)
