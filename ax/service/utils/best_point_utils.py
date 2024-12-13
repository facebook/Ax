#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment import Experiment
from pyre_extensions import none_throws

BASELINE_ARM_NAME = "baseline_arm"


def select_baseline_name_default_first_trial(
    experiment: Experiment, baseline_arm_name: str | None
) -> tuple[str, bool]:
    """
    Choose a baseline arm from arms on the experiment. Logic:
    1. If ``baseline_arm_name`` provided, validate that arm exists
       and return that arm name.
    2. If ``experiment.status_quo`` is set, return its arm name.
    3. If there is at least one trial on the experiment, use the
       first trial's first arm as the baseline.
    4. Error if 1-3 all don't apply.

    Returns:
        Tuple:
            baseline arm name (str)
            true when baseline selected from first arm of experiment (bool)
        raise ValueError if no valid baseline found
    """

    arms_dict = experiment.arms_by_name

    if baseline_arm_name:
        if baseline_arm_name not in arms_dict:
            raise ValueError(f"Arm by name {baseline_arm_name=} not found.")
        return baseline_arm_name, False

    if experiment.status_quo and none_throws(experiment.status_quo).name in arms_dict:
        baseline_arm_name = none_throws(experiment.status_quo).name
        return baseline_arm_name, False

    if (
        experiment.trials
        and experiment.trials[0].arms
        and experiment.trials[0].arms[0].name in arms_dict
    ):
        baseline_arm_name = experiment.trials[0].arms[0].name
        return baseline_arm_name, True

    else:
        raise ValueError("Could not find valid baseline arm.")
