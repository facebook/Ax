# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws
from torch import Tensor


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def generic_equals(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor):
        return isinstance(second, Tensor) and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(
            first, second, equal_nan=True
        )
    if isinstance(first, dict):
        return isinstance(second, dict) and generic_equals(
            sorted(first.items()), sorted(second.items())
        )
    if isinstance(first, (tuple, list)):
        if type(first) is not type(second) or len(first) != len(second):
            return False
        for f, s in zip(first, second):
            if not generic_equals(f, s):
                return False
        return True
    return first == second


def run_trials_with_gs(
    experiment: Experiment, gs: GenerationStrategy, num_trials: int
) -> None:
    r"""Runs and completes `num_trials` trials for the given experiment with the
    given GS. The trials are completed with random metric values between 0 and 1.

    Args:
        experiment: The experiment to run trials on. Must have an optimization config.
        gs: The generation strategy to use.
        num_trials: The number of trials to run.
    """
    if experiment.optimization_config is None:
        raise UnsupportedError(  # pragma: no cover
            "`run_trials_with_gs` requires the experiment to have "
            "an optimization config."
        )
    existing_trials = len(experiment.trials)
    for i in range(existing_trials, existing_trials + num_trials):
        trial = experiment.new_trial(generator_run=gs.gen(experiment=experiment))
        data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": arm.name,
                        "metric_name": m,
                        "mean": random.random(),
                        "sem": None,
                        "trial_index": i,
                    }
                    for m in none_throws(experiment.optimization_config).metrics
                    for arm in trial.arms
                ]
            )
        )
        experiment.attach_data(data)
        trial.run().complete()
