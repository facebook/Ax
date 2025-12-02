#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment import Experiment
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy


def get_default_ess_or_none(
    experiment: Experiment,
) -> PercentileEarlyStoppingStrategy | None:
    """Get the default ESS for the given experiment.

    Returns a PercentileEarlyStoppingStrategy with a conservative configuration
    for single objective unconstrained problems. For all other problem types
    (multi-objective, constrained, or no optimization config), returns None.

    Args:
        experiment: The experiment to create an early stopping strategy for.

    Returns:
        A PercentileEarlyStoppingStrategy with default configuration if the
        experiment is a single objective unconstrained problem, None otherwise.
    """
    opt_config = experiment.optimization_config
    if (
        opt_config is None
        or isinstance(opt_config, MultiObjectiveOptimizationConfig)
        or len(opt_config.outcome_constraints) > 0
    ):
        return None

    return PercentileEarlyStoppingStrategy(
        percentile_threshold=50,
        min_curves=3,
        min_progression=0.2,
        max_progression=0.9,
        normalize_progressions=True,
        n_best_trials_to_complete=3,
        check_safe=True,
    )
