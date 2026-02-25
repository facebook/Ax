#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus


class MultiTypeExperiment(Experiment):
    """Class for experiment with multiple trial types.

    .. deprecated::
        The `MultiTypeExperiment` class is deprecated. Use `Experiment` with
        `default_trial_type` parameter instead. All multi-type experiment
        functionality has been moved to the base `Experiment` class.

    """


def filter_trials_by_type(
    trials: Sequence[BaseTrial],
    trial_type: str | None,
) -> list[BaseTrial]:
    """Filter trials by trial type if provided.

    This filters trials by trial type if the experiment is a
    MultiTypeExperiment.

    Args:
        trials: Trials to filter.

    Returns:
        Filtered trials.
    """
    if trial_type is not None:
        return [t for t in trials if t.trial_type == trial_type]
    return list(trials)


def get_trial_indices_for_statuses(
    experiment: Experiment,
    statuses: set[TrialStatus],
    trial_type: str | None = None,
) -> set[int]:
    """Get trial indices for a set of statuses.

    Args:
        statuses: Set of statuses to get trial indices for.

    Returns:
        Set of trial indices for the given statuses.
    """
    return {
        i
        for i, t in experiment.trials.items()
        if (t.status in statuses)
        and (
            (trial_type is None)
            or ((trial_type is not None) and (t.trial_type == trial_type))
        )
    }
