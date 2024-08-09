#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from collections.abc import Iterable

from ax.core.base_trial import BaseTrial, TrialStatus


class SingleRunningTrialMixin:
    """Mixin for Runners with a single running trial.

    This mixin implements a simple poll_trial_status method that
    allows for a single running trial (the latest running trial).
    The returned status of trials that currently are marked as
    running is completed.
    """

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        trials = list(trials)
        if len(trials) == 0:
            return {}
        trial_statuses = defaultdict(set)
        running_trial_indices = trials[0].experiment.running_trial_indices
        max_running_trial_index = (
            -1 if len(running_trial_indices) == 0 else max(running_trial_indices)
        )
        for trial in trials:
            if trial.status in (
                TrialStatus.ABANDONED,
                TrialStatus.FAILED,
                TrialStatus.COMPLETED,
            ):
                continue
            elif (trial.status == TrialStatus.RUNNING) and (
                trial.index < max_running_trial_index
            ):
                trial_statuses[TrialStatus.COMPLETED].add(trial.index)
            else:
                trial_statuses[trial.status].add(trial.index)
        return dict(trial_statuses)
