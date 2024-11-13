#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from typing import Any

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.utils.common.base import Base


class BaseGlobalStoppingStrategy(ABC, Base):
    """Interface for strategies used to stop the optimization.

    Note that this is different from the `BaseEarlyStoppingStrategy`,
    the functionality of which is to decide whether a trial with partial
    results available during evaluation should be stopped before
    fully completing. In global early stopping, the decision is about
    whether or not to stop the overall optimzation altogether (e.g. b/c
    the expected marginal gains of running additional evaluations do not
    justify the cost of running these trials).
    """

    def __init__(
        self, min_trials: int, inactive_when_pending_trials: bool = True
    ) -> None:
        """
        Initiating a base stopping strategy.

        Args:
            min_trials: Minimum number of trials before the stopping strategy kicks in.
            inactive_when_pending_trials: If set, the optimization will not stop as
                long as it has running trials.
        """
        self.min_trials = min_trials
        self.inactive_when_pending_trials = inactive_when_pending_trials

    @abstractmethod
    def _should_stop_optimization(self, experiment: Experiment) -> tuple[bool, str]:
        """
        Decide whether to stop optimization.

        Must be implemented by the subclass. Typical examples include stopping
        the optimization loop when the objective appears to not improve anymore.

        Args:
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """

    def should_stop_optimization(
        self,
        experiment: Experiment,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Decide whether to stop optimization.

        Args:
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """
        if (
            self.inactive_when_pending_trials
            and len(experiment.trials_by_status[TrialStatus.RUNNING]) > 0
        ):
            message = "There are pending trials in the experiment."
            return False, message

        num_completed_trials = len(experiment.trials_by_status[TrialStatus.COMPLETED])
        if num_completed_trials < self.min_trials:
            message = (
                f"There are only {num_completed_trials} completed trials, "
                f"but {self} has a minumum of {self.min_trials}."
            )
            return False, message

        return self._should_stop_optimization(experiment, **kwargs)

    def estimate_global_stopping_savings(
        self, experiment: Experiment, num_remaining_requested_trials: int
    ) -> float:
        """Estimate global stopping savings by considering the number of requested
        trials versus the number of trials run before the decision to stop was made.

        This is formulated as 1 - (actual_num_trials / total_requested_trials). i.e.
        0.11 estimated savings indicates we would expect the experiment to have used
        11% more resources without global stopping present.

        Returns:
            The estimated resource savings as a fraction of total resource usage.
        """
        num_trials = len(experiment.trials)

        if num_remaining_requested_trials == 0:
            # Note that when no trials were requested, then savings
            # are 1 - 0 / 0. We resolve the zero division issue by
            # setting savings to 0 in that case.
            return 0.0

        return 1 - num_trials / (num_trials + num_remaining_requested_trials)
