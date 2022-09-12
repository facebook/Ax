#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Tuple

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
            inactive_when_pending_trials: If set, the optimization will not stopped as
                long as it has running trials.
        """
        self.min_trials = min_trials
        self.inactive_when_pending_trials = inactive_when_pending_trials

    @abstractmethod
    def should_stop_optimization(
        self,
        experiment: Experiment,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Decide whether to stop optimization.

        Typical examples include stopping the optimization loop when the objective
        appears to not improve anymore.

        Args:
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """
        pass  # pragma: nocover
