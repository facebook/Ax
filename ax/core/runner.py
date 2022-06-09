#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Set, TYPE_CHECKING

from ax.utils.common.base import Base
from ax.utils.common.serialization import SerializationMixin


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Runner(Base, SerializationMixin, ABC):
    """Abstract base class for custom runner classes"""

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False

    @abstractmethod
    def run(self, trial: core.base_trial.BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass  # pragma: no cover

    def run_multiple(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> Dict[int, Dict[str, Any]]:
        """Runs a single evaluation for each of the given trials. Useful when deploying
        multiple trials at once is more efficient than deploying them one-by-one.
        Used in Ax ``Scheduler``.

        NOTE: By default simply loops over `run_trial`. Should be overwritten
        if deploying multiple trials in batch is preferable.

        Args:
            trials: Iterable of trials to be deployed, each containing arms with
                parameterizations to be evaluated. Can be a `Trial`
                if contains only one arm or a `BatchTrial` if contains
                multiple arms.

        Returns:
            Dict of trial index to the run metadata of that trial from the deployment
            process.
        """
        return {trial.index: self.run(trial=trial) for trial in trials}

    def poll_available_capacity(self) -> int:
        """Checks how much available capacity there is to schedule trial evaluations.
        Required for runners used with Ax ``Scheduler``.

        NOTE: This method might be difficult to implement in some systems. Returns -1
        if capacity of the system is "unlimited" or "unknown"
        (meaning that the ``Scheduler`` should be trying to schedule as many trials
        as is possible without violating scheduler settings). There is no need to
        artificially force this method to limit capacity; ``Scheduler`` has other
        limitations in place to limit number of trials running at once,
        like the ``SchedulerOptions.max_pending_trials`` setting, or
        more granular control in the form of the `max_parallelism`
        setting in each of the `GenerationStep`s of a `GenerationStrategy`).

        Returns:
            An integer, representing how many trials there is available capacity for;
            -1 if capacity is "unlimited" or not possible to know in advance.
        """
        return -1

    def poll_trial_status(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> Dict[core.base_trial.TrialStatus, Set[int]]:
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `poll_trial_status` "
            "method."
        )

    def stop(
        self, trial: core.base_trial.BaseTrial, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stop a trial based on custom runner subclass implementation.

        Optional method.

        Args:
            trial: The trial to stop.
            reason: A message containing information why the trial is to be stopped.

        Returns:
            A dictionary of run metadata from the stopping process.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `stop` method."
        )

    def clone(self) -> Runner:
        """Create a copy of this Runner."""
        cls = type(self)
        # pyre-ignore[45]: Cannot instantiate abstract class `Runner`.
        return cls(
            **cls.deserialize_init_args(args=cls.serialize_init_args(obj=self)),
        )

    def __eq__(self, other: Runner) -> bool:
        same_class = self.__class__ == other.__class__
        same_init_args = self.serialize_init_args(
            obj=self
        ) == other.serialize_init_args(obj=other)
        return same_class and same_init_args
