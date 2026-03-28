#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Any, Self, TYPE_CHECKING

from ax.utils.common.base import Base
from ax.utils.common.serialization import SerializationMixin


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Runner(Base, SerializationMixin, ABC):
    """Abstract base class for custom runner classes"""

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False

    @property
    def run_metadata_report_keys(self) -> list[str]:
        """A list of keys of the metadata dict returned by ``run_trials()`` that
        are relevant outside the runner-internal implementation. These can e.g.
        be reported in ``orchestrator.report_results()``."""
        return []

    def run(self, trial: core.base_trial.BaseTrial) -> dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        .. deprecated::
            Override ``run_trials`` instead. This method exists only for
            backward compatibility and will be removed in a future release.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `run`. "
            "Override `run_trials` instead."
        )

    def run_trials(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[int, dict[str, Any]]:
        """Deploys one or more trials based on custom runner subclass
        implementation.

        Subclasses should override this method. The default implementation
        falls back to calling the deprecated ``run()`` method for each trial,
        which allows existing runners to continue working until they are
        migrated.

        Args:
            trials: Iterable of trials to be deployed, each containing arms
                with parameterizations to be evaluated. Can be a ``Trial``
                if contains only one arm or a ``BatchTrial`` if contains
                multiple arms.

        Returns:
            Dict of trial index to the run metadata of that trial from the
            deployment process.
        """
        return {trial.index: self.run(trial=trial) for trial in trials}

    def run_multiple(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[int, dict[str, Any]]:
        """Runs a single evaluation for each of the given trials.

        .. deprecated::
            Use ``run_trials`` instead. This method exists only for backward
            compatibility and will be removed in a future release.
        """
        warnings.warn(
            "`run_multiple` is deprecated. Use `run_trials` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_trials(trials=trials)

    def stage_trials(self, trials: Iterable[core.base_trial.BaseTrial]) -> None:
        """Stage trials for later execution.

        Does nothing by default. Override for runners that support a staging
        step before running (e.g. creating a paused job).

        Args:
            trials: Trials to stage.
        """
        pass

    def poll_available_capacity(self) -> int:
        """Checks how much available capacity there is to schedule trial
        evaluations. Required for runners used with Ax ``Orchestrator``.

        NOTE: This method might be difficult to implement in some systems.
        Returns -1 if capacity of the system is "unlimited" or "unknown"
        (meaning that the ``Orchestrator`` should be trying to schedule as many
        trials as is possible without violating Orchestrator settings). There is
        no need to artificially force this method to limit capacity;
        ``Orchestrator`` has other limitations in place to limit number of
        trials running at once, like the
        ``OrchestratorOptions.max_pending_trials`` setting, or more granular
        control in the form of the `max_parallelism` setting in each of the
        `GenerationStep`s of a `GenerationStrategy`).

        Returns:
            An integer, representing how many trials there is available
            capacity for; -1 if capacity is "unlimited" or not possible to
            know in advance.
        """
        return -1

    def poll_trial_status(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[core.base_trial.TrialStatus, set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Orchestrator``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that
            have the respective status at the time of the polling. This does
            not need to include trials that at the time of polling already have
            a terminal (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a "
            "`poll_trial_status` method."
        )

    def poll_run_metadata(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[int, dict[str, Any]]:
        """Update run metadata based on actual deployment results.

        For example, a trial might not actually start when ``run_trials`` is
        called (it may be queued/"staged" for a while). This method allows
        runners to update start/end dates and other run metadata to reflect
        reality.

        Args:
            trials: Trials whose run metadata should be refreshed.

        Returns:
            Dict of trial index to updated run metadata. Only trials whose
            metadata changed need to be included.
        """
        return {}

    def poll_exception(self, trial: core.base_trial.BaseTrial) -> str:
        """Returns the exception from a trial.

        .. deprecated::
            Fold exception reporting into ``poll_trial_status`` by including
            a ``status_reason`` in the trial's run metadata. This method will
            be removed in a future release.

        Args:
            trial: Trial to get exception for.

        Returns:
            Exception string.
        """
        warnings.warn(
            "`poll_exception` is deprecated. Fold exception reporting into "
            "`poll_trial_status` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `poll_exception` method."
        )

    def stop_trial(
        self, trial: core.base_trial.BaseTrial, reason: str | None = None
    ) -> dict[str, Any]:
        """Stop a trial based on custom runner subclass implementation.

        Optional method.

        Args:
            trial: The trial to stop.
            reason: A message containing information why the trial is to
                be stopped.

        Returns:
            A dictionary of run metadata from the stopping process.
        """
        # Fall back to the deprecated `stop` method for backward compatibility.
        return self.stop(trial=trial, reason=reason)

    def stop(
        self, trial: core.base_trial.BaseTrial, reason: str | None = None
    ) -> dict[str, Any]:
        """Stop a trial based on custom runner subclass implementation.

        .. deprecated::
            Override ``stop_trial`` instead. This method exists only for
            backward compatibility and will be removed in a future release.

        Args:
            trial: The trial to stop.
            reason: A message containing information why the trial is to
                be stopped.

        Returns:
            A dictionary of run metadata from the stopping process.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `stop` method."
        )

    def stop_arms(
        self,
        trial: core.base_trial.BaseTrial,
        arm_names: list[str],
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Stop specific arms within a running trial.

        Optional method. Override for runners that support stopping individual
        arms without stopping the entire trial.

        Args:
            trial: The trial containing the arms to stop.
            arm_names: Names of the arms to stop.
            reason: A message containing information why the arms are to
                be stopped.

        Returns:
            A dictionary of run metadata from the stopping process.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `stop_arms` method."
        )

    def apply_status_change_side_effects(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> set[int]:
        """Apply side effects triggered by trial status changes.

        Called after ``poll_trial_status`` to handle actions that should occur
        when a trial's status transitions (e.g. treating trials that ran for
        "too short" as failed). Gives these side effects a dedicated place
        rather than chaining them onto ``poll_trial_status``.

        Args:
            trials: Trials whose status may have changed.

        Returns:
            Set of trial indices that were updated by side effects.
        """
        return set()

    def clone(self) -> Self:
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
