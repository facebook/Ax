#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.runner import Runner
from ax.core.types import TCandidateMetadata
from ax.utils.common.typeutils import not_none


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


class TrialStatus(int, Enum):
    """Enum of trial status.

    General lifecycle of a trial is:::

        CANDIDATE --> STAGED --> RUNNING --> COMPLETED
                  ------------->         --> FAILED (machine failure)
                  -------------------------> ABANDONED (human-initiated action)

    Trials may be abandoned at any time prior to completion or failure
    via human intervention. The difference between abandonment and failure
    is that the former is human-directed, while the latter is an internal
    failure state.

    Additionally, when trials are deployed, they may be in an intermediate
    staged state (e.g. scheduled but waiting for resources) or immediately
    transition to running.
    """

    CANDIDATE = 0
    STAGED = 1
    FAILED = 2
    COMPLETED = 3
    RUNNING = 4
    ABANDONED = 5
    DISPATCHED = 6  # Deprecated.

    @property
    def is_terminal(self) -> bool:
        """True if trial is completed."""
        return (
            self == TrialStatus.ABANDONED
            or self == TrialStatus.COMPLETED
            or self == TrialStatus.FAILED
        )

    @property
    def expecting_data(self) -> bool:
        """True if trial is expecting data."""
        return self == TrialStatus.RUNNING or self == TrialStatus.COMPLETED

    @property
    def is_deployed(self) -> bool:
        """True if trial has been deployed but not completed."""
        return self == TrialStatus.STAGED or self == TrialStatus.RUNNING

    @property
    def is_failed(self) -> bool:
        """True if this trial is a failed one."""
        return self == TrialStatus.FAILED

    @property
    def is_abandoned(self) -> bool:
        """True if this trial is an abandoned one."""
        return self == TrialStatus.ABANDONED

    @property
    def is_candidate(self) -> bool:
        """True if this trial is a candidate."""
        return self == TrialStatus.CANDIDATE

    @property
    def is_completed(self) -> bool:
        """True if this trial is a successfully completed one."""
        return self == TrialStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """True if this trial is a running one."""
        return self == TrialStatus.RUNNING

    def __format__(self, fmt: str) -> str:
        """Define `__format__` to avoid pulling the `__format__` from the `int`
        mixin (since its better for statuses to show up as `RUNNING` than as
        just an int that is difficult to interpret).

        E.g. batch trial representation with the overridden method is:
        "BatchTrial(experiment_name='test', index=0, status=TrialStatus.CANDIDATE)".

        Docs on enum formatting: https://docs.python.org/3/library/enum.html#others.
        """
        return f"{self!s}"


def immutable_once_run(func: Callable) -> Callable:
    """Decorator for methods that should throw Error when
    trial is running or has ever run and immutable.
    """

    # no type annotation for now; breaks sphinx-autodoc-typehints
    def _immutable_once_run(self, *args, **kwargs):
        if self._status != TrialStatus.CANDIDATE:
            raise ValueError(
                "Cannot modify a trial that is running or has ever run.",
                "Create a new trial using `experiment.new_trial()` "
                "or clone an existing trial using `trial.clone()`.",
            )
        return func(self, *args, **kwargs)

    return _immutable_once_run


class BaseTrial(ABC, Base):
    """Base class for representing trials.

    Trials are containers for arms that are deployed together. There are
    two kinds of trials: regular Trial, which only contains a single arm,
    and BatchTrial, which contains an arbitrary number of arms.

    Args:
        experiment: Experiment, of which this trial is a part
        trial_type: Type of this trial, if used in MultiTypeExperiment.
        ttl_seconds: If specified, trials will be considered failed after
            this many seconds since the time the trial was ran, unless the
            trial is completed before then. Meant to be used to detect
            'dead' trials, for which the evaluation process might have
            crashed etc., and which should be considered failed after
            their 'time to live' has passed.
    """

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        trial_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Initialize trial.

        Args:
            experiment: The experiment this trial belongs to.
        """
        self._experiment = experiment
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("TTL must be a positive integer (or None).")
        self._ttl_seconds: Optional[int] = ttl_seconds
        self._index = self._experiment._attach_trial(self)

        if trial_type is not None:
            if not self._experiment.supports_trial_type(trial_type):
                raise ValueError(
                    f"Experiment does not support trial_type {trial_type}."
                )
        else:
            trial_type = self._experiment.default_trial_type
        self._trial_type: Optional[str] = trial_type

        self.__status = None
        # Uses `_status` setter, which updates trial statuses to trial indices
        # mapping on the experiment, with which this trial is associated.
        self._status = TrialStatus.CANDIDATE
        self._time_created: datetime = datetime.now()

        # Initialize fields to be used later in lifecycle
        self._time_completed: Optional[datetime] = None
        self._time_staged: Optional[datetime] = None
        self._time_run_started: Optional[datetime] = None

        self._abandoned_reason: Optional[str] = None
        self._run_metadata: Dict[str, Any] = {}

        self._runner: Optional[Runner] = None

        # Counter to maintain how many arms have been named by this BatchTrial
        self._num_arms_created = 0

        # If generator run(s) in this trial were generated from a generation
        # strategy, this property will be set to the generation step that produced
        # the generator run(s).
        self._generation_step_index = None

    @property
    def experiment(self) -> core.experiment.Experiment:
        """The experiment this trial belongs to."""
        return self._experiment

    @property
    def index(self) -> int:
        """The index of this trial within the experiment's trial list."""
        return self._index

    @property
    def status(self) -> TrialStatus:
        """The status of the trial in the experimentation lifecycle."""
        self._mark_failed_if_past_TTL()
        return self._status

    @property
    def ttl_seconds(self) -> Optional[int]:
        """This trial's time-to-live once ran, in seconds. If not set, trial
        will never be automatically considered failed (i.e. infinite TTL).
        Reflects after how many seconds since the time the trial was run it
        will be considered failed unless completed.
        """
        return self._ttl_seconds

    @ttl_seconds.setter
    def ttl_seconds(self, ttl_seconds: Optional[int]) -> None:
        """Sets this trial's time-to-live once ran, in seconds. If None, trial
        will never be automatically considered failed (i.e. infinite TTL).
        Reflects after how many seconds since the time the trial was run it
        will be considered failed unless completed.
        """
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("TTL must be a positive integer (or None).")
        self._ttl_seconds = ttl_seconds

    @property
    def completed_successfully(self) -> bool:
        """Checks if trial status is `COMPLETED`."""
        return self.status == TrialStatus.COMPLETED

    @property
    def did_not_complete(self) -> bool:
        """Checks if trial status is terminal, but not `COMPLETED`."""
        return self.status.is_terminal and not self.completed_successfully

    @status.setter
    def status(self, status: TrialStatus) -> None:
        raise NotImplementedError("Use `trial.mark_*` methods to set trial status.")

    @property
    def runner(self) -> Optional[Runner]:
        """The runner object defining how to deploy the trial."""
        return self._runner

    @runner.setter
    @immutable_once_run
    def runner(self, runner: Optional[Runner]) -> None:
        if self.experiment.is_simple_experiment:
            raise NotImplementedError(
                "SimpleExperiment does not support addition of runners."
            )
        self._runner = runner

    @property
    def deployed_name(self) -> Optional[str]:
        """Name of the experiment created in external framework.

        This property is derived from the name field in run_metadata.
        """
        return self._run_metadata.get("name") if self._run_metadata else None

    @property
    def run_metadata(self) -> Dict[str, Any]:
        """Dict containing metadata from the deployment process.

        This is set implicitly during `trial.run()`.
        """
        return self._run_metadata

    @property
    def trial_type(self) -> Optional[str]:
        """The type of the trial.

        Relevant for experiments containing different kinds of trials
        (e.g. different deployment types).
        """
        return self._trial_type

    @trial_type.setter
    @immutable_once_run
    def trial_type(self, trial_type: Optional[str]) -> None:
        """Identifier used to distinguish trial types in experiments
           with multiple trial types.
        """
        if self._experiment is not None:
            if not self._experiment.supports_trial_type(trial_type):
                raise ValueError(f"{trial_type} is not supported by the experiment.")

        self._trial_type = trial_type

    def assign_runner(self) -> BaseTrial:
        """Assigns default experiment runner if trial doesn't already have one."""
        self._runner = self._runner or self.experiment.runner_for_trial(self)
        return self

    def update_run_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the run metadata dict stored on this trial and returns the
        updated dict."""
        self._run_metadata.update(metadata)
        return self._run_metadata

    def run(self) -> BaseTrial:
        """Deploys the trial according to the behavior on the runner.

        The runner returns a `run_metadata` dict containining metadata
        of the deployment process. It also returns a `deployed_name` of the trial
        within the system to which it was deployed. Both these fields are set on
        the trial.

        Returns:
            The trial instance.
        """
        if self.status != TrialStatus.CANDIDATE:
            raise ValueError("Can only run a candidate trial.")

        # Default to experiment runner if trial doesn't have one
        self.assign_runner()

        if self._runner is None:
            raise ValueError("No runner set on trial or experiment.")

        self._run_metadata = not_none(self._runner).run(self)

        if not_none(self._runner).staging_required:
            self.mark_staged()
        else:
            self.mark_running()
        return self

    def complete(self) -> BaseTrial:
        """Stops the trial if functionality is defined on runner
            and marks trial completed.

        Returns:
            The trial instance.
        """
        if self.status != TrialStatus.RUNNING:
            raise ValueError("Can only stop a running trial.")

        not_none(self._runner).stop(self)
        self.mark_completed()
        return self

    def fetch_data(self, metrics: Optional[List[Metric]] = None, **kwargs: Any) -> Data:
        """Fetch data for this trial for all metrics on experiment.

        Args:
            trial_index: The index of the trial to fetch data for.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for this trial.
        """
        return self.experiment._fetch_trial_data(
            trial_index=self.index, metrics=metrics, **kwargs
        )

    def _check_existing_and_name_arm(self, arm: Arm) -> None:
        """Sets name for given arm; if this arm is already in the
        experiment, uses the existing arm name.
        """
        proposed_name = f"{self.index}_{self._num_arms_created}"
        self.experiment._name_and_store_arm_if_not_exists(
            arm=arm, proposed_name=proposed_name
        )
        # If arm was named using given name, incremement the count
        if arm.name == proposed_name:
            self._num_arms_created += 1

    def _set_generation_step_index(self, generation_step_index: Optional[int]) -> None:
        """Sets the `generation_step_index` property of the trial, to reflect which
        generation step of a given generation strategy (if any) produced the generator
        run(s) attached to this trial.
        """
        if (
            self._generation_step_index is not None
            and generation_step_index is not None
            and self._generation_step_index != generation_step_index
        ):
            raise ValueError(  # pragma: no cover
                "Cannot add generator runs from different generation steps to a "
                "single trial."
            )
        self._generation_step_index = generation_step_index

    @abstractproperty
    def arms(self) -> List[Arm]:
        pass  # pragma: no cover

    @abstractproperty
    def arms_by_name(self) -> Dict[str, Arm]:
        pass  # pragma: no cover

    @abstractmethod
    def __repr__(self) -> str:
        pass  # pragma: no cover

    @abstractproperty
    def abandoned_arms(self) -> List[Arm]:
        """All abandoned arms, associated with this trial."""
        pass  # pragma: no cover

    @abstractmethod
    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> Dict[str, TCandidateMetadata]:
        """Retrieves combined candidate metadata from all generator runs associated
        with this trial.
        """
        ...

    # --- Trial lifecycle management functions ---

    @property
    def time_created(self) -> datetime:
        """Creation time of the trial."""
        return self._time_created

    @property
    def time_completed(self) -> Optional[datetime]:
        """Completion time of the trial."""
        return self._time_completed

    @property
    def time_staged(self) -> Optional[datetime]:
        """Staged time of the trial."""
        return self._time_staged

    @property
    def time_run_started(self) -> Optional[datetime]:
        """Time the trial was started running (i.e. collecting data)."""
        return self._time_run_started

    @property
    def is_abandoned(self) -> bool:
        """Whether this trial is abandoned."""
        return self._status == TrialStatus.ABANDONED

    @property
    def abandoned_reason(self) -> Optional[str]:
        return self._abandoned_reason

    def mark_staged(self) -> BaseTrial:
        """Mark the trial as being staged for running.

        Returns:
            The trial instance.
        """
        if self._status != TrialStatus.CANDIDATE:
            raise ValueError("Can only stage a candidate trial.")
        self._status = TrialStatus.STAGED
        self._time_staged = datetime.now()
        return self

    def mark_running(self, no_runner_required: bool = False) -> BaseTrial:
        """Mark trial has started running.

        Returns:
            The trial instance.
        """
        if self.experiment.is_simple_experiment:
            self._status = TrialStatus.RUNNING
            self._time_run_started = datetime.now()
            return self

        if self._runner is None and not no_runner_required:
            raise ValueError("Cannot mark trial running without setting runner.")

        prev_step = (
            TrialStatus.STAGED
            # pyre-fixme[16]: `Optional` has no attribute `staging_required`.
            if self._runner is not None and self._runner.staging_required
            else TrialStatus.CANDIDATE
        )
        prev_step_str = "staged" if prev_step == TrialStatus.STAGED else "candidate"
        if self._status != prev_step:
            raise ValueError(
                f"Can only mark this trial as running when {prev_step_str}."
            )
        self._status = TrialStatus.RUNNING
        self._time_run_started = datetime.now()
        return self

    def mark_completed(self) -> BaseTrial:
        """Mark trial as completed.

        Args:
            allow_repeat_completion: If set to True, this function will not raise an
                error is a trial that has already been marked as completed is
                being marked as completed again.

        Returns:
            The trial instance.
        """
        if self._status != TrialStatus.RUNNING:
            raise ValueError("Can only complete trial that is currently running.")
        self._status = TrialStatus.COMPLETED
        self._time_completed = datetime.now()
        return self

    def mark_abandoned(self, reason: Optional[str] = None) -> BaseTrial:
        """Mark trial as abandoned.

        Args:
            abandoned_reason: The reason the trial was abandoned.

        Returns:
            The trial instance.
        """
        if self._status.is_terminal:
            raise ValueError("Cannot abandon a trial in a terminal state.")

        self._abandoned_reason = reason
        self._status = TrialStatus.ABANDONED
        self._time_completed = datetime.now()
        return self

    def mark_failed(self) -> BaseTrial:
        """Mark trial as failed.

        Returns:
            The trial instance.
        """
        if self._status != TrialStatus.RUNNING:
            raise ValueError("Can only mark failed a trial that is currently running.")

        self._status = TrialStatus.FAILED
        self._time_completed = datetime.now()
        return self

    def mark_as(self, status: TrialStatus, **kwargs: Any) -> BaseTrial:
        """Mark trial with a new TrialStatus.

        Args:
            status: The new status of the trial.
            kwargs: Additional keyword args, as can be ued in the respective `mark_`
                methods associated with the trial status.

        Returns:
            The trial instance.
        """
        if status == TrialStatus.STAGED:
            self.mark_staged()
        if status == TrialStatus.RUNNING:
            no_runner_required = kwargs.get("no_runner_required", False)
            self.mark_running(no_runner_required=no_runner_required)
        if status == TrialStatus.ABANDONED:
            self.mark_abandoned(reason=kwargs.get("reason"))
        if status == TrialStatus.FAILED:
            self.mark_failed()
        if status == TrialStatus.COMPLETED:
            self.mark_completed()
        return self

    def _mark_failed_if_past_TTL(self) -> None:
        """If trial has TTL set and is running, check if the TTL has elapsed
        and mark the trial failed if so.
        """
        if self.ttl_seconds is None or not self._status.is_running:
            return
        time_run_started = self._time_run_started
        assert time_run_started is not None
        dt = datetime.now() - time_run_started
        if dt > timedelta(seconds=not_none(self.ttl_seconds)):
            self.mark_failed()

    @property
    def _status(self) -> TrialStatus:
        """The status of the trial in the experimentation lifecycle. This private
        property exists to allow for a corresponding setter, since its important
        that the trial statuses mapping on the experiment is updated always when
        a trial status is updated.
        """
        return self.__status

    @_status.setter
    def _status(self, trial_status: TrialStatus) -> None:
        """Setter for the `_status` attribute that also updates the experiment's
        `_trial_indices_by_status mapping according to the newly set trial status.
        """
        if self._status is not None:
            assert self.index in self._experiment._trial_indices_by_status[self._status]
            self._experiment._trial_indices_by_status[self._status].remove(self.index)
        self._experiment._trial_indices_by_status[trial_status].add(self.index)
        self.__status = trial_status
