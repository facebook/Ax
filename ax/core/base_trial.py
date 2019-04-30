#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.runner import Runner


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


class TrialStatus(Enum):
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
    two types of trials: regular Trial, which only contains a single arm,
    and BatchTrial, which contains an arbitrary number of arms.
    """

    def __init__(
        self, experiment: "core.experiment.Experiment", trial_type: Optional[str] = None
    ) -> None:
        """Initialize trial.

        Args:
            experiment: The experiment this trial belongs to.
        """
        self._experiment = experiment
        self._index = self._experiment._attach_trial(self)

        if trial_type is not None:
            if not self._experiment.supports_trial_type(trial_type):
                raise ValueError(
                    f"Experiment does not support trial_type {trial_type}."
                )
        else:
            trial_type = self._experiment.default_trial_type
        self._trial_type: Optional[str] = trial_type

        self._status: TrialStatus = TrialStatus.CANDIDATE
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

    @property
    def experiment(self) -> "core.experiment.Experiment":
        """The experiment this trial belongs to."""
        return self._experiment

    @property
    def index(self) -> int:
        """The index of this trial within the experiment's trial list."""
        return self._index

    @property
    def status(self) -> TrialStatus:
        """The status of the trial in the experimentation lifecycle."""
        return self._status

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

    def assign_runner(self) -> "BaseTrial":
        """Assigns default experiment runner if trial doesn't already have one."""
        self._runner = self._runner or self.experiment.runner_for_trial(self)
        return self

    def run(self) -> "BaseTrial":
        """Deploys the trial according to the behavior on the runner.

        The runner returns a `run_metadata` dict containining metadata
        of the deployment process. It also returns a `deployed_name` of the trial
        within the system to which it was deployed. Both these fields are set on
        the trial.

        Returns:
            The trial instance.
        """

        # Default to experiment runner if trial doesn't have one
        self.assign_runner()

        if self._runner is None:
            raise ValueError("No runner set on trial or experiment.")

        self._run_metadata = self._runner.run(self)

        if self._runner.staging_required:
            self.mark_staged()
        else:
            self.mark_running()
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

    # --- Batch lifecycle management functions ---

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

    def mark_staged(self) -> "BaseTrial":
        """Mark the trial as being staged for running.

        Returns:
            The trial instance.

        """
        if self._status != TrialStatus.CANDIDATE:
            raise ValueError("Can only stage a candidate trial.")
        self._status = TrialStatus.STAGED
        self._time_staged = datetime.now()
        return self

    def mark_running(self) -> "BaseTrial":
        """Mark trial has started running.

        Returns:
            The trial instance.

        """
        if self.experiment.is_simple_experiment:
            self._status = TrialStatus.RUNNING
            self._time_run_started = datetime.now()
            return self

        if self._runner is None:
            raise ValueError("Cannot mark trial running without setting runner.")

        prev_step = (
            TrialStatus.STAGED
            if self._runner.staging_required
            else TrialStatus.CANDIDATE
        )
        prev_step_str = "staged." if prev_step == TrialStatus.STAGED else "candidate."
        if self._status != prev_step:
            raise ValueError(
                f"Can only mark this trial as running when {prev_step_str}"
            )
        self._status = TrialStatus.RUNNING
        self._time_run_started = datetime.now()
        return self

    def mark_completed(self) -> "BaseTrial":
        """Mark trial as completed.

        Returns:
            The trial instance.
        """
        if self._status != TrialStatus.RUNNING:
            raise ValueError("Can only complete trial that is currently running.")
        self._status = TrialStatus.COMPLETED
        self._time_completed = datetime.now()
        return self

    def mark_abandoned(self, reason: Optional[str] = None) -> "BaseTrial":
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

    def mark_failed(self) -> "BaseTrial":
        """Mark trial as failed.

        Returns:
            The trial instance.
        """
        if self._status != TrialStatus.RUNNING:
            raise ValueError(
                "Can only mark as failed a trial that is currently running."
            )

        self._status = TrialStatus.FAILED
        self._time_completed = datetime.now()
        return self
