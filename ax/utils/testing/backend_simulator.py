#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ax.core.base_trial import TrialStatus
from ax.utils.common.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SimTrial:
    """Container for the simulation tasks.

    Attributes:
        trial_index: The index of the trial (should match Ax trial index).
        sim_runtime: The runtime of the trial (sampled at creation).
        sim_start_time: When the trial started running (or exits queued state).
        sim_queued_time: When the trial was initially queued.
        sim_completed_time: When the trial was marked as completed. Currently,
            this is used by an early-stopper via ``stop_trial``.
    """

    # The (Ax) trial index
    trial_index: int
    # The simulation runtime in seconds
    sim_runtime: float
    # the start time in seconds
    sim_start_time: Optional[float] = None
    # the queued time in seconds
    sim_queued_time: Optional[float] = None
    # the completed time (used for early stopping)
    sim_completed_time: Optional[float] = None


@dataclass
class SimStatus:
    """Container for status of the simulation.

    Attributes:
        queued: List of indices of queued trials.
        running: List of indices of running trials.
        failed: List of indices of failed trials.
        time_remaining: List of sim time remaining for running trials.
        completed: List of indicies of completed trials.
    """

    queued: List[int]
    running: List[int]
    failed: List[int]
    time_remaining: List[float]
    completed: List[int]


@dataclass
class BackendSimulatorOptions:
    """Settings for the BackendSimulator.

    Args:
        max_concurrency: The maximum number of trials that can be run
            in parallel.
        time_scaling: The factor to scale down the runtime of the tasks by.
            If ``runtime`` is the actual runtime of a trial, the simulation
            time will be ``runtime / time_scaling``.
        failure_rate: The rate at which the trials are failing. For now, trials
            fail independently with at coin flip based on that rate.
        internal_clock: The initial state of the internal clock. If `None`,
            the simulator uses ``time.time()`` as the clock.
        use_update_as_start_time: Whether the start time of a new trial should be logged
            as the current time (at time of update) or end time of previous trial.
            This makes sense when using the internal clock and the BackendSimulator
            is simulated forward by an external process (such as Scheduler).
    """

    max_concurrency: int = 1
    time_scaling: float = 1.0
    failure_rate: float = 0.0
    internal_clock: Optional[float] = None
    use_update_as_start_time: bool = False


@dataclass
class BackendSimulatorState:
    """State of the BackendSimulator.

    Args:
        options: The BackendSimulatorOptions associated with this simulator.
        verbose_logging: Whether the simulator is using verbose logging.
        queued: Currently queued trials.
        running: Currently running trials.
        failed: Currently failed trials.
        completed: Currently completed trials.
    """

    options: BackendSimulatorOptions
    verbose_logging: bool
    queued: List[Dict[str, Optional[float]]]
    running: List[Dict[str, Optional[float]]]
    failed: List[Dict[str, Optional[float]]]
    completed: List[Dict[str, Optional[float]]]


class BackendSimulator:
    """Simulator for a backend deployment with concurrent dispatch and a queue."""

    def __init__(
        self,
        options: Optional[BackendSimulatorOptions] = None,
        queued: Optional[List[SimTrial]] = None,
        running: Optional[List[SimTrial]] = None,
        failed: Optional[List[SimTrial]] = None,
        completed: Optional[List[SimTrial]] = None,
        verbose_logging: bool = True,
    ) -> None:
        """A simulator for a concurrent dispatch with a queue.

        Args:
            options: A ``BackendSimulatorOptions`` object with various settings
                of the backend simulator.
            queued: A list of SimTrial objects representing the queued trials
                (only used for testing particular initialization cases)
            running: A list of SimTrial objects representing the running trials
                (only used for testing particular initialization cases)
            failed: A list of SimTrial objects representing the failed trials
                (only used for testing particular initialization cases)
            completed: A list of SimTrial objects representing the completed trials
                (only used for testing particular initialization cases)
            verbose_logging: If False, sets the logging level to WARNING.
        """
        if not verbose_logging:
            logger.setLevel(logging.WARNING)  # pragma: no cover

        if options is None:
            options = BackendSimulatorOptions()

        self.max_concurrency = options.max_concurrency
        self.time_scaling = options.time_scaling
        self.failure_rate = options.failure_rate
        self.use_update_as_start_time = options.use_update_as_start_time
        self._queued: List[SimTrial] = queued or []
        self._running: List[SimTrial] = running or []
        self._failed: List[SimTrial] = failed or []
        self._completed: List[SimTrial] = completed or []
        self._internal_clock = options.internal_clock
        self._verbose_logging = verbose_logging
        self._init_state = self.state()
        self._create_index_to_trial_map()

    @property
    def num_queued(self) -> int:
        """The number of queued trials (to run as soon as capacity is available)."""
        return len(self._queued)

    @property
    def num_running(self) -> int:
        """The number of currently running trials."""
        return len(self._running)

    @property
    def num_failed(self) -> int:
        """The number of failed trials."""
        return len(self._failed)

    @property
    def num_completed(self) -> int:
        """The number of completed trials."""
        return len(self._completed)

    @property
    def use_internal_clock(self) -> bool:
        """Whether or not we are using the internal clock."""
        return self._internal_clock is not None

    @property
    def time(self) -> float:
        """The current time."""
        return self._internal_clock if self.use_internal_clock else time.time()

    @property
    def all_trials(self) -> List[SimTrial]:
        """All trials on the simulator."""
        return self._queued + self._running + self._completed + self._failed

    def update(self) -> None:
        """Update the state of the simulator."""
        if self.use_internal_clock:
            self._internal_clock += 1
        self._update(self.time)
        state = self.state()
        logger.info(
            "\n-----------\n"
            f"Updated backend simulator state (time = {self.time}):\n"
            f"** Queued:\n{format(state.queued)}\n"
            f"** Running:\n{format(state.running)}\n"
            f"** Failed:\n{format(state.failed)}\n"
            f"** Completed:\n{format(state.completed)}\n"
            f"-----------\n"
        )

    def reset(self) -> None:
        """Reset the simulator."""
        self.max_concurrency = self._init_state.options.max_concurrency
        self.time_scaling = self._init_state.options.time_scaling
        self._internal_clock = self._init_state.options.internal_clock
        self._queued = [SimTrial(**args) for args in self._init_state.queued]
        self._running = [SimTrial(**args) for args in self._init_state.running]
        self._failed = [SimTrial(**args) for args in self._init_state.failed]
        self._completed = [SimTrial(**args) for args in self._init_state.completed]
        self._create_index_to_trial_map()

    def state(self) -> BackendSimulatorState:
        """Return a ``BackendSimulatorState`` containing the state of the simulator."""

        options = BackendSimulatorOptions(
            max_concurrency=self.max_concurrency,
            time_scaling=self.time_scaling,
            failure_rate=self.failure_rate,
            internal_clock=self._internal_clock,
            use_update_as_start_time=self.use_update_as_start_time,
        )
        return BackendSimulatorState(
            options=options,
            verbose_logging=self._verbose_logging,
            queued=[q.__dict__.copy() for q in self._queued],
            running=[r.__dict__.copy() for r in self._running],
            failed=[r.__dict__.copy() for r in self._failed],
            completed=[c.__dict__.copy() for c in self._completed],
        )

    @classmethod
    def from_state(cls, state: BackendSimulatorState):
        """Construct a simulator from a state.

        Args:
            state: A ``BackendSimulatorState`` to set the simulator to.

        Returns:
            A ``BackendSimulator`` with the desired state.
        """
        trial_types = {
            "queued": state.queued,
            "running": state.running,
            "failed": state.failed,
            "completed": state.completed,
        }
        trial_kwargs = {
            key: [SimTrial(**kwargs) for kwargs in trial_types[key]]  # pyre-ignore [6]
            for key in ("queued", "running", "failed", "completed")
        }
        return cls(
            options=state.options, verbose_logging=state.verbose_logging, **trial_kwargs
        )

    def run_trial(self, trial_index: int, runtime: float) -> None:
        """Run a simulated trial.

        Args:
            trial_index: The index of the trial (usually the Ax trial index)
            runtime: The runtime of the simulation. Typically sampled from the
                runtime model of a simulation model.

        Internally, the runtime is scaled by the `time_scaling` factor, so that
        the simulation can run arbitrarily faster than the underlying evaluation.
        """
        # scale runtime to simulation
        sim_runtime = runtime / self.time_scaling

        # flip a coin to see if the trial fails (for now fail instantly)
        # TODO: Allow failure behavior based on a survival rate
        if self.failure_rate > 0:
            if random.random() < self.failure_rate:
                self._failed.append(
                    SimTrial(
                        trial_index=trial_index,
                        sim_runtime=sim_runtime,
                        sim_start_time=self.time,
                    )
                )
                return

        if self.num_running < self.max_concurrency:
            # note that though these are running for simulation purposes,
            # the trial status does not yet get updated (this is also how it
            # works in the real world, this requires updating the trial status manually)
            curr_time = self.time
            new_trial = SimTrial(
                trial_index=trial_index,
                sim_runtime=sim_runtime,
                sim_start_time=curr_time,
                sim_queued_time=curr_time,
            )
            self.new_trial(trial=new_trial, status=TrialStatus.RUNNING)
        else:
            new_trial = SimTrial(
                trial_index=trial_index,
                sim_runtime=sim_runtime,
                sim_queued_time=self.time,
            )
            self.new_trial(trial=new_trial, status=TrialStatus.STAGED)

    def new_trial(self, trial: SimTrial, status: TrialStatus) -> None:
        """Register a trial into the simulator.

        Args:
            trial: A new trial to add.
            status: The status of the new trial, either STAGED (add to ``self._queued``)
                or RUNNING (add to ``self._running``).
        """
        if status == TrialStatus.STAGED:
            self._queued.append(trial)
        elif status == TrialStatus.RUNNING:
            self._running.append(trial)
        else:
            raise ValueError("New trials must be either staged or running.")
        self._index_to_trial_map[trial.trial_index] = trial

    def stop_trial(self, trial_index: int) -> None:
        """Stop a simulated trial by setting the completed time to the current time.

        Args:
            trial_index: The index of the trial to stop.
        """
        trial_status = self.lookup_trial_index_status(trial_index)
        if trial_status is not TrialStatus.RUNNING:
            logger.info(
                f"Trial {trial_index} is not currently running (has status "
                f"{trial_status}) and cannot be stopped."
            )
        else:
            trial = self._index_to_trial_map[trial_index]
            trial.sim_completed_time = self.time
            logger.info(
                f"Trial {trial_index} stopped at time {trial.sim_completed_time}."
            )

    def status(self) -> SimStatus:
        """Return the internal status of the simulator.

        Returns:
            A ``SimStatus`` object representing the current simulator state.
        """
        now = self.time
        return SimStatus(
            queued=[t.trial_index for t in self._queued],
            running=[t.trial_index for t in self._running],
            failed=[t.trial_index for t in self._failed],
            time_remaining=[
                # pyre-fixme[58]: `+` is not supported for operand types
                #  `Optional[float]` and `float`.
                t.sim_start_time + t.sim_runtime - now
                for t in self._running
            ],
            completed=[t.trial_index for t in self._completed],
        )

    def lookup_trial_index_status(self, trial_index: int) -> Optional[TrialStatus]:
        """Lookup the trial status of a ``trial_index``.

        Args:
            trial_index: The index of the trial to check.

        Returns:
            A ``TrialStatus``.
        """
        sim_status = self.status()
        if trial_index in sim_status.queued:
            return TrialStatus.STAGED
        elif trial_index in sim_status.running:
            return TrialStatus.RUNNING
        elif trial_index in sim_status.completed:
            return TrialStatus.COMPLETED
        elif trial_index in sim_status.failed:
            return TrialStatus.FAILED
        return None

    def get_sim_trial_by_index(self, trial_index: int) -> Optional[SimTrial]:
        """Get a ``SimTrial`` by ``trial_index``.

        Args:
            trial_index: The index of the trial to return.

        Returns:
            A ``SimTrial`` with the index ``trial_index`` or None if not found.
        """
        return self._index_to_trial_map.get(trial_index)

    def _update_completed(self, timestamp: float) -> List[SimTrial]:
        """Look through running trials and see if any trials have completed
        since the last check. Such trials could have completed naturally (in
        this case, ``sim_completed_time`` is None) or have been given a artificial
        completion time (``sim_completed_time`` is not None) via early stopping.

        Args:
            timestamp: The current timestamp.
        """
        completed_since_last = []
        new_running = []
        for trial in self._running:
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `Optional[float]` and `float`.
            if timestamp >= trial.sim_start_time + trial.sim_runtime:
                completed_since_last.append(trial)
                trial.sim_completed_time = (
                    trial.sim_start_time + trial.sim_runtime  # pyre-ignore[58]
                )
            elif (
                trial.sim_completed_time is not None
                and timestamp >= trial.sim_completed_time  # pyre-ignore[58]
            ):
                completed_since_last.append(trial)  # was early stopped
            else:
                new_running.append(trial)
        self._running = new_running
        self._completed.extend(completed_since_last)
        return completed_since_last

    def _update(self, timestamp: float) -> None:
        """Check if trials have completed (or stopped) and update the simulator.

        Args:
            timestamp: The current timestamp.
        """

        completed_since_last = self._update_completed(timestamp)

        # if no trial has finished since the last call we're done
        if len(completed_since_last) == 0:
            return

        # if at least one trial has finished, we need to graduate queued trials to
        # running trials. Since all we need to keep track of is the start_time, we can
        # do this retroactively.
        # TODO: Improve performance / make less ad hoc by using a priority queue
        for c in completed_since_last:
            if self.num_queued > 0:
                new_running_trial = self._queued.pop(0)
                sim_start_time = (
                    # pyre-fixme[58]: `+` is not supported for operand types
                    #  `Optional[float]` and `float`.
                    c.sim_start_time + c.sim_runtime
                    if not self.use_update_as_start_time
                    else self.time
                )
                new_running_trial.sim_start_time = sim_start_time
                self._running.append(new_running_trial)

        # since these graduated trials could both have started and finished in between
        # the simulation updates, we need to re-run the update with the new state
        self._update(timestamp)

    def _create_index_to_trial_map(self) -> None:
        """Create the index to trial map, which is useful for getting
        the ``SimTrial`` objects, as in ``get_sim_trial_by_index``."""
        self._index_to_trial_map = {t.trial_index: t for t in self.all_trials}


def format(trial_list: List[Dict[str, Optional[float]]]) -> str:
    """Helper function for formatting a list."""
    trial_list_str = [str(i) for i in trial_list]
    return "\n".join(trial_list_str)
