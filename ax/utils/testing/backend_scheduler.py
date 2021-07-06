#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.runners.simulated_backend import SimulatedBackendRunner
from ax.service.scheduler import (
    Scheduler,
    SchedulerOptions,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from ax.utils.testing.backend_simulator import BackendSimulator

logger = get_logger(__name__)


class AsyncSimulatedBackendScheduler(Scheduler):
    """A Scheduler that uses a simulated backend for Ax asynchronous benchmarks."""

    def __init__(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        max_pending_trials: int,
        options: SchedulerOptions,
    ) -> None:
        """A Scheduler for Ax asynchronous benchmarks.

        Args:
            experiment: Experiment, in which results of the optimization
                will be recorded.
            generation_strategy: Generation strategy for the optimization,
                describes models that will be used in optimization.
            max_pending_trials: The maximum number of pending trials allowed.
            options: `SchedulerOptions` for this Scheduler instance.
        """
        if not isinstance(experiment.runner, SimulatedBackendRunner):
            raise ValueError(
                "experiment must have runner of type SimulatedBackendRunner attached"
            )

        super().__init__(
            experiment=experiment,
            generation_strategy=generation_strategy,
            options=options,
            _skip_experiment_save=True,
        )
        self.max_pending_trials = max_pending_trials

    @property
    def backend_simulator(self) -> BackendSimulator:
        """Get the ``BackendSimulator`` stored on the runner of the experiment.

        Returns:
            The backend simulator.
        """
        return self.experiment.runner.simulator  # pyre-ignore[16]

    def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
        """Poll trial status from the ``BackendSimulator``. NOTE: The ``Scheduler``
        currently marks trials as running when they are created, but some of these
        trials may actually be in queued on the ``BackendSimulator``.

        Returns:
            A Dict mapping statuses to sets of trial indices.
        """
        self.backend_simulator.update()
        trials_by_status = self.experiment.trials_by_status
        trial_status = defaultdict(set)
        for ts in (TrialStatus.CANDIDATE, TrialStatus.STAGED, TrialStatus.RUNNING):
            for trial in trials_by_status[ts]:
                t_index = trial.index
                status = self.backend_simulator.lookup_trial_index_status(t_index)
                trial_status[status].add(t_index)
        return dict(trial_status)

    def has_capacity(self, n: int = 1) -> bool:
        """Whether or not there is available capacity for ``n`` trials.

        Args:
            n: The number of trials

        Returns:
            A boolean representing whether or not there is available capacity.
        """
        return not_none(self.poll_available_capacity()) >= n

    def poll_available_capacity(self) -> Optional[int]:
        """Get the capacity remaining after accounting for staged and running
        trials, with the maximum being ``max_pending_trials``.

        Returns:
            The available capacity.
        """
        trials_by_status = self.experiment.trials_by_status
        num_staged = len(trials_by_status[TrialStatus.STAGED])
        num_running = len(trials_by_status[TrialStatus.RUNNING])
        capacity = self.max_pending_trials - (num_staged + num_running)
        return capacity

    def should_stop_trials_early(
        self, trial_indices: Set[int]
    ) -> Dict[int, Optional[str]]:
        """Given a set of trial indices, decide whether or not to early-stop
        running trials using the ``early_stopping_strategy``.

        Args:
            trial_indices: Indices of trials to consider for early stopping.

        Returns:
            Dict with new suggested ``TrialStatus`` as keys and a set of
            indices of trials to update (subset of initially-passed trials) as values.
        """
        # TODO: The status on the experiment does not distinguish between
        # running and queued trials, so here we check status on the
        # ``backend_simulator`` directly to make sure it is running.
        running_trials = set()
        skipped_trials = set()
        for trial_index in trial_indices:
            sim_trial = self.backend_simulator.get_sim_trial_by_index(trial_index)
            if sim_trial.sim_start_time is not None and (  # pyre-ignore[16]
                self.backend_simulator.time - sim_trial.sim_start_time > 0
            ):
                running_trials.add(trial_index)
            else:
                skipped_trials.add(trial_index)
        if len(skipped_trials) > 0:
            logger.info(
                f"Not sending {skipped_trials} to base `should_stop_trials_early` "
                "because they have not been running for a positive amount of time "
                "on the backend simulator."
            )
        return super().should_stop_trials_early(trial_indices=running_trials)
