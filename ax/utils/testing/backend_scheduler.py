#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace as dataclass_replace

from logging import Logger
from typing import Dict, Optional, Set

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.runners.simulated_backend import SimulatedBackendRunner
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.utils.common.logger import get_logger
from ax.utils.testing.backend_simulator import BackendSimulator

logger: Logger = get_logger(__name__)


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
        if (
            options.max_pending_trials is not None
            and options.max_pending_trials != max_pending_trials
        ):
            raise ValueError(
                f"`SchedulerOptions.max_pending_trials`: {options.max_pending_trials} "
                f"does not match argument to `Scheduler`: {max_pending_trials}."
            )
        if options.max_pending_trials is None:
            options = dataclass_replace(options, max_pending_trials=max_pending_trials)

        super().__init__(
            experiment=experiment,
            generation_strategy=generation_strategy,
            options=options,
            _skip_experiment_save=True,
        )

    @property
    def backend_simulator(self) -> BackendSimulator:
        """Get the ``BackendSimulator`` stored on the runner of the experiment.

        Returns:
            The backend simulator.
        """
        return self.runner.simulator  # pyre-ignore[16]

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
