#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Callable

import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.utils.testing.backend_simulator import BackendSimulator


class SimulatedBackendRunner(Runner):
    """Class for a runner that works with the BackendSimulator."""

    def __init__(
        self,
        simulator: BackendSimulator,
        sample_runtime_func: Optional[Callable[[BaseTrial], float]] = None,
    ) -> None:
        """Runner for a BackendSimulator.

        Args:
            simulator: The backend simulator.
            sample_runtime_func: A Callable that samples a runtime given a trial.
        """
        self.simulator = simulator
        if sample_runtime_func is None:
            sample_runtime_func = sample_runtime_unif
        self.sample_runtime_func: Callable[[BaseTrial], float] = sample_runtime_func

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Start a trial on the BackendSimulator.

        Args:
            trial: Trial to deploy via the runner.

        Returns:
            Dict containing the sampled runtime of the trial.
        """
        runtime = self.sample_runtime_func(trial)
        self.simulator.run_trial(trial_index=trial.index, runtime=runtime)
        return {"runtime": runtime}

    def stop(self, trial: BaseTrial, reason: Optional[str] = None) -> Dict[str, Any]:
        """Stop a trial on the BackendSimulator.

        Args:
            trial: Trial to stop on the simulator.
            reason: A message containing information why the trial is to be stopped.

        Returns:
            A dictionary containing a single key "reason" that maps to the reason
            passed to the function. If no reason was given, returns an empty dictionary.
        """
        self.simulator.stop_trial(trial.index)
        return {"reason": reason} if reason else {}


def sample_runtime_unif(trial: BaseTrial, low: float = 1.0, high: float = 5.0) -> float:
    """Return a uniform runtime in [low, high]

    Args:
        trial: Trial for which to sample runtime.
        low: Lower bound of uniform runtime distribution.
        high: Upper bound of uniform runtime distribution.

    Returns:
        A float representing the simulated trial runtime.
    """
    return np.random.uniform(low, high)
