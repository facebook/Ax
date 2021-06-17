#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData
from ax.metrics.noisy_function_map import NoisyFunctionMapMetric


class BackendSimulatorTimestampMapMetric(NoisyFunctionMapMetric):
    """A metric that interfaces with an underlying ``BackendSimulator`` and
    returns timestamp map data."""

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        """Fetch data for one trial."""
        backend_simulator = trial.experiment.runner.simulator  # pyre-ignore[16]
        sim_trial = backend_simulator.get_sim_trial_by_index(trial.index)
        end_time = (
            backend_simulator.time
            if sim_trial.sim_completed_time is None
            else sim_trial.sim_completed_time
        )
        timestamps = self.convert_to_timestamps(
            start_time=sim_trial.sim_start_time, end_time=end_time
        )
        timestamp_kwargs = {"map_keys": ["timestamp"], "timestamp": timestamps}
        return NoisyFunctionMapMetric.fetch_trial_data(
            self, trial=trial, noisy=noisy, **kwargs, **timestamp_kwargs
        )

    def convert_to_timestamps(self, start_time: float, end_time: float) -> List[float]:
        """Given a starting and current time, get the list of intermediate
        timestamps at which we have observations."""
        raise NotImplementedError
