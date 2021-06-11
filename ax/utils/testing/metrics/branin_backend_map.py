#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import numpy as np
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import branin
from ax.utils.testing.metrics.backend_simulator_map import (
    BackendSimulatorTimestampMapMetric,
)

logger = get_logger(__name__)


class BraninBackendMapMetric(BackendSimulatorTimestampMapMetric):
    """A Branin ``BackendSimulatorTimestampMapMetric`` with a multiplicative
    factor of ``1 - exp(-rate * t)`` where ``t`` is the runtime of the trial."""

    def __init__(
        self,
        name: str,
        param_names: List[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = True,
        cache_evaluations: bool = True,
        rate: float = 0.5,
        delta_t: float = 1.0,
    ) -> None:
        """Branin metric with multiplicative factor ``1 - exp(-rate * t)``.

        Args:
            name: Name of the metric.
            param_names: An ordered list of names of parameters to be passed
                to the deterministic function.
            noise_sd: Scale of normal noise added to the function result.
            lower_is_better: Flag for metrics which should be minimized.
            rate: Parameter of the multiplicative factor.
            delta_t: The time delta between intermediate results, used in
                ``convert_to_timestamps``.
        """
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
            cache_evaluations=cache_evaluations,
        )
        self.rate = rate
        self.delta_t = delta_t

    def f(self, x: np.ndarray) -> float:
        """The standard Branin function multiplied by ``1 - exp(-rate * t)``."""
        x1, x2, t = x
        weight = 1.0 - np.exp(-self.rate * t)
        return checked_cast(float, branin(x1=x1, x2=x2)) * weight

    def convert_to_timestamps(
        self, start_time: Optional[float], end_time: float
    ) -> List[float]:
        """Given a starting and current time, get the list of intermediate
        timestamps at which we have observations."""
        if start_time is None:
            # NOTE: This can be the case for trials on backend simulator
            # that are queued.
            return []
        num_periods_running = (end_time - start_time) // self.delta_t
        return list(np.arange(num_periods_running) * self.delta_t)
