#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from random import random
from typing import Mapping, Iterable, Any, Optional

import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapKeyInfo, MapData
from ax.metrics.noisy_function_map import NoisyFunctionMapMetric
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.measurement.synthetic_functions import branin

FIDELITY = [0.1, 0.4, 0.7, 1.0]


class BraninTimestampMapMetric(NoisyFunctionMapMetric):
    def __init__(
        self,
        name: str,
        param_names: Iterable[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
        rate: Optional[float] = None,
    ) -> None:
        """A Branin map metric with an optional multiplicative factor
        of `1 + exp(-rate * t)` where `t` is the runtime of the trial.
        If the multiplicative factor is used, then at `t = 0`, the function
        is twice the usual value, while as `t` becomes large, the values
        approach the standard Branin values.

        Args:
            name: Name of the metric.
            param_names: An ordered list of names of parameters to be passed
                to the deterministic function.
            noise_sd: Scale of normal noise added to the function result.
            lower_is_better: Flag for metrics which should be minimized.
            rate: Parameter of the multiplicative factor.
        """
        self.rate = rate
        self._timestamp = -1

        super().__init__(
            name=name,
            param_names=param_names,
            map_key_infos=[MapKeyInfo(key="timestamp", default_value=0.0)],
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )

    def __eq__(self, o: BraninTimestampMapMetric) -> bool:
        """Ignore _timestamp on equality checks"""
        return (
            self.name == o.name
            and self.param_names == o.param_names
            and self.map_key_infos == o.map_key_infos
            and self.noise_sd == o.noise_sd
            and self.lower_is_better == o.lower_is_better
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        # This timestamp parameter will be incremented each time f is called to
        # simulate a true timestamp.
        self._timestamp = -1

        s = super()  # Must assign super() to capture outer scope inside comprehension
        rows = [
            s.fetch_trial_data(
                trial=trial,
                noisy=noisy,
                **kwargs,
            )
            for _ in range(3)
        ]

        return MapData.from_multiple_map_data(rows)

    def f(self, x: np.ndarray) -> Mapping[str, Any]:
        self._timestamp += 1

        x1, x2 = x

        if self.rate is not None:
            weight = 1.0 + np.exp(-not_none(self.rate) * self._timestamp)
        else:
            weight = 1.0

        mean = checked_cast(float, branin(x1=x1, x2=x2)) * weight

        return {"mean": mean, "timestamp": self._timestamp}


class BraninFidelityMapMetric(NoisyFunctionMapMetric):
    def __init__(
        self,
        name: str,
        param_names: Iterable[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
    ) -> None:
        super().__init__(
            name=name,
            param_names=param_names,
            map_key_infos=[MapKeyInfo(key="fidelity", default_value=0.0)],
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )

        self.index = -1

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        self.index = -1

        return super().fetch_trial_data(
            trial=trial,
            noisy=noisy,
            **kwargs,
        )

    def f(self, x: np.ndarray) -> Mapping[str, Any]:
        if self.index < len(FIDELITY):
            self.index += 1

        x1, x2 = x
        fidelity = FIDELITY[self.index]

        fidelity_penalty = random() * math.pow(1.0 - fidelity, 2.0)
        mean = checked_cast(float, branin(x1=x1, x2=x2)) - fidelity_penalty

        return {"mean": mean, "fidelity": fidelity}
