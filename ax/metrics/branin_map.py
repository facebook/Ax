#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from random import random
from typing import Any, List, Optional

import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData
from ax.metrics.noisy_function_map import NoisyFunctionMapMetric
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.measurement.synthetic_functions import branin


TIMESTAMP_KWARGS = {"map_keys": ["timestamp"], "timestamp": [0, 1, 2]}
FIDELITY_KWARGS = {"map_keys": ["fidelity"], "fidelity": [0.1, 0.4, 0.7, 1.0]}


class BraninTimestampMapMetric(NoisyFunctionMapMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
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
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        return super().fetch_trial_data(
            trial=trial, noisy=noisy, **kwargs, **TIMESTAMP_KWARGS
        )

    def f(self, x: np.ndarray) -> float:
        x1, x2, t = x
        if self.rate is not None:
            weight = 1.0 + np.exp(-not_none(self.rate) * t)
        else:
            weight = 1.0
        return checked_cast(float, branin(x1=x1, x2=x2)) * weight


class BraninFidelityMapMetric(NoisyFunctionMapMetric):
    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        return super().fetch_trial_data(
            trial=trial, noisy=noisy, **kwargs, **FIDELITY_KWARGS
        )

    def f(self, x: np.ndarray) -> float:
        x1, x2, fidelity = x
        fidelity_penalty = random() * math.pow(1.0 - fidelity, 2.0)
        return checked_cast(float, branin(x1=x1, x2=x2)) - fidelity_penalty


class BraninIncrementalTimestampMapMetric(BraninTimestampMapMetric):
    def __init__(
        self,
        name: str,
        param_names: List[str],
        noise_sd: float = 0.0,
        lower_is_better: Optional[bool] = None,
        rate: Optional[float] = None,
    ) -> None:
        self.timestamp = 0
        super().__init__(
            name=name,
            param_names=param_names,
            noise_sd=noise_sd,
            lower_is_better=lower_is_better,
        )

    @classmethod
    def overwrite_existing_data(cls) -> bool:
        return False

    @classmethod
    def combine_with_last_data(cls) -> bool:
        return True

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        timestamp_kwargs = {"map_keys": ["timestamp"], "timestamp": [self.timestamp]}
        self.timestamp += 1
        return NoisyFunctionMapMetric.fetch_trial_data(
            self, trial=trial, noisy=noisy, **kwargs, **timestamp_kwargs
        )
