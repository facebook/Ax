#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from random import random
from typing import Any

import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData
from ax.metrics.noisy_function_map import NoisyFunctionMapMetric
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import branin


TIMESTAMP_KWARGS = {"map_keys": ["timestamp"], "timestamp": [0, 1, 2]}
FIDELITY_KWARGS = {"map_keys": ["fidelity"], "fidelity": [0.1, 0.4, 0.7, 1.0]}


class BraninTimestampMapMetric(NoisyFunctionMapMetric):
    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> MapData:
        return super().fetch_trial_data(
            trial=trial, noisy=noisy, **kwargs, **TIMESTAMP_KWARGS
        )

    def f(self, x: np.ndarray) -> float:
        x1, x2, timestamp = x
        return checked_cast(float, branin(x1=x1, x2=x2))


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
