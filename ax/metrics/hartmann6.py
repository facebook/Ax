#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import hartmann6


class Hartmann6Metric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return checked_cast(float, hartmann6(x))
