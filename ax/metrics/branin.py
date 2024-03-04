#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import aug_branin, branin


class BraninMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        x1, x2 = x
        return checked_cast(float, branin(x1=x1, x2=x2))


class NegativeBraninMetric(BraninMetric):
    def f(self, x: np.ndarray) -> float:
        fpos = super().f(x)
        return -fpos


class AugmentedBraninMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return checked_cast(float, aug_branin(x))
