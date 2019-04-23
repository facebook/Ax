#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric


def branin(x1: float, x2: float) -> float:
    """Branin synthetic function."""
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y


class BraninMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        x1, x2 = x
        return branin(x1=x1, x2=x2)


class NegativeBraninMetric(BraninMetric):
    def f(self, x: np.ndarray) -> float:
        fpos = super().f(x)
        return -fpos
