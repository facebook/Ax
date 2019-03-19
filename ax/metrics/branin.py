#!/usr/bin/env python3

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric


class BraninMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        x1, x2 = x
        y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
        return y


class BraninConstraintMetric(NoisyFunctionMetric):
    """Constraint is f(x) <= 0"""

    def f(self, x: np.ndarray) -> float:
        return (x[0] - 2.5) ** 2 + (x[1] - 7.5) ** 2 - 50


class NegativeBraninMetric(BraninMetric):
    def f(self, x: np.ndarray) -> float:
        fpos = super().f(x)
        return -fpos
