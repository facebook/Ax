#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric


class Hartmann6Metric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return y
