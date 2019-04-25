#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric


class L2NormMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return np.sqrt((x ** 2).sum())
