#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.metrics.noisy_function import NoisyFunctionMetric


class L2NormMetric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return np.sqrt((x**2).sum())
