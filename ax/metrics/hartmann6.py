#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy.typing as npt
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.utils.measurement.synthetic_functions import aug_hartmann6, hartmann6
from pyre_extensions import assert_is_instance


class Hartmann6Metric(NoisyFunctionMetric):
    def f(self, x: npt.NDArray) -> float:
        return assert_is_instance(hartmann6(x), float)


class AugmentedHartmann6Metric(NoisyFunctionMetric):
    def f(self, x: npt.NDArray) -> float:
        return assert_is_instance(aug_hartmann6(x), float)
