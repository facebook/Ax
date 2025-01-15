#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy.typing as npt
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.utils.measurement.synthetic_functions import aug_branin, branin
from pyre_extensions import assert_is_instance


class BraninMetric(NoisyFunctionMetric):
    def f(self, x: npt.NDArray) -> float:
        x1, x2 = x
        return assert_is_instance(branin(x1=x1, x2=x2), float)


class NegativeBraninMetric(BraninMetric):
    def f(self, x: npt.NDArray) -> float:
        fpos = super().f(x)
        return -fpos


class AugmentedBraninMetric(NoisyFunctionMetric):
    def f(self, x: npt.NDArray) -> float:
        return assert_is_instance(aug_branin(x), float)
