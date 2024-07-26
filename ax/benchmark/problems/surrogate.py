# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import List

from ax.benchmark.benchmark_problem import BenchmarkProblem

from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.optimization_config import MultiObjectiveOptimizationConfig


@dataclass(kw_only=True)
class SurrogateBenchmarkProblemBase(BenchmarkProblem):
    """
    Base class for SOOSurrogateBenchmarkProblem and MOOSurrogateBenchmarkProblem.

    Its `runner` is a `SurrogateRunner`, which allows for the surrogate to be
    constructed lazily and datasets to be downloaded lazily.
    """

    runner: SurrogateRunner = field(repr=False)


class SOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    pass


@dataclass(kw_only=True)
class MOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `MultiObjectiveBenchmarkProblem`,
    but its runner is not constructed until needed, to allow for deferring
    constructing the surrogate and downloading data.
    """

    optimization_config: MultiObjectiveOptimizationConfig
    reference_point: List[float]
