# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
Benchmark problems based on surrogates.

These problems might appear to function identically to their non-surrogate
counterparts, `BenchmarkProblem` and `MultiObjectiveBenchmarkProblem`, aside
from the restriction that their runners are of type `SurrogateRunner`. However,
they are treated specially within JSON storage because surrogates cannot be
easily serialized.
"""

from dataclasses import dataclass, field

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.optimization_config import MultiObjectiveOptimizationConfig


@dataclass(kw_only=True)
class SurrogateBenchmarkProblemBase(BenchmarkProblem):
    """
    Base class for SOOSurrogateBenchmarkProblem and MOOSurrogateBenchmarkProblem.

    Its `runner` is a `SurrogateRunner`, which allows for the surrogate to be
    constructed lazily and datasets to be downloaded lazily.

    For argument descriptions, see `BenchmarkProblem`.
    """

    runner: SurrogateRunner = field(repr=False)


class SOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    pass


@dataclass(kw_only=True)
class MOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `MultiObjectiveBenchmarkProblem`,
    but its `runner` is a `SurrogateRunner`, which allows for the surrogate to be
    constructed lazily and datasets to be downloaded lazily.

    For argument descriptions, see `BenchmarkProblem`.
    """

    optimization_config: MultiObjectiveOptimizationConfig
