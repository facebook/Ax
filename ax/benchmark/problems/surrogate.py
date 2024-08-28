# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
Benchmark problem based on surrogate.

This problem class might appear to function identically to its non-surrogate
counterpart, `BenchmarkProblem`, aside from the restriction that its runners is
of type `SurrogateRunner`. However, it is treated specially within JSON storage
because surrogates cannot be easily serialized.
"""

from dataclasses import dataclass, field

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.runners.surrogate import SurrogateRunner


@dataclass(kw_only=True)
class SurrogateBenchmarkProblem(BenchmarkProblem):
    """
    Benchmark problem whose `runner` is a `SurrogateRunner`.

    `SurrogateRunner` allows for the surrogate to be constructed lazily and for
    datasets to be downloaded lazily.

    For argument descriptions, see `BenchmarkProblem`.
    """

    runner: SurrogateRunner = field(repr=False)
