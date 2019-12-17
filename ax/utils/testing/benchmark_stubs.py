#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from types import FunctionType
from typing import Any, cast

from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.utils.measurement.synthetic_functions import branin
from ax.utils.testing.core_stubs import (
    get_branin_optimization_config,
    get_branin_search_space,
)


def get_branin_simple_benchmark_problem() -> SimpleBenchmarkProblem:
    return SimpleBenchmarkProblem(f=branin)


def get_sum_simple_benchmark_problem() -> SimpleBenchmarkProblem:
    return SimpleBenchmarkProblem(f=sum, name="Sum", domain=[(0.0, 1.0), (0.0, 1.0)])


def sample_multiplication_fxn(*args: Any) -> float:
    return reduce(lambda x, y: x * y, args)


def get_mult_simple_benchmark_problem() -> SimpleBenchmarkProblem:
    return SimpleBenchmarkProblem(
        f=cast(FunctionType, sample_multiplication_fxn),
        name="Sum",
        domain=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )


def get_branin_benchmark_problem() -> BenchmarkProblem:
    return BenchmarkProblem(
        search_space=get_branin_search_space(),
        optimization_config=get_branin_optimization_config(),
        optimal_value=branin.fmin,
        evaluate_suggested=False,
    )
