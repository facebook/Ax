# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import TypeVar

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace

TProblem = TypeVar("TProblem", bound=BenchmarkProblem)


def embed_higher_dimension(problem: TProblem, total_dimensionality: int) -> TProblem:
    """
    Return a new `BenchmarkProblem` with enough `RangeParameter`s added to the
    search space to make its total dimensionality equal to `total_dimensionality`
    and add `total_dimensionality` to its name.

    The search space of the original `problem` is within the search space of the
    new problem, and the constraints are copied from the original problem.
    """
    num_dummy_dimensions = total_dimensionality - len(problem.search_space.parameters)

    search_space = SearchSpace(
        parameters=[
            *problem.search_space.parameters.values(),
            *[
                RangeParameter(
                    name=f"embedding_dummy_{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                )
                for i in range(num_dummy_dimensions)
            ],
        ],
        parameter_constraints=problem.search_space.parameter_constraints,
    )

    # if problem name already has dimensionality in it, strip it
    def _is_dim_suffix(s: str) -> bool:
        return s[-1] == "d" and all(char in "0123456789" for char in s[:-1])

    orig_name_without_dimensionality = "_".join(
        [substr for substr in problem.name.split("_") if not _is_dim_suffix(substr)]
    )
    new_name = f"{orig_name_without_dimensionality}_{total_dimensionality}d"

    new_problem = copy.copy(problem)
    new_problem.name = new_name
    new_problem.search_space = search_space
    return new_problem
