# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace


def embed_higher_dimension(
    problem: BenchmarkProblem, total_dimensionality: int
) -> BenchmarkProblem:
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

    problem_kwargs = asdict(problem)
    problem_kwargs["name"] = f"{problem_kwargs['name']}_{total_dimensionality}d"
    problem_kwargs["search_space"] = search_space

    return problem.__class__(**problem_kwargs)
