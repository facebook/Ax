# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Mixed integer extensions of some common synthetic test functions.
These are adapted from [Daulton2022bopr]_.

References

.. [Daulton2022bopr]
    S. Daulton, X. Wan, D. Eriksson, M. Balandat, M. A. Osborne, E. Bakshy.
    Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic
    Reparameterization. Advances in Neural Information Processing Systems
    35, 2022.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.metrics.benchmark import BenchmarkMetric
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from botorch.test_functions.synthetic import (
    Ackley,
    Hartmann,
    Rosenbrock,
    SyntheticTestFunction,
)


def _get_problem_from_common_inputs(
    bounds: List[Tuple[float, float]],
    dim_int: int,
    metric_name: str,
    lower_is_better: bool,
    observe_noise_sd: bool,
    test_problem_class: Type[SyntheticTestFunction],
    benchmark_name: str,
    num_trials: int,
    test_problem_bounds: Optional[List[Tuple[float, float]]] = None,
) -> BenchmarkProblem:
    """This is a helper that deduplicates common bits of the below problems.

    Args:
        bounds: The parameter bounds.
        dim_int: The number of integer dimensions. First `dim_int` parameters
            are assumed to be integers.
        metric_name: The name of the metric.
        lower_is_better: If true, the goal is to minimize the metric.
        observe_noise_sd: Whether to report the standard deviation of the
            observation noise.
        test_problem_class: The BoTorch test problem class.
        benchmark_name: The name of the benchmark problem.
        num_trials: The number of trials.
        test_problem_bounds: Optional bounds to evaluate the base test problem on.
            These are passed in as `bounds` while initializing the test problem.

    Returns:
        A mixed-integer BenchmarkProblem constructed from the given inputs.
    """
    dim = len(bounds)
    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i + 1}",
                parameter_type=(
                    ParameterType.INT if i < dim_int else ParameterType.FLOAT
                ),
                lower=bounds[i][0],
                upper=bounds[i][1],
            )
            for i in range(dim)
        ]
    )
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(
                name=metric_name,
                lower_is_better=lower_is_better,
                observe_noise_sd=observe_noise_sd,
            ),
            minimize=True,
        )
    )
    test_problem_kwargs: Dict[str, Any] = {"dim": dim}
    if test_problem_bounds is not None:
        test_problem_kwargs["bounds"] = test_problem_bounds
    runner = BotorchTestProblemRunner(
        test_problem_class=test_problem_class,
        test_problem_kwargs=test_problem_kwargs,
        outcome_names=[metric_name],
        modified_bounds=bounds,
    )
    return BenchmarkProblem(
        name=benchmark_name + ("_observed_noise" if observe_noise_sd else ""),
        search_space=search_space,
        optimization_config=optimization_config,
        runner=runner,
        num_trials=num_trials,
        is_noiseless=True,
        observe_noise_sd=observe_noise_sd,
        has_ground_truth=True,
    )


def get_discrete_hartmann(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> BenchmarkProblem:
    """6D Hartmann problem where first 4 dimensions are discretized."""
    dim_int = 4
    if bounds is None:
        bounds = [
            (0, 3),
            (0, 3),
            (0, 19),
            (0, 19),
            (0.0, 1.0),
            (0.0, 1.0),
        ]
    return _get_problem_from_common_inputs(
        bounds=bounds,
        dim_int=dim_int,
        metric_name="Hartmann",
        lower_is_better=True,
        observe_noise_sd=observe_noise_sd,
        test_problem_class=Hartmann,
        benchmark_name="Discrete Hartmann",
        num_trials=num_trials,
    )


def get_discrete_ackley(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> BenchmarkProblem:
    """13D Ackley problem where first 10 dimensions are discretized.

    This also restricts Ackley evaluation bounds to [0, 1].
    """
    dim = 13
    dim_int = 10
    if bounds is None:
        bounds = [
            *[(0, 2)] * 5,
            *[(0, 4)] * 5,
            *[(0.0, 1.0)] * 3,
        ]
    return _get_problem_from_common_inputs(
        bounds=bounds,
        dim_int=dim_int,
        metric_name="Ackley",
        lower_is_better=True,
        observe_noise_sd=observe_noise_sd,
        test_problem_class=Ackley,
        benchmark_name="Discrete Ackley",
        num_trials=num_trials,
        test_problem_bounds=[(0.0, 1.0)] * dim,
    )


def get_discrete_rosenbrock(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> BenchmarkProblem:
    """10D Rosenbrock problem where first 6 dimensions are discretized."""
    dim_int = 6
    if bounds is None:
        bounds = [
            *[(0, 3)] * 6,
            *[(0.0, 1.0)] * 4,
        ]
    return _get_problem_from_common_inputs(
        bounds=bounds,
        dim_int=dim_int,
        metric_name="Rosenbrock",
        lower_is_better=True,
        observe_noise_sd=observe_noise_sd,
        test_problem_class=Rosenbrock,
        benchmark_name="Discrete Rosenbrock",
        num_trials=num_trials,
    )
