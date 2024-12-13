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

from ax.benchmark.benchmark_problem import BenchmarkProblem, get_soo_opt_config
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from botorch.test_functions.synthetic import Ackley, Hartmann, Rosenbrock


def _get_problem_from_common_inputs(
    *,
    bounds: list[tuple[float, float]],
    dim_int: int,
    metric_name: str,
    lower_is_better: bool,
    observe_noise_sd: bool,
    test_problem_class: type[Hartmann | Ackley | Rosenbrock],
    benchmark_name: str,
    num_trials: int,
    optimal_value: float,
    baseline_value: float,
    test_problem_bounds: list[tuple[float, float]] | None = None,
) -> BenchmarkProblem:
    """This is a helper that deduplicates common bits of the below problems.

    Args:
        bounds: The parameter bounds. These will be passed to
            `BotorchTestFunction` as `modified_bounds`, and the parameters
            will be renormalized from these bounds to the bounds of the original
            problem. For example, if `bounds` are [(0, 3)] and the test
            problem's original bounds are [(0, 2)], then the original problem
            can be evaluated at 0, 2/3, 4/3, and 2.
        dim_int: The number of integer dimensions. First `dim_int` parameters
            are assumed to be integers.
        metric_name: The name of the metric.
        lower_is_better: If true, the goal is to minimize the metric.
        observe_noise_sd: Whether to report the standard deviation of the
            observation noise.
        test_problem_class: The BoTorch test problem class.
        benchmark_name: The name of the benchmark problem.
        num_trials: The number of trials.
        optimal_value: Best attainable value, if known. If unknown, as may be
            the case for mixed-integer problems, choose a value that is known to
            be at least as good as the true optimum, to prevent benchmarks from
            attaining scores of over 100%. One strategy for choosing this value
            is to choose the overall optimum of the problem without regard to
            the integer restrictions.
        baseline_value: Will be passed to the `BenchmarkProblem`.
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
    optimization_config = get_soo_opt_config(
        outcome_names=[metric_name],
        lower_is_better=lower_is_better,
        observe_noise_sd=observe_noise_sd,
    )

    if test_problem_bounds is None:
        test_problem = test_problem_class(dim=dim)
    else:
        test_problem = test_problem_class(dim=dim, bounds=test_problem_bounds)
    test_function = BoTorchTestFunction(
        botorch_problem=test_problem,
        modified_bounds=bounds,
        outcome_names=[metric_name],
    )
    return BenchmarkProblem(
        name=benchmark_name + ("_observed_noise" if observe_noise_sd else ""),
        search_space=search_space,
        optimization_config=optimization_config,
        test_function=test_function,
        num_trials=num_trials,
        optimal_value=optimal_value,
        baseline_value=baseline_value,
    )


def get_discrete_hartmann(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: list[tuple[float, float]] | None = None,
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
        # The best value we've found so far on this mixed problem is -2.89. The
        # optimum without regards to the integer constraints is -3.3224, but
        # that won't be attainable here.
        optimal_value=-3.0,
        # Baseline values were obtained with `compute_baseline_value_from_sobol`
        baseline_value=-0.6755184773211834,
    )


def get_discrete_ackley(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: list[tuple[float, float]] | None = None,
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
        # Ackley's lowest value is at (0, 0, ..., 0), which is in the search
        # space, so the restriction to integers doesn't change the optimum
        optimal_value=0.0,
        baseline_value=3.1869268815137968,
    )


def get_discrete_rosenbrock(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    bounds: list[tuple[float, float]] | None = None,
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
        # Rosenbrock's lowest value is at (1, 1, ..., 1), which is in the search
        # space, so the restriction to integers doesn't change the optimum
        optimal_value=0.0,
        baseline_value=705714.2851460224,
    )
