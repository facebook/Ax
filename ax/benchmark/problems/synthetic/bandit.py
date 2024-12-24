# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from warnings import warn

import numpy as np

from ax.benchmark.benchmark_problem import BenchmarkProblem, get_soo_opt_config
from ax.benchmark.benchmark_test_functions.synthetic import IdentityTestFunction
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace


def get_baseline(num_choices: int, n_sims: int = 100000000) -> float:
    """
    Compute the baseline value.

    The baseline for this problem takes into account noise, because it uses the
    inference trace, and the bandit structure, which allows for running all arms
    in one noisy batch:

    Run a BatchTrial with every arm, with equal size.  Choose the arm with the
    best observed value and take its true value.  Take the expectation of the
    outcome of this process.
    """
    noise_per_arm = num_choices**0.5
    sim_observed_effects = (
        np.random.normal(0, noise_per_arm, (n_sims, num_choices))
        + np.arange(num_choices)[None, :]
    )
    identified_best_arm = sim_observed_effects.argmin(axis=1)
    # because of the use of IdentityTestFunction
    baseline = identified_best_arm.mean()
    return baseline


def get_bandit_problem(num_choices: int = 30, num_trials: int = 3) -> BenchmarkProblem:
    parameter = ChoiceParameter(
        name="x0",
        parameter_type=ParameterType.INT,
        values=list(range(num_choices)),
        is_ordered=False,
        sort_values=False,
    )
    search_space = SearchSpace(parameters=[parameter])
    test_function = IdentityTestFunction()
    optimization_config = get_soo_opt_config(
        outcome_names=test_function.outcome_names, observe_noise_sd=True
    )
    baselines = {
        10: 1.40736478,
        30: 2.4716703,
        100: 4.403284,
    }
    if num_choices not in baselines:
        warn(
            f"Baseline value is not available for num_choices={num_choices}. Use "
            "`get_baseline` to compute the baseline and add it to `baselines`."
        )
        baseline_value = baselines[30]
    else:
        baseline_value = baselines[num_choices]
    return BenchmarkProblem(
        name="Bandit",
        num_trials=num_trials,
        search_space=search_space,
        optimization_config=optimization_config,
        optimal_value=0,
        baseline_value=baseline_value,
        test_function=test_function,
        report_inference_value_as_trace=True,
        noise_std=1.0,
        status_quo_params={"x0": num_choices // 2},
    )
