# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from ax.benchmark.benchmark_problem import BenchmarkProblem, get_soo_opt_config
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import HierarchicalSearchSpace
from pyre_extensions import none_throws

JENATTON_OPTIMAL_VALUE = 0.1
# Baseline value was obtained with `compute_baseline_value_from_sobol`
JENATTON_BASELINE_VALUE = 0.5797074938368603


def jenatton_test_function(
    x1: int | None = None,
    x2: int | None = None,
    x3: int | None = None,
    x4: float | None = None,
    x5: float | None = None,
    x6: float | None = None,
    x7: float | None = None,
    r8: float | None = None,
    r9: float | None = None,
) -> float:
    """Jenatton test function for hierarchical search spaces.

    This function is taken from:

    R. Jenatton, C. Archambeau, J. González, and M. Seeger. Bayesian
    optimization with tree-structured dependencies. ICML 2017.
    """
    if x1 == 0:
        if x2 == 0:
            return none_throws(x4) ** 2 + 0.1 + none_throws(r8)
        return none_throws(x5) ** 2 + 0.2 + none_throws(r8)
    if x3 == 0:
        return none_throws(x6) ** 2 + 0.3 + none_throws(r9)
    return none_throws(x7) ** 2 + 0.4 + none_throws(r9)


@dataclass(kw_only=True)
class Jenatton(BenchmarkTestFunction):
    """Jenatton test function for hierarchical search spaces."""

    # pyre-fixme[14]: Inconsistent override
    def evaluate_true(self, params: Mapping[str, float | int | None]) -> torch.Tensor:
        # pyre-fixme: Incompatible parameter type [6]: In call
        # `jenatton_test_function`, for 1st positional argument, expected
        # `Optional[float]` but got `Union[None, bool, float, int, str]`.
        value = jenatton_test_function(**params)
        return torch.tensor(value, dtype=torch.double)


def get_jenatton_search_space() -> HierarchicalSearchSpace:
    search_space = HierarchicalSearchSpace(
        parameters=[
            ChoiceParameter(
                name="x1",
                parameter_type=ParameterType.INT,
                values=[0, 1],
                dependents={0: ["x2", "r8"], 1: ["x3", "r9"]},
            ),
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.INT,
                values=[0, 1],
                dependents={0: ["x4"], 1: ["x5"]},
            ),
            ChoiceParameter(
                name="x3",
                parameter_type=ParameterType.INT,
                values=[0, 1],
                dependents={0: ["x6"], 1: ["x7"]},
            ),
            *[
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                )
                for i in range(4, 8)
            ],
            RangeParameter(
                name="r8", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            ),
            RangeParameter(
                name="r9", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            ),
        ]
    )
    return search_space


def get_jenatton_benchmark_problem(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    noise_std: float = 0.0,
) -> BenchmarkProblem:
    name = "Jenatton" + ("_observed_noise" if observe_noise_sd else "")
    optimization_config = get_soo_opt_config(
        outcome_names=[name], observe_noise_sd=observe_noise_sd
    )

    return BenchmarkProblem(
        name=name,
        search_space=get_jenatton_search_space(),
        optimization_config=optimization_config,
        test_function=Jenatton(outcome_names=[name]),
        noise_std=noise_std,
        num_trials=num_trials,
        optimal_value=JENATTON_OPTIMAL_VALUE,
        baseline_value=JENATTON_BASELINE_VALUE,
    )
