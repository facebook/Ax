# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Optional

import torch
from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.runners.botorch_test import (
    ParamBasedTestProblem,
    ParamBasedTestProblemRunner,
)
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import HierarchicalSearchSpace
from ax.core.types import TParameterization
from pyre_extensions import none_throws


def jenatton_test_function(
    x1: Optional[int] = None,
    x2: Optional[int] = None,
    x3: Optional[int] = None,
    x4: Optional[float] = None,
    x5: Optional[float] = None,
    x6: Optional[float] = None,
    x7: Optional[float] = None,
    r8: Optional[float] = None,
    r9: Optional[float] = None,
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
class Jenatton(ParamBasedTestProblem):
    """Jenatton test function for hierarchical search spaces."""

    noise_std: Optional[float] = None
    negate: bool = False
    num_objectives: int = 1
    optimal_value: float = 0.1
    _is_constrained: bool = False

    def evaluate_true(self, params: TParameterization) -> torch.Tensor:
        # pyre-fixme: Incompatible parameter type [6]: In call
        # `jenatton_test_function`, for 1st positional argument, expected
        # `Optional[float]` but got `Union[None, bool, float, int, str]`.
        value = jenatton_test_function(**params)
        return torch.tensor(value)


def get_jenatton_benchmark_problem(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
    noise_std: float = 0.0,
) -> BenchmarkProblem:
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
    name = "Jenatton" + ("_observed_noise" if observe_noise_sd else "")

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(
                name=name, observe_noise_sd=observe_noise_sd, lower_is_better=True
            ),
            minimize=True,
        )
    )
    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=ParamBasedTestProblemRunner(
            test_problem_class=Jenatton,
            test_problem_kwargs={"noise_std": noise_std},
            outcome_names=[name],
        ),
        num_trials=num_trials,
        observe_noise_stds=observe_noise_sd,
        optimal_value=Jenatton.optimal_value,
    )
