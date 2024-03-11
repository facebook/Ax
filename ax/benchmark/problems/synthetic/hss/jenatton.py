# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.benchmark.metrics.jenatton import JenattonMetric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import HierarchicalSearchSpace
from ax.runners.synthetic import SyntheticRunner


def get_jenatton_benchmark_problem(
    num_trials: int = 50,
    observe_noise_sd: bool = False,
) -> SingleObjectiveBenchmarkProblem:
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

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=JenattonMetric(observe_noise_sd=observe_noise_sd),
            minimize=True,
        )
    )

    name = "Jenatton" + ("_observed_noise" if observe_noise_sd else "")

    return SingleObjectiveBenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
        num_trials=num_trials,
        is_noiseless=True,
        observe_noise_sd=observe_noise_sd,
        has_ground_truth=True,
        optimal_value=0.1,
    )
