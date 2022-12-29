# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from ax.core.metric import Metric

from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.metrics.botorch_test_problem import BotorchTestProblemMetric
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.utils.common.base import Base
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction

# NOTE: Do not add `from __future__ import annotatations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkProblem` as return type annotation, used for serialization and rendering
# in the UI.


@dataclass(frozen=True)
class BenchmarkProblem(Base):
    """Benchmark problem, represented in terms of Ax search space, optimization
    config, and runner.
    """

    name: str
    search_space: SearchSpace
    optimization_config: OptimizationConfig
    runner: Runner
    num_trials: int
    infer_noise: bool
    tracking_metrics: List[Metric] = field(default_factory=list)

    @classmethod
    def from_botorch(
        cls,
        test_problem_class: Type[BaseTestProblem],
        test_problem_kwargs: Dict[str, Any],
        num_trials: int,
        infer_noise: bool = True,
    ) -> "BenchmarkProblem":
        """Create a BenchmarkProblem from a BoTorch BaseTestProblem using specialized
        Metrics and Runners. The test problem's result will be computed on the Runner
        and retrieved by the Metric.
        """

        # pyre-fixme [45]: Invalid class instantiation
        test_problem = test_problem_class(**test_problem_kwargs)

        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=test_problem._bounds[i][0],
                    upper=test_problem._bounds[i][1],
                )
                for i in range(test_problem.dim)
            ]
        )

        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=BotorchTestProblemMetric(
                    name=f"{test_problem.__class__.__name__}",
                    noise_sd=None if infer_noise else (test_problem.noise_std or 0),
                ),
                minimize=True,
            )
        )

        return cls(
            name=f"{test_problem.__class__.__name__}",
            search_space=search_space,
            optimization_config=optimization_config,
            runner=BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
            ),
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


@dataclass(frozen=True, init=False)
class SingleObjectiveBenchmarkProblem(BenchmarkProblem):
    """The most basic BenchmarkProblem, with a single objective and a known optimal
    value.
    """

    optimal_value: float = field()

    def __init__(self, optimal_value: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "optimal_value", optimal_value)

    @classmethod
    def from_botorch_synthetic(
        cls,
        test_problem_class: Type[SyntheticTestFunction],
        test_problem_kwargs: Dict[str, Any],
        num_trials: int,
        infer_noise: bool = True,
    ) -> "SingleObjectiveBenchmarkProblem":
        """Create a BenchmarkProblem from a BoTorch BaseTestProblem using specialized
        Metrics and Runners. The test problem's result will be computed on the Runner
        and retrieved by the Metric.
        """

        # pyre-fixme [45]: Invalid class instantiation
        test_problem = test_problem_class(**test_problem_kwargs)

        problem = BenchmarkProblem.from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )

        return cls(
            name=f"{test_problem.__class__.__name__}",
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            optimal_value=test_problem.optimal_value,
        )


@dataclass(frozen=True, init=False)
class MultiObjectiveBenchmarkProblem(BenchmarkProblem):
    """A BenchmarkProblem support multiple objectives. Rather than knowing each
    objective's optimal value we track a known maximum hypervolume computed from a
    given reference point.
    """

    maximum_hypervolume: float = field()
    reference_point: List[float] = field()

    def __init__(
        self,
        maximum_hypervolume: float,
        reference_point: List[float],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "maximum_hypervolume", maximum_hypervolume)
        object.__setattr__(self, "reference_point", reference_point)

    @classmethod
    def from_botorch_multi_objective(
        cls,
        test_problem_class: Type[MultiObjectiveTestProblem],
        test_problem_kwargs: Dict[str, Any],
        num_trials: int,
        infer_noise: bool = True,
    ) -> "MultiObjectiveBenchmarkProblem":
        """Create a BenchmarkProblem from a BoTorch BaseTestProblem using specialized
        Metrics and Runners. The test problem's result will be computed on the Runner
        once per trial and each Metric will retrieve its own result by index.
        """

        # pyre-fixme [45]: Invalid class instantiation
        test_problem = test_problem_class(**test_problem_kwargs)

        problem = BenchmarkProblem.from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )

        metrics = [
            BotorchTestProblemMetric(
                name=f"{test_problem.__class__.__name__}_{i}",
                noise_sd=None if infer_noise else (test_problem.noise_std or 0),
                index=i,
            )
            for i in range(test_problem.num_objectives)
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(
                        metric=metric,
                        minimize=True,
                    )
                    for metric in metrics
                ]
            ),
            objective_thresholds=[
                ObjectiveThreshold(
                    metric=metrics[i],
                    # pyre-fixme[6]: For 2nd param expected `float` but got `Tensor`.
                    bound=test_problem.ref_point[i],
                    relative=False,
                    op=ComparisonOp.LEQ,
                )
                for i in range(test_problem.num_objectives)
            ],
        )

        return cls(
            name=f"{test_problem.__class__.__name__}",
            search_space=problem.search_space,
            optimization_config=optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            maximum_hypervolume=test_problem.max_hv,
            reference_point=test_problem._ref_point,
        )
