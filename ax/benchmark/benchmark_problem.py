# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import abc
from typing import Any, Dict, List, Optional, Type

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


def _get_name(
    test_problem: BaseTestProblem, infer_noise: bool, dim: Optional[int] = None
) -> str:
    """
    Get a string name describing the problem, in a format such as
    "hartmann_fixed_noise_6d" or "jenatton" (where the latter would
    not have fixed noise and have the default dimensionality).
    """
    base_name = f"{test_problem.__class__.__name__}"
    fixed_noise = "" if infer_noise else "_fixed_noise"
    dim_str = "" if dim is None else f"_{dim}d"
    return f"{base_name}{fixed_noise}{dim_str}"


class BenchmarkProblemBase(abc.ABC):
    """
    Specifies the interface any benchmark problem must adhere to.

    Subclasses include BenchmarkProblem, SurrogateBenchmarkProblem, and
    MOOSurrogateBenchmarkProblem.
    """

    name: str
    search_space: SearchSpace
    optimization_config: OptimizationConfig
    num_trials: int
    infer_noise: bool
    tracking_metrics: List[Metric]

    @abc.abstractproperty
    def runner(self) -> Runner:
        pass  # pragma: no cover


class BenchmarkProblem(Base, BenchmarkProblemBase):
    """Benchmark problem, represented in terms of Ax search space, optimization
    config, and runner.
    """

    def __init__(
        self,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        runner: Runner,
        num_trials: int,
        infer_noise: bool,
        tracking_metrics: Optional[List[Metric]] = None,
    ) -> None:
        self.name = name
        self.search_space = search_space
        self.optimization_config = optimization_config
        self._runner = runner
        self.num_trials = num_trials
        self.infer_noise = infer_noise
        self.tracking_metrics: List[Metric] = (
            [] if tracking_metrics is None else tracking_metrics
        )

    @property
    def runner(self) -> Runner:
        return self._runner

    @classmethod
    def from_botorch(
        cls,
        test_problem_class: Type[BaseTestProblem],
        test_problem_kwargs: Dict[str, Any],
        num_trials: int,
        infer_noise: bool = True,
    ) -> "BenchmarkProblem":
        """
        Create a BenchmarkProblem from a BoTorch BaseTestProblem using
        specialized Metrics and Runners. The test problem's result will be
        computed on the Runner and retrieved by the Metric.

        Args:
            test_problem_class: The BoTorch test problem class which will be
                used to define the `search_space`, `optimization_config`, and
                `runner`.
            test_problem_kwargs: Keyword arguments used to instantiate the
                `test_problem_class`.
            num_trials: Simply the `num_trials` of the `BenchmarkProblem`
                created.
            infer_noise: Whether noise will be inferred. This is separate from
                whether synthetic noise is added to the problem, which is
                controlled by the `noise_std` of the test problem.
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

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(test_problem, infer_noise, dim)

        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=BotorchTestProblemMetric(
                    name=name,
                    noise_sd=None if infer_noise else (test_problem.noise_std or 0),
                ),
                minimize=True,
            )
        )

        return cls(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
            ),
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


class SingleObjectiveBenchmarkProblem(BenchmarkProblem):
    """The most basic BenchmarkProblem, with a single objective and a known optimal
    value.
    """

    def __init__(
        self,
        optimal_value: float,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        runner: Runner,
        num_trials: int,
        infer_noise: bool,
        tracking_metrics: Optional[List[Metric]] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            tracking_metrics=tracking_metrics,
        )
        self.optimal_value = optimal_value

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

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(test_problem, infer_noise, dim)

        return cls(
            name=name,
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            optimal_value=test_problem.optimal_value,
        )


class MultiObjectiveBenchmarkProblem(BenchmarkProblem):
    """A BenchmarkProblem support multiple objectives. Rather than knowing each
    objective's optimal value we track a known maximum hypervolume computed from a
    given reference point.
    """

    def __init__(
        self,
        maximum_hypervolume: float,
        reference_point: List[float],
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        runner: Runner,
        num_trials: int,
        infer_noise: bool,
        tracking_metrics: Optional[List[Metric]] = None,
    ) -> None:
        self.maximum_hypervolume = maximum_hypervolume
        self.reference_point = reference_point
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            tracking_metrics=tracking_metrics,
        )

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

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(test_problem, infer_noise, dim)

        metrics = [
            BotorchTestProblemMetric(
                name=f"{name}_{i}",
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
                    bound=test_problem.ref_point[i].item(),
                    relative=False,
                    op=ComparisonOp.LEQ,
                )
                for i in range(test_problem.num_objectives)
            ],
        )

        return cls(
            name=name,
            search_space=problem.search_space,
            optimization_config=optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
            maximum_hypervolume=test_problem.max_hv,
            reference_point=test_problem._ref_point,
        )
