# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# NOTE: Do not add `from __future__ import annotations` to this file. Adding
# `annotations` postpones evaluation of types and will break FBLearner's usage of
# `BenchmarkProblem` as return type annotation, used for serialization and rendering
# in the UI.

import abc
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Type, Union

from ax.benchmark.metrics.base import BenchmarkMetricBase

from ax.benchmark.metrics.benchmark import BenchmarkMetric
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.utils.common.base import Base
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction


def _get_name(
    test_problem: BaseTestProblem,
    observe_noise_sd: bool,
    dim: Optional[int] = None,
) -> str:
    """
    Get a string name describing the problem, in a format such as
    "hartmann_fixed_noise_6d" or "jenatton" (where the latter would
    not have fixed noise and have the default dimensionality).
    """
    base_name = f"{test_problem.__class__.__name__}"
    observed_noise = "_observed_noise" if observe_noise_sd else ""
    dim_str = "" if dim is None else f"_{dim}d"
    return f"{base_name}{observed_noise}{dim_str}"


@runtime_checkable
class BenchmarkProblemProtocol(Protocol):
    """
    Specifies the interface any benchmark problem must adhere to.

    Classes implementing this interface include BenchmarkProblem,
    SurrogateBenchmarkProblem, and MOOSurrogateBenchmarkProblem.
    """

    name: str
    search_space: SearchSpace
    optimization_config: OptimizationConfig
    num_trials: int
    tracking_metrics: List[BenchmarkMetricBase]
    is_noiseless: bool  # If True, evaluations are deterministic
    observe_noise_stds: Union[
        bool, Dict[str, bool]
    ]  # Whether we observe the observation noise level
    has_ground_truth: bool  # if True, evals (w/o synthetic noise) are determinstic

    @abc.abstractproperty
    def runner(self) -> Runner:
        pass  # pragma: no cover


@runtime_checkable
class BenchmarkProblemWithKnownOptimum(Protocol):
    optimal_value: float


class BenchmarkProblem(Base):
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
        is_noiseless: bool = False,
        observe_noise_sd: bool = False,
        has_ground_truth: bool = False,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
    ) -> None:
        self.name = name
        self.search_space = search_space
        self.optimization_config = optimization_config
        self._runner = runner
        self.num_trials = num_trials
        self.is_noiseless = is_noiseless
        self.observe_noise_sd = observe_noise_sd
        self.has_ground_truth = has_ground_truth
        self.tracking_metrics: List[BenchmarkMetricBase] = tracking_metrics or []

    @property
    def runner(self) -> Runner:
        return self._runner

    @property
    def observe_noise_stds(self) -> Union[bool, Dict[str, bool]]:
        # TODO: Handle cases where some outcomes have noise levels observed
        # and others do not.
        return self.observe_noise_sd

    @classmethod
    def from_botorch(
        cls,
        test_problem_class: Type[BaseTestProblem],
        test_problem_kwargs: Dict[str, Any],
        lower_is_better: bool,
        num_trials: int,
        observe_noise_sd: bool = False,
    ) -> "BenchmarkProblem":
        """
        Create a BenchmarkProblem from a BoTorch BaseTestProblem using
        specialized Metrics and Runners. The test problem's result will be
        computed on the Runner and retrieved by the Metric.

        Args:
            test_problem_class: The BoTorch test problem class which will be used
                to define the `search_space`, `optimization_config`, and `runner`.
            test_problem_kwargs: Keyword arguments used to instantiate the
                `test_problem_class`.
            num_trials: Simply the `num_trials` of the `BenchmarkProblem` created.
            observe_noise_sd: Whether the standard deviation of the observation noise is
                observed or not (in which case it must be inferred by the model).
                This is separate from whether synthetic noise is added to the
                problem, which is controlled by the `noise_std` of the test problem.
        """

        # pyre-fixme [45]: Invalid class instantiation
        test_problem = test_problem_class(**test_problem_kwargs)
        is_constrained = isinstance(test_problem, ConstrainedBaseTestProblem)

        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=lower,
                    upper=upper,
                )
                for i, (lower, upper) in enumerate(test_problem._bounds)
            ]
        )

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(
            test_problem=test_problem, observe_noise_sd=observe_noise_sd, dim=dim
        )

        # TODO: Support constrained MOO problems.

        objective = Objective(
            metric=BenchmarkMetric(
                name=name,
                lower_is_better=lower_is_better,
                observe_noise_sd=observe_noise_sd,
                outcome_index=0,
            ),
            minimize=True,
        )

        outcome_names = [name]
        outcome_constraints = []

        # NOTE: Currently we don't support the case where only some of the
        # outcomes have noise levels observed.

        if is_constrained:
            for i in range(test_problem.num_constraints):
                outcome_name = f"constraint_slack_{i}"
                outcome_constraints.append(
                    OutcomeConstraint(
                        metric=BenchmarkMetric(
                            name=outcome_name,
                            lower_is_better=False,  # positive slack = feasible
                            observe_noise_sd=observe_noise_sd,
                            outcome_index=i,
                        ),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=False,
                    )
                )
                outcome_names.append(outcome_name)

        optimization_config = OptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
        )

        return cls(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=BotorchTestProblemRunner(
                test_problem_class=test_problem_class,
                test_problem_kwargs=test_problem_kwargs,
                outcome_names=outcome_names,
            ),
            num_trials=num_trials,
            observe_noise_sd=observe_noise_sd,
            is_noiseless=test_problem.noise_std in (None, 0.0),
            has_ground_truth=True,  # all synthetic problems have ground truth
        )

    def __repr__(self) -> str:
        """
        Return a string representation that includes only the attributes that
        print nicely and contain information likely to be useful.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"optimization_config={self.optimization_config}, "
            f"num_trials={self.num_trials}, "
            f"is_noiseless={self.is_noiseless}, "
            f"observe_noise_sd={self.observe_noise_sd}, "
            f"has_ground_truth={self.has_ground_truth}, "
            f"tracking_metrics={self.tracking_metrics})"
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
        is_noiseless: bool = False,
        observe_noise_sd: bool = False,
        has_ground_truth: bool = False,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=runner,
            num_trials=num_trials,
            is_noiseless=is_noiseless,
            observe_noise_sd=observe_noise_sd,
            has_ground_truth=has_ground_truth,
            tracking_metrics=tracking_metrics,
        )
        self.optimal_value = optimal_value

    @classmethod
    def from_botorch_synthetic(
        cls,
        test_problem_class: Type[SyntheticTestFunction],
        test_problem_kwargs: Dict[str, Any],
        lower_is_better: bool,
        num_trials: int,
        observe_noise_sd: bool = False,
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
            lower_is_better=lower_is_better,
            num_trials=num_trials,
            observe_noise_sd=observe_noise_sd,
        )

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(
            test_problem=test_problem, observe_noise_sd=observe_noise_sd, dim=dim
        )

        return cls(
            name=name,
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            is_noiseless=problem.is_noiseless,
            observe_noise_sd=problem.observe_noise_sd,
            has_ground_truth=problem.has_ground_truth,
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
        is_noiseless: bool = False,
        observe_noise_sd: bool = False,
        has_ground_truth: bool = False,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
    ) -> None:
        self.maximum_hypervolume = maximum_hypervolume
        self.reference_point = reference_point
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=runner,
            num_trials=num_trials,
            is_noiseless=is_noiseless,
            observe_noise_sd=observe_noise_sd,
            has_ground_truth=has_ground_truth,
            tracking_metrics=tracking_metrics,
        )

    @property
    def optimal_value(self) -> float:
        return self.maximum_hypervolume

    @classmethod
    def from_botorch_multi_objective(
        cls,
        test_problem_class: Type[MultiObjectiveTestProblem],
        test_problem_kwargs: Dict[str, Any],
        # TODO: Figure out whether we should use `lower_is_better` here.
        num_trials: int,
        observe_noise_sd: bool = False,
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
            lower_is_better=True,  # Seems like we always assume minimization for MOO?
            num_trials=num_trials,
            observe_noise_sd=observe_noise_sd,
        )

        dim = test_problem_kwargs.get("dim", None)
        name = _get_name(
            test_problem=test_problem, observe_noise_sd=observe_noise_sd, dim=dim
        )

        n_obj = test_problem.num_objectives
        if not observe_noise_sd:
            noise_sds = [None] * n_obj
        elif isinstance(test_problem.noise_std, list):
            noise_sds = test_problem.noise_std
        else:
            noise_sds = [checked_cast(float, test_problem.noise_std or 0.0)] * n_obj

        metrics = [
            BenchmarkMetric(
                name=f"{name}_{i}",
                lower_is_better=True,
                observe_noise_sd=observe_noise_sd,
                outcome_index=i,
            )
            for i, noise_sd in enumerate(noise_sds)
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=metric, minimize=True) for metric in metrics
                ]
            ),
            objective_thresholds=[
                ObjectiveThreshold(
                    metric=metric,
                    bound=test_problem.ref_point[i].item(),
                    relative=False,
                    op=ComparisonOp.LEQ,
                )
                for i, metric in enumerate(metrics)
            ],
        )

        return cls(
            name=name,
            search_space=problem.search_space,
            optimization_config=optimization_config,
            runner=problem.runner,
            num_trials=num_trials,
            is_noiseless=problem.is_noiseless,
            observe_noise_sd=observe_noise_sd,
            has_ground_truth=problem.has_ground_truth,
            maximum_hypervolume=test_problem.max_hv,
            reference_point=test_problem._ref_point,
        )
