# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

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
from botorch.test_functions.base import (
    BaseTestProblem,
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
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


@dataclass(kw_only=True, repr=True)
class BenchmarkProblem(Base):
    """
    Problem against which diffrent methods can be benchmarked.

    Defines how data is generated, the objective (via the OptimizationConfig),
    and the SearchSpace.

    Args:
        name: Can be generated programmatically with `_get_name`.
        optimization_config: Defines the objective of optimizaiton.
        num_trials: Number of optimization iterations to run. BatchTrials count
            as one trial.
        observe_noise_stds: If boolean, whether the standard deviation of the
            observation noise is observed for all metrics. If a dictionary,
            whether noise levels are observed on a per-metric basis.
        has_ground_truth: Whether the Runner produces underlying ground truth
            values, which are not observed in real noisy problems but may be
            known in benchmarks.
        tracking_metrics: Tracking metrics are not optimized, and for the
            purpose of benchmarking, they will not be fit. The ground truth may
            be provided as `tracking_metrics`.
        optimal_value: The best ground-truth objective value. Hypervolume for
            multi-objective problems. If the best value is not known, it is
            conventional to set it to a value that is almost certainly better
            than the best value, so that a benchmark's score will not exceed 100%.
        search_space: The search space.
        runner: The Runner that will be used to generate data for the problem,
            including any ground-truth data stored as tracking metrics.
    """

    name: str
    optimization_config: OptimizationConfig
    num_trials: int
    observe_noise_stds: Union[bool, Dict[str, bool]] = False
    has_ground_truth: bool = True
    tracking_metrics: List[BenchmarkMetricBase] = field(default_factory=list)
    optimal_value: float

    search_space: SearchSpace = field(repr=False)
    runner: Runner = field(repr=False)
    is_noiseless: bool


class SingleObjectiveBenchmarkProblem(BenchmarkProblem):
    """A `BenchmarkProblem` that supports a single objective."""

    pass


def create_single_objective_problem_from_botorch(
    test_problem_class: Type[SyntheticTestFunction],
    test_problem_kwargs: Dict[str, Any],
    lower_is_better: bool,
    num_trials: int,
    observe_noise_sd: bool = False,
) -> SingleObjectiveBenchmarkProblem:
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
        minimize=lower_is_better,
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
    optimal_value = (
        test_problem.max_hv
        if isinstance(test_problem, MultiObjectiveTestProblem)
        else test_problem.optimal_value
    )
    return SingleObjectiveBenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=BotorchTestProblemRunner(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            outcome_names=outcome_names,
        ),
        num_trials=num_trials,
        observe_noise_stds=observe_noise_sd,
        is_noiseless=test_problem.noise_std in (None, 0.0),
        has_ground_truth=True,  # all synthetic problems have ground truth
        optimal_value=optimal_value,
    )


@dataclass(kw_only=True, repr=True)
class MultiObjectiveBenchmarkProblem(BenchmarkProblem):
    """
    A `BenchmarkProblem` that supports multiple objectives.

    For multi-objective problems, `optimal_value` indicates the maximum
    hypervolume attainable with the given `reference_point`.

    For argument descriptions, see `BenchmarkProblem`; it additionally takes a `runner`
    and a `reference_point`.
    """

    reference_point: List[float]
    optimization_config: MultiObjectiveOptimizationConfig


def create_multi_objective_problem_from_botorch(
    test_problem_class: Type[MultiObjectiveTestProblem],
    test_problem_kwargs: Dict[str, Any],
    # TODO: Figure out whether we should use `lower_is_better` here.
    num_trials: int,
    observe_noise_sd: bool = False,
) -> MultiObjectiveBenchmarkProblem:
    """Create a BenchmarkProblem from a BoTorch BaseTestProblem using specialized
    Metrics and Runners. The test problem's result will be computed on the Runner
    once per trial and each Metric will retrieve its own result by index.
    """
    if issubclass(test_problem_class, ConstrainedBaseTestProblem):
        raise NotImplementedError(
            "Constrained multi-objective problems are not supported."
        )

    # pyre-fixme [45]: Invalid class instantiation
    test_problem = test_problem_class(**test_problem_kwargs)

    problem = create_single_objective_problem_from_botorch(
        # pyre-fixme [6]: Passing a multi-objective problem where a
        # single-objective problem is expected.
        test_problem_class=test_problem_class,
        test_problem_kwargs=test_problem_kwargs,
        lower_is_better=True,  # Seems like we always assume minimization for MOO?
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
    )

    name = problem.name

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
            objectives=[Objective(metric=metric, minimize=True) for metric in metrics]
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

    return MultiObjectiveBenchmarkProblem(
        name=name,
        search_space=problem.search_space,
        optimization_config=optimization_config,
        runner=problem.runner,
        num_trials=num_trials,
        is_noiseless=problem.is_noiseless,
        observe_noise_stds=observe_noise_sd,
        has_ground_truth=problem.has_ground_truth,
        optimal_value=test_problem.max_hv,
        reference_point=test_problem._ref_point,
    )
