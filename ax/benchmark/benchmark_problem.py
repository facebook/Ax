# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ax.benchmark.benchmark_metric import BenchmarkMapMetric, BenchmarkMetric
from ax.benchmark.benchmark_step_runtime_function import TBenchmarkStepRuntimeFunction
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.noise import GaussianNoise, Noise
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.exceptions.core import UserInputError
from ax.utils.common.base import Base


def _is_minimizing(objective: Objective) -> bool:
    """Determine if an objective is minimizing.

    Handles scalarized objectives by checking if all weights are negative
    (which indicates minimization in the expression-based API).
    """
    if objective.is_scalarized_objective:
        return all(w < 0 for _, w in objective.metric_weights)
    if objective.is_multi_objective:
        return all(w < 0 for _, w in objective.metric_weights)
    return objective.minimize


@dataclass(kw_only=True, repr=True)
class BenchmarkProblem(Base):
    """
    Problem against which different methods can be benchmarked.

    Defines how data is generated, the objective (via the OptimizationConfig),
    and the SearchSpace.

    Args:
        name: Can be generated programmatically with `_get_name`.
        optimization_config: Defines the objective of optimization. Metrics must
            be `BenchmarkMetric`s.
        num_trials: Number of optimization iterations to run. BatchTrials count
            as one trial.
        test_function: A `BenchmarkTestFunction`, which will generate noiseless
            data. This will be used by a `BenchmarkRunner`.
        noise: A `Noise` object that determines how noise is added to the
            ground-truth evaluations produced by the `test_function`. Defaults
            to noiseless (`GaussianNoise(noise_std=0.0)`).
        noise_std: Deprecated. Use `noise` instead.
        optimal_value: The best ground-truth objective value, used for scoring
            optimization results on a scale from 0 to 100, where achieving the
            `optimal_value` receives a score of 100. The `optimal_value` should
            be a hypervolume for multi-objective problems. If the best value is
            not known, it is conventional to set it to a value that is almost
            certainly better than the best value, so that a benchmark's score
            will not exceed 100%.
        baseline_value: Similar to `optimal_value`, but a not-so-good value
            which benchmarks are expected to do better than. A baseline value
            can be derived using the function
            `compute_baseline_value_from_sobol`, which takes the best of five
            quasi-random Sobol trials.
        worst_feasible_value: The worst possible objective value for a feasible trial.
            This must be provided for constrained problems. This value is assigned to
            infeasible trials when computing the score of a given benchmark probem.
            This has the desirable property that any feasible trial has a better score
            than an infeasible trial.
        search_space: The search space.
        report_inference_value_as_trace: Whether the ``optimization_trace`` on a
            ``BenchmarkResult`` should use the ``oracle_trace`` (if False,
            default) or the ``inference_trace``. See ``BenchmarkResult`` for
            more information. Currently, this is only supported for
            single-objective problems.
        step_runtime_function: Optionally, a function that takes in ``params``
            (typically dictionaries mapping strings to ``TParamValue``s) and
            returns the runtime of an step. If ``step_runtime_function`` is
            left as ``None``, each step will take one simulated second.  (When
            data is not time-series, the whole trial consists of one step.)
        target_fidelity_and_task: A mapping from names of task and fidelity
            parameters to their respective target values.
        status_quo_params: The parameterization of the status quo arm. Required
            when using relative constraints.
        auxiliary_experiments_by_purpose: A mapping from experiment purpose to
            a list of auxiliary experiments.
        tracking_metrics: A list of metrics to track on the experiment in
            addition to the metrics contained in the OptimizationConfig.
            Tracking metrics appear in the data stored on the Experiment
            and do not affect the traces in a BenchmarkResult.
    """

    name: str
    optimization_config: OptimizationConfig
    num_trials: int
    test_function: BenchmarkTestFunction
    noise: Noise = field(default_factory=GaussianNoise)
    noise_std: float | Mapping[str, float] | None = None
    optimal_value: float
    baseline_value: float
    worst_feasible_value: float | None = None
    search_space: SearchSpace = field(repr=False)
    report_inference_value_as_trace: bool = False
    step_runtime_function: TBenchmarkStepRuntimeFunction | None = None
    target_fidelity_and_task: Mapping[str, TParamValue] = field(default_factory=dict)
    status_quo_params: Mapping[str, TParamValue] | None = None
    auxiliary_experiments_by_purpose: (
        dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]] | None
    ) = None
    tracking_metrics: list[Metric] | None = None
    opt_config_metrics: list[Metric] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Handle backward compatibility for noise_std parameter
        if self.noise_std is not None:
            warnings.warn(
                "noise_std is deprecated. Use noise=GaussianNoise(noise_std=...) "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Check if noise was also explicitly set (not default)
            if not isinstance(self.noise, GaussianNoise) or self.noise.noise_std != 0.0:
                raise ValueError(
                    "Cannot specify both 'noise_std' and a non-default 'noise'. "
                    "Use only 'noise=GaussianNoise(noise_std=...)' instead."
                )
            # Convert noise_std to GaussianNoise (use 0 if None)
            self.noise = GaussianNoise(noise_std=self.noise_std or 0)

        # Validate inputs
        if self.report_inference_value_as_trace and self.is_moo:
            raise NotImplementedError(
                "Inference trace is not supported for MOO. Please set "
                "`report_inference_value_as_trace` to False."
            )

        # Validate that the optimal value is actually better than the baseline
        # value
        # If MOO, values represent hypervolumes
        if isinstance(self.optimization_config, MultiObjectiveOptimizationConfig):
            if self.baseline_value >= self.optimal_value:
                raise ValueError(
                    "The baseline value must be strictly less than the optimal "
                    "value for MOO problems. (These represent hypervolumes.)"
                )
        elif _is_minimizing(self.optimization_config.objective):
            if self.baseline_value <= self.optimal_value:
                raise ValueError(
                    "The baseline value must be strictly greater than the optimal "
                    "value for minimization problems."
                )
        elif self.baseline_value >= self.optimal_value:
            raise ValueError(
                "The baseline value must be strictly less than the optimal "
                "value for maximization problems."
            )
        # Validate worst_feasible_value
        if len(self.optimization_config.outcome_constraints) > 0:
            if isinstance(self.optimization_config, MultiObjectiveOptimizationConfig):
                if self.worst_feasible_value != 0.0:
                    raise ValueError(
                        "The worst feasible value must be 0.0 for multi-objective "
                        "problems."
                    )
            elif self.worst_feasible_value is None:
                raise ValueError(
                    "The worst feasible value must be provided for constrained "
                    "problems (got `None`)"
                )
            else:
                wfv = self.worst_feasible_value
                if _is_minimizing(self.optimization_config.objective):
                    if self.optimal_value > wfv:
                        raise ValueError(
                            "The worst feasible value must be greater than or "
                            "equal to the optimal value for minimization "
                            "problems."
                        )
                elif self.optimal_value < wfv:
                    raise ValueError(
                        "The worst feasible value must be less than or equal to "
                        "the optimal value for maximization problems."
                    )

        # Validate that names on optimization config are contained in names on
        # test function
        objective = self.optimization_config.objective
        objective_names = set(objective.metric_names)

        test_function_names = set(self.test_function.outcome_names)
        missing = objective_names - test_function_names
        if len(missing) > 0:
            raise ValueError(
                "The following objectives are defined on "
                "`optimization_config` but not included in "
                f"`runner.test_function.outcome_names`: {missing}."
            )

        constraints = self.optimization_config.outcome_constraints
        constraint_names: set[str] = set()
        for c in constraints:
            constraint_names.update(c.metric_names)
        missing = constraint_names - test_function_names
        if len(missing) > 0:
            raise ValueError(
                "The following constraints are defined on "
                "`optimization_config` but not included in "
                f"`runner.test_function.outcome_names`: {missing}."
            )
        if any(c.relative for c in constraints) and self.status_quo_params is None:
            raise ValueError(
                "Relative constraints require specifying status_quo_params."
            )

        self.target_fidelity_and_task = {
            p.name: p.target_value
            for p in self.search_space.parameters.values()
            if (isinstance(p, ChoiceParameter) and p.is_task) or p.is_fidelity
        }
        if (
            self.status_quo_params is not None
            and not self.search_space.check_membership(
                parameterization=self.status_quo_params
            )
        ):
            raise UserInputError("Status quo parameters are not in the search space.")

    @property
    def is_moo(self) -> bool:
        """Whether the problem is multi-objective."""
        return isinstance(self.optimization_config, MultiObjectiveOptimizationConfig)


def _get_constraints(
    constraint_names: Sequence[str],
    observe_noise_sd: bool,
    use_map_metric: bool = False,
) -> tuple[list[OutcomeConstraint], list[Metric]]:
    """
    Create a list of ``OutcomeConstraint``s and corresponding
    ``BenchmarkMetric``s.

    Args:
        constraint_names: Names of the constraints. One constraint will be
            created for each.
        observe_noise_sd: Whether the standard deviation of the observation
            noise is observed, for each constraint. This doesn't handle the case
            where only some of the outcomes have noise levels observed.
        use_map_metric: Whether to use a ``BenchmarkMapMetric``.

    Returns:
        A tuple of (outcome_constraints, metrics).
    """
    metric_cls = BenchmarkMapMetric if use_map_metric else BenchmarkMetric
    metrics: list[Metric] = [
        metric_cls(
            name=name,
            lower_is_better=False,  # positive slack = feasible
            observe_noise_sd=observe_noise_sd,
        )
        for name in constraint_names
    ]
    outcome_constraints = [
        OutcomeConstraint(expression=f"{name} >= 0.0") for name in constraint_names
    ]
    return outcome_constraints, metrics


def get_soo_opt_config(
    *,
    outcome_names: Sequence[str],
    lower_is_better: bool = True,
    observe_noise_sd: bool = False,
    use_map_metric: bool = False,
) -> tuple[OptimizationConfig, list[Metric]]:
    """
    Create a single-objective ``OptimizationConfig``, potentially with
    constraints, along with the corresponding ``BenchmarkMetric``s.

    Args:
        outcome_names: Names of the outcomes. If ``outcome_names`` has more than
            one element, constraints will be created. The first element of
            ``outcome_names`` will be the name of the ``BenchmarkMetric`` on
            the objective, and others (if pressent) will be for constraints.
        lower_is_better: Whether the objective is a minimization problem. This
            only affects objectives, not constraints; for constraints, higher is
            better (feasible).
        observe_noise_sd: Whether the standard deviation of the observation
            noise is observed. Applies to all objective and constraints.
        use_map_metric: Whether to use a ``BenchmarkMapMetric``.

    Returns:
        A tuple of (OptimizationConfig, list of BenchmarkMetrics).
    """
    metric_cls = BenchmarkMapMetric if use_map_metric else BenchmarkMetric
    obj_metric = metric_cls(
        name=outcome_names[0],
        lower_is_better=lower_is_better,
        observe_noise_sd=observe_noise_sd,
    )
    expression = f"-{outcome_names[0]}" if lower_is_better else outcome_names[0]
    objective = Objective(expression=expression)

    outcome_constraints, constraint_metrics = _get_constraints(
        constraint_names=outcome_names[1:],
        observe_noise_sd=observe_noise_sd,
        use_map_metric=use_map_metric,
    )

    config = OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )
    return config, [obj_metric] + constraint_metrics


def get_moo_opt_config(
    *,
    outcome_names: Sequence[str],
    ref_point: Sequence[float],
    num_constraints: int = 0,
    lower_is_better: bool = True,
    observe_noise_sd: bool = False,
    use_map_metric: bool = False,
) -> tuple[MultiObjectiveOptimizationConfig, list[Metric]]:
    """
    Create a ``MultiObjectiveOptimizationConfig``, potentially with
    constraints, along with the corresponding ``BenchmarkMetric``s.

    Args:
        outcome_names: Names of the outcomes. If ``num_constraints`` is greater
            than zero, the last ``num_constraints`` elements of
            ``outcome_names`` will become the names of ``BenchmarkMetric``s on
            constraints, and the others will correspond to the objectives.
        ref_point: Objective thresholds for the objective metrics. Note:
            Although this method requires providing a threshold for each
            objective, this is not required in general and could be enabled for
            this method.
        num_constraints: Number of constraints.
        lower_is_better: Whether the objectives are lower-is-better. Applies to
            all objectives and not to constraints. For constraints, higher is
            better (feasible). Note: Ax allows different metrics to have
            different values of ``lower_is_better``; that isn't enabled for this
            method, but could be.
        observe_noise_sd: Whether the standard deviation of the observation
        noise is observed. Applies to all objective and constraints.

    Returns:
        A tuple of (MultiObjectiveOptimizationConfig, list of BenchmarkMetrics).
    """
    n_objectives = len(outcome_names) - num_constraints
    metric_cls = BenchmarkMapMetric if use_map_metric else BenchmarkMetric
    objective_metrics = [
        metric_cls(
            name=outcome_names[i],
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
        )
        for i in range(n_objectives)
    ]
    outcome_constraints, constraint_metrics = _get_constraints(
        constraint_names=outcome_names[n_objectives:],
        observe_noise_sd=observe_noise_sd,
    )

    if n_objectives < 2:
        raise ValueError(
            "get_moo_opt_config requires at least 2 objectives. "
            f"Got {n_objectives} objective(s) with {num_constraints} constraint(s) "
            f"from {len(outcome_names)} outcome names."
        )

    # Build multi-objective expression: comma-separated, negated if lower_is_better
    obj_expressions = []
    for metric in objective_metrics:
        if lower_is_better:
            obj_expressions.append(f"-{metric.name}")
        else:
            obj_expressions.append(metric.name)
    objective = Objective(expression=", ".join(obj_expressions))

    # Build objective thresholds as OutcomeConstraints
    objective_thresholds = []
    for ref_p, metric in zip(ref_point, objective_metrics, strict=True):
        if metric.lower_is_better:
            expr = f"{metric.name} <= {ref_p}"
        else:
            expr = f"{metric.name} >= {ref_p}"
        objective_thresholds.append(OutcomeConstraint(expression=expr))

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=objective_thresholds,
        outcome_constraints=outcome_constraints,
    )
    return optimization_config, objective_metrics + constraint_metrics


def get_continuous_search_space(
    bounds: list[tuple[float, float]], n_dummy_dimensions: int = 0
) -> SearchSpace:
    """
    Create a continuous ``SearchSpace``.

    Args:
        bounds: A list of tuples of lower and upper bounds for each dimension.
            These apply only to the original problem dimensions. Any extra
            dimensions will have bounds [0, 1].
        n_dummy_dimensions: Number of extra dimensions to add to the search
            space.
    """
    original_problem_parameters = [
        RangeParameter(
            name=f"x{i}",
            parameter_type=ParameterType.FLOAT,
            lower=lower,
            upper=upper,
        )
        for i, (lower, upper) in enumerate(bounds)
    ]
    dummy_parameters = [
        RangeParameter(
            name=f"embedding_dummy_{i}",
            parameter_type=ParameterType.FLOAT,
            lower=0,
            upper=1,
        )
        for i in range(n_dummy_dimensions)
    ]
    return SearchSpace(parameters=original_problem_parameters + dummy_parameters)
