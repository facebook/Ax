# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.runners.base import BenchmarkRunner
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp, TParamValue
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from ax.utils.common.base import Base
from botorch.test_functions.base import (
    BaseTestProblem,
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)


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
    Problem against which different methods can be benchmarked.

    Defines how data is generated, the objective (via the OptimizationConfig),
    and the SearchSpace.

    Args:
        name: Can be generated programmatically with `_get_name`.
        optimization_config: Defines the objective of optimization.
        num_trials: Number of optimization iterations to run. BatchTrials count
            as one trial.
        observe_noise_stds: If boolean, whether the standard deviation of the
            observation noise is observed for all metrics. If a dictionary,
            whether noise levels are observed on a per-metric basis.
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
    observe_noise_stds: Union[bool, dict[str, bool]] = False
    optimal_value: float

    search_space: SearchSpace = field(repr=False)
    runner: BenchmarkRunner = field(repr=False)

    def get_oracle_experiment_from_params(
        self,
        dict_of_dict_of_params: Mapping[int, Mapping[str, [Mapping[str, TParamValue]]]],
    ) -> Experiment:
        """
        Get a new experiment with the same search space and optimization config
        as those belonging to this problem, but with parameterizations evaluated
        at oracle values.

        Args:
            dict_of_dict_of_params: Keys are trial indices, values are Mappings
                (e.g. dicts) that map arm names to parameterizations.

        Example:
            >>> problem.get_oracle_experiment_from_params(
            ...     {
            ...         0: {
            ...            "0_0": {"x0": 0.0, "x1": 0.0},
            ...            "0_1": {"x0": 0.3, "x1": 0.4},
            ...         },
            ...         1: {"1_0": {"x0": 0.0, "x1": 0.0}},
            ...     }
            ... )
        """
        records = []

        experiment = Experiment(
            search_space=self.search_space, optimization_config=self.optimization_config
        )
        if len(dict_of_dict_of_params) == 0:
            return experiment

        for trial_index, dict_of_params in dict_of_dict_of_params.items():
            if len(dict_of_params) == 0:
                raise ValueError(
                    "Can't create a trial with no arms. Each sublist in "
                    "list_of_list_of_params must have at least one element."
                )
            for arm_name, params in dict_of_params.items():
                for metric_name, metric_value in zip(
                    self.runner.outcome_names,
                    self.runner.evaluate_oracle(parameters=params),
                ):
                    records.append(
                        {
                            "arm_name": arm_name,
                            "metric_name": metric_name,
                            "mean": metric_value,
                            "sem": 0.0,
                            "trial_index": trial_index,
                        }
                    )

            experiment.attach_trial(
                parameterizations=list(dict_of_params.values()),
                arm_names=list(dict_of_params.keys()),
            )
        for trial in experiment.trials.values():
            trial.mark_completed()

        data = Data(df=pd.DataFrame.from_records(records))
        experiment.attach_data(data=data, overwrite_existing_data=True)
        return experiment

    def get_oracle_experiment_from_experiment(
        self, experiment: Experiment
    ) -> Experiment:
        return self.get_oracle_experiment_from_params(
            dict_of_dict_of_params={
                trial.index: {arm.name: arm.parameters for arm in trial.arms}
                for trial in experiment.trials.values()
            }
        )

    @property
    def is_moo(self) -> bool:
        """Whether the problem is multi-objective."""
        return isinstance(self.optimization_config, MultiObjectiveOptimizationConfig)


def _get_constraints(
    num_constraints: int, observe_noise_sd: bool
) -> list[OutcomeConstraint]:
    """
    NOTE: Currently we don't support the case where only some of the
    outcomes have noise levels observed.
    """
    outcome_constraints = []

    for i in range(num_constraints):
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
    return outcome_constraints


def get_soo_config_and_outcome_names(
    *,
    num_constraints: int,
    lower_is_better: bool,
    observe_noise_sd: bool,
    objective_name: str,
) -> tuple[OptimizationConfig, list[str]]:

    objective = Objective(
        metric=BenchmarkMetric(
            name=objective_name,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            outcome_index=0,
        ),
        minimize=lower_is_better,
    )

    outcome_constraints = _get_constraints(
        num_constraints=num_constraints, observe_noise_sd=observe_noise_sd
    )
    constraint_names = [oc.metric.name for oc in outcome_constraints]

    opt_config = OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )
    outcome_names = [objective_name] + constraint_names
    return opt_config, outcome_names


def get_moo_opt_config_and_outcome_names(
    *,
    num_constraints: int,
    lower_is_better: bool,
    observe_noise_sd: bool,
    objective_names: list[str],
    ref_point: list[float],
) -> tuple[MultiObjectiveOptimizationConfig, list[str]]:
    metrics = [
        BenchmarkMetric(
            name=objective_name,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            outcome_index=i,
        )
        for i, objective_name in enumerate(objective_names)
    ]
    constraints = _get_constraints(
        num_constraints=num_constraints, observe_noise_sd=observe_noise_sd
    )
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(
            objectives=[
                Objective(metric=metric, minimize=lower_is_better) for metric in metrics
            ]
        ),
        objective_thresholds=[
            ObjectiveThreshold(
                metric=metric,
                bound=ref_point[i],
                relative=False,
                op=ComparisonOp.LEQ if metric.lower_is_better else ComparisonOp.GEQ,
            )
            for i, metric in enumerate(metrics)
        ],
        outcome_constraints=constraints,
    )
    outcome_names = objective_names + [oc.metric.name for oc in constraints]
    return optimization_config, outcome_names


def get_continuous_search_space(bounds: list[tuple[float, float]]) -> SearchSpace:
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=lower,
                upper=upper,
            )
            for i, (lower, upper) in enumerate(bounds)
        ]
    )


def create_problem_from_botorch(
    *,
    test_problem_class: type[BaseTestProblem],
    test_problem_kwargs: dict[str, Any],
    num_trials: int,
    lower_is_better: bool = True,
    observe_noise_sd: bool = False,
    search_space: SearchSpace | None = None,
) -> BenchmarkProblem:
    """
    Create a `BenchmarkProblem` from a BoTorch `BaseTestProblem`.

    Uses specialized Metrics and Runners for benchmarking. The test problem's
    result will be computed by the Runner, `BoTorchTestProblemRunner`, and
    retrieved by the Metric(s), which are `BenchmarkMetric`s.

    Args:
        test_problem_class: The BoTorch test problem class which will be used
            to define the `search_space`, `optimization_config`, and `runner`.
        test_problem_kwargs: Keyword arguments used to instantiate the
            `test_problem_class`.
        lower_is_better: Whether this is a minimization problem. For MOO, this
            applies to all objectives.
        num_trials: Simply the `num_trials` of the `BenchmarkProblem` created.
        observe_noise_sd: Whether the standard deviation of the observation noise is
            observed or not (in which case it must be inferred by the model).
            This is separate from whether synthetic noise is added to the
            problem, which is controlled by the `noise_std` of the test problem.
        search_space: If provided, the `search_space` of the `BenchmarkProblem`.
            Otherwise, a `SearchSpace` with all `RangeParameter`s is created
            from the bounds of the test problem.
    """
    # pyre-fixme [45]: Invalid class instantiation
    test_problem = test_problem_class(**test_problem_kwargs)
    is_constrained = isinstance(test_problem, ConstrainedBaseTestProblem)

    search_space = (
        get_continuous_search_space(test_problem._bounds)
        if search_space is None
        else search_space
    )

    dim = test_problem_kwargs.get("dim", None)

    n_obj = test_problem.num_objectives
    name = _get_name(
        test_problem=test_problem, observe_noise_sd=observe_noise_sd, dim=dim
    )

    num_constraints = test_problem.num_constraints if is_constrained else 0
    if isinstance(test_problem, MultiObjectiveTestProblem):
        optimization_config, outcome_names = get_moo_opt_config_and_outcome_names(
            num_constraints=num_constraints,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            objective_names=[f"{name}_{i}" for i in range(n_obj)],
            ref_point=test_problem._ref_point,
        )
    else:
        optimization_config, outcome_names = get_soo_config_and_outcome_names(
            num_constraints=num_constraints,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            objective_name=name,
        )

    optimal_value = (
        test_problem.max_hv
        if isinstance(test_problem, MultiObjectiveTestProblem)
        else test_problem.optimal_value
    )
    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        runner=BotorchTestProblemRunner(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            outcome_names=outcome_names,
            search_space_digest=extract_search_space_digest(
                search_space=search_space,
                param_names=list(search_space.parameters.keys()),
            ),
        ),
        num_trials=num_trials,
        observe_noise_stds=observe_noise_sd,
        optimal_value=optimal_value,
    )
