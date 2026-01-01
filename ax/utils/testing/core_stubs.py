#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import datetime, timedelta
from functools import partial
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any, cast, Sequence

import numpy as np
import pandas as pd
import torch
from ax.adapter.factory import get_factorial, get_sobol
from ax.adapter.registry import Cont_X_trans, Generators
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.evaluations_to_data import raw_evaluations_to_data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import (
    ComparisonOp,
    TCandidateMetadata,
    TModelCov,
    TModelMean,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.early_stopping.strategies import (
    BaseEarlyStoppingStrategy,
    PercentileEarlyStoppingStrategy,
    ThresholdEarlyStoppingStrategy,
)
from ax.early_stopping.strategies.logical import (
    AndEarlyStoppingStrategy,
    OrEarlyStoppingStrategy,
)
from ax.generation_strategy.generation_node_input_constructors import (
    InputConstructorPurpose,
    NodeInputConstructors,
)
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    MaxGenerationParallelism,
    MinTrials,
    TrialBasedCriterion,
)
from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.kernels import (
    DefaultRBFKernel,
    ScaleMaternKernel,
)
from ax.generators.torch.botorch_modular.surrogate import (
    ModelConfig,
    Surrogate,
    SurrogateSpec,
)
from ax.generators.winsorization_config import WinsorizationConfig
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.metrics.branin import BraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.orchestrator_options import OrchestratorOptions, TrialType
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.random import set_rng_seed
from ax.utils.measurement.synthetic_functions import branin
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Round
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import DEFAULT
from gpytorch.constraints import Interval
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from pyre_extensions import assert_is_instance, none_throws
from typing_extensions import Self

logger: Logger = get_logger(__name__)

TEST_SOBOL_SEED = 1234
TORCH_RAND_SEED = 42
DEFAULT_USER = "foo-user"

##############################
# Experiments
##############################


def get_experiment(
    with_status_quo: bool = True, constrain_search_space: bool = True
) -> Experiment:
    experiment = Experiment(
        name="test",
        search_space=get_search_space(constrain_search_space=constrain_search_space),
        optimization_config=get_optimization_config(),
        status_quo=get_status_quo() if with_status_quo else None,
        description="test description",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )
    experiment._properties = {"owners": [DEFAULT_USER]}
    return experiment


def get_experiment_with_map_data_type() -> Experiment:
    """Returns an experiment with the search space including parameters
    (with mixture of types) ["w", "x", "y", "z"], a status quo, a single
    objective optimization config with MapMetric "m1", and a tracking
    MapMetric "tracking", both using the default MapKeyInfo.
    """
    experiment = Experiment(
        name="test_map_data",
        search_space=get_search_space(),
        optimization_config=get_map_optimization_config(),
        status_quo=get_status_quo(),
        description="test description",
        tracking_metrics=[MapMetric(name="tracking")],
        is_test=True,
    )
    experiment._properties = {"owners": [DEFAULT_USER]}
    return experiment


def get_trial_based_criterion() -> list[TrialBasedCriterion]:
    return [
        MinTrials(
            threshold=3,
            only_in_statuses=[TrialStatus.RUNNING, TrialStatus.COMPLETED],
            not_in_statuses=None,
        ),
        MaxGenerationParallelism(
            threshold=5,
            only_in_statuses=None,
            not_in_statuses=[
                TrialStatus.RUNNING,
            ],
        ),
    ]


def get_experiment_with_custom_runner_and_metric(
    constrain_search_space: bool = True,
    immutable: bool = False,
    multi_objective: bool = False,
    scalarized_objective: bool = False,
    num_trials: int = 3,
    has_outcome_constraint: bool = False,
) -> Experiment:
    # Validate mutually exclusive parameters
    if multi_objective and scalarized_objective:
        raise ValueError("multi_objective and scalarized_objective cannot both be True")

    # Create optimization config based on the requested type
    if multi_objective:
        optimization_config = get_multi_objective_optimization_config(
            custom_metric=True,
            outcome_constraint=has_outcome_constraint,
            relative=False,
        )
    elif scalarized_objective:
        # Create scalarized objective with custom metrics
        custom_scalarized_objective = ScalarizedObjective(
            metrics=[
                CustomTestMetric(name="m1", test_attribute="test"),
                CustomTestMetric(name="m3", test_attribute="test"),
            ],
            weights=[1.0, 2.0],
            minimize=False,
        )

        outcome_constraints = []
        if has_outcome_constraint:
            # When both scalarized_objective and has_outcome_constraint are True,
            # automatically use scalarized outcome constraint
            custom_scalarized_constraint = ScalarizedOutcomeConstraint(
                metrics=[
                    CustomTestMetric(name="oc_m3", test_attribute="test"),
                    CustomTestMetric(name="oc_m4", test_attribute="test"),
                ],
                weights=[0.2, 0.8],
                op=ComparisonOp.GEQ,
                bound=-0.25,
            )
            outcome_constraints.append(custom_scalarized_constraint)

        optimization_config = OptimizationConfig(
            objective=custom_scalarized_objective,
            outcome_constraints=outcome_constraints,
        )
    else:
        optimization_config = get_optimization_config(
            outcome_constraint=has_outcome_constraint, relative=False
        )

    # Create experiment with custom runner and metric
    experiment = Experiment(
        name="test",
        search_space=get_search_space(constrain_search_space=constrain_search_space),
        optimization_config=optimization_config,
        description="test description",
        tracking_metrics=[
            CustomTestMetric(name="custom_test_metric", test_attribute="test")
        ],
        runner=CustomTestRunner(test_attribute="test"),
        is_test=True,
    )
    experiment._properties = {"owners": [DEFAULT_USER]}

    # Create a trial, set its runner and complete it.
    for _ in range(num_trials):
        sobol_generator = get_sobol(
            search_space=experiment.search_space,
        )
        sobol_run = sobol_generator.gen(
            n=1,
            optimization_config=(
                experiment.optimization_config if not immutable else None
            ),
        )
        trial = experiment.new_trial(generator_run=sobol_run)
        trial.mark_running()
        data = Data.from_multiple_data(
            get_data(
                metric_name=metric_name,
                trial_index=trial.index,
                num_non_sq_arms=len(trial.arms),
                include_sq=False,
            )
            for metric_name in experiment.metrics
        )
        experiment.attach_data(data)
        trial.mark_completed()

    if immutable:
        experiment._properties = {Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True}

    return experiment


def get_branin_experiment(
    has_optimization_config: bool = True,
    with_batch: bool = False,
    with_trial: bool = False,
    with_status_quo: bool = False,
    status_quo_unknown_parameters: bool = False,
    with_fidelity_parameter: bool = False,
    with_choice_parameter: bool = False,
    with_str_choice_param: bool = False,
    with_derived_parameter: bool = False,
    with_parameter_constraint: bool = False,
    search_space: SearchSpace | None = None,
    minimize: bool = False,
    named: bool = True,
    num_trial: int = 1,
    num_batch_trial: int = 1,
    with_completed_batch: bool = False,
    with_completed_trial: bool = False,
    num_arms_per_trial: int = 15,
    with_relative_constraint: bool = False,
    with_absolute_constraint: bool = False,
) -> Experiment:
    search_space = search_space or get_branin_search_space(
        with_fidelity_parameter=with_fidelity_parameter,
        with_choice_parameter=with_choice_parameter,
        with_str_choice_param=with_str_choice_param,
        with_derived_parameter=with_derived_parameter,
        with_parameter_constraint=with_parameter_constraint,
    )

    status_quo = (
        get_status_quo_branin(
            with_fidelity_parameter=with_fidelity_parameter,
            with_derived_parameter=with_derived_parameter,
            with_str_choice_param=with_str_choice_param,
            status_quo_unknown_parameters=status_quo_unknown_parameters,
        )
        if with_status_quo
        else None
    )

    exp = Experiment(
        name="branin_test_experiment" if named else None,
        search_space=search_space,
        optimization_config=(
            get_branin_optimization_config(
                minimize=minimize,
                with_relative_constraint=with_relative_constraint,
                with_absolute_constraint=with_absolute_constraint,
            )
            if has_optimization_config or with_relative_constraint
            else None
        ),
        runner=SyntheticRunner(),
        is_test=True,
        status_quo=status_quo,
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    if with_batch or with_completed_batch:
        for _ in range(num_batch_trial):
            sobol_generator = get_sobol(search_space=exp.search_space)
            sobol_run = sobol_generator.gen(n=num_arms_per_trial)
            trial = exp.new_batch_trial(should_add_status_quo_arm=with_status_quo)
            trial.add_generator_run(sobol_run)

            if with_completed_batch:
                trial.mark_running(no_runner_required=True)
                exp.attach_data(
                    get_branin_data_batch(batch=trial, metrics=[*exp.metrics.keys()])
                )
                trial.mark_completed()

    if with_trial or with_completed_trial:
        for _ in range(num_trial):
            sobol_generator = get_sobol(search_space=exp.search_space)
            sobol_run = sobol_generator.gen(n=1)
            trial = exp.new_trial(generator_run=sobol_run)

            if with_completed_trial:
                trial.mark_running(no_runner_required=True)
                exp.attach_data(
                    get_branin_data(trials=[trial], metrics=exp.metrics)
                )  # Add data for one trial
                trial.mark_completed()

    return exp


def get_branin_experiment_with_status_quo_trials(
    num_sobol_trials: int = 5,
    multi_objective: bool = False,
) -> Experiment:
    if multi_objective:
        exp = get_branin_experiment_with_multi_objective(
            with_batch=True,
            with_status_quo=True,
        )
    else:
        exp = get_branin_experiment(with_status_quo=True)
    exp._properties = {"owners": [DEFAULT_USER]}
    sobol = get_sobol(search_space=exp.search_space)
    for _ in range(num_sobol_trials):
        sobol_run = sobol.gen(n=1)
        t = exp.new_batch_trial().add_generator_run(sobol_run)
        t.add_status_quo_arm(weight=0.5)
        exp.attach_data(get_branin_data_batch(batch=t))
        t.run().mark_completed()
    return exp


def get_map_metric(
    name: str,
    noise_sd: float = 0.0,
    rate: float | None = None,
    decay_function_name: str = "exp_decay",
) -> BraninTimestampMapMetric:
    return BraninTimestampMapMetric(
        name=name,
        param_names=["x1", "x2"],
        noise_sd=noise_sd,
        rate=rate,
        lower_is_better=True,
        decay_function_name=decay_function_name,
    )


def get_branin_experiment_with_timestamp_map_metric(
    with_status_quo: bool = False,
    noise_sd: float = 0.0,
    rate: float | None = None,
    map_tracking_metric: bool = False,
    decay_function_name: str = "exp_decay",
    with_trials_and_data: bool = False,
    multi_objective: bool = False,
    has_objective_thresholds: bool = False,
    bounds: list[float] | None = None,
    with_choice_parameter: bool = False,
    with_outcome_constraint: bool = False,
) -> Experiment:
    """Returns an experiment with the search space including parameters

    Args:
        with_status_quo: Whether to include a status quo arm.
        noise_sd: Standard deviation of noise to add to the metric.
        rate: Rate of decay for the map metric.
        map_tracking_metric: Whether to include a tracking map metric.
        decay_function_name: Name of the decay function to use.
        with_trials_and_data: Whether to include trials and data.
        multi_objective: Whether to include multiple objectives and tracking metrics.
        has_objective_thresholds: For multi-objective experiments, toggles adding
            objective thresholds.
        bounds: For multi-objective experiments where has_objective_thresholds is True,
            bounds determines the precise objective thresholds.
        with_choice_parameter: Whether to include a choice parameter.
            If true, `x2` will be a ChoiceParameter.
        with_outcome_constraint: If True, adds an outcome constraint with an additional
            non-map Branin metric.


    Returns:
        A Branin single or multi-objective experiment with map metrics.
    """
    local_get_map_metric = partial(
        get_map_metric,
        noise_sd=noise_sd,
        rate=rate,
        decay_function_name=decay_function_name,
    )
    experiment_name = "branin_with_timestamp_map_metric"
    if with_outcome_constraint:
        outcome_constraints = [
            OutcomeConstraint(
                metric=BraninMetric(name="branin_constraint", param_names=["x1", "x2"]),
                op=ComparisonOp.LEQ,
                bound=100.0,
            )
        ]
    else:
        outcome_constraints = None
    if multi_objective:
        experiment_name = "multi_objective_" + experiment_name
        num_objectives = 2
        bounds = bounds or [99.0 for _ in range(num_objectives)]
        if has_objective_thresholds:
            objective_thresholds = [
                ObjectiveThreshold(
                    metric=local_get_map_metric(name=f"branin_map_{m}"),
                    bound=bound,
                    op=ComparisonOp.LEQ,
                    relative=False,
                )
                for m, bound in zip(range(num_objectives), bounds)
            ]
        else:
            objective_thresholds = None

        if map_tracking_metric:
            tracking_metrics = [
                local_get_map_metric(f"tracking_branin_map_{m}")
                for m in range(num_objectives)
            ]
        else:
            tracking_metrics = [BraninMetric(name="branin", param_names=["x1", "x2"])]

        objectives = [
            Objective(metric=local_get_map_metric(f"branin_map_{m}"))
            for m in range(num_objectives)
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives),
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
        )

    else:  # single objective case
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=local_get_map_metric(name="branin_map"),
                minimize=True,
            ),
            outcome_constraints=outcome_constraints,
        )

        if map_tracking_metric:
            tracking_metric = local_get_map_metric(name="tracking_branin_map")
        else:
            tracking_metric = BraninMetric(
                name="branin", param_names=["x1", "x2"], lower_is_better=True
            )

        tracking_metrics = [tracking_metric]

    exp = Experiment(
        name=experiment_name,
        search_space=get_branin_search_space(
            with_choice_parameter=with_choice_parameter
        ),
        optimization_config=optimization_config,
        tracking_metrics=cast(list[Metric], tracking_metrics),
        runner=SyntheticRunner(),
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    if with_status_quo:
        exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})

    if with_trials_and_data:
        # Add a couple trials with different number of timestamps.
        # Each fetch attaches data with a new timestamp / progression.
        # We end up with 4 rows of data for trial 0 and 2 for trial 1.
        exp.new_trial().add_arm(Arm(parameters={"x1": 0.0, "x2": 0.0})).run()
        for _ in range(2):
            exp.fetch_data()
        exp.new_trial().add_arm(Arm(parameters={"x1": 1.0, "x2": 1.0})).run()
        for _ in range(2):
            exp.fetch_data()
        # Add a trial with no data.
        exp.new_trial().add_arm(Arm(parameters={"x1": 2.0, "x2": 2.0})).run()

    return exp


def run_branin_experiment_with_generation_strategy(
    generation_strategy: GenerationStrategy,
    num_trials: int = 6,
    kwargs_for_get_branin_experiment: dict[str, Any] | None = None,
) -> Experiment:
    """Gets a Branin experiment using any given kwargs and runs
    num_trials trials using the given generation strategy."""
    kwargs_for_get_branin_experiment = kwargs_for_get_branin_experiment or {}
    exp = get_branin_experiment(**kwargs_for_get_branin_experiment)
    for _ in range(num_trials):
        gr = generation_strategy.gen_single_trial(n=1, experiment=exp)
        trial = exp.new_trial(generator_run=gr)
        trial.mark_running(no_runner_required=True)
        exp.attach_data(get_branin_data(trials=[trial]))
        trial.mark_completed()
    exp._properties = {"owners": [DEFAULT_USER]}
    return exp


def get_test_map_data_experiment(
    num_trials: int,
    num_fetches: int,
    num_complete: int,
    map_tracking_metric: bool = False,
    multi_objective: bool = False,
    bounds: list[float] | None = None,
    has_objective_thresholds: bool = False,
) -> Experiment:
    experiment = get_branin_experiment_with_timestamp_map_metric(
        rate=0.5,
        map_tracking_metric=map_tracking_metric,
        multi_objective=multi_objective,
        bounds=bounds,
        has_objective_thresholds=has_objective_thresholds,
    )
    for i in range(num_trials):
        trial = experiment.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
        trial.run()
    for _ in range(num_fetches):
        # each time we call fetch, we grab another timestamp
        experiment.fetch_data()
    for i in range(num_complete):
        experiment.trials[i].mark_as(status=TrialStatus.COMPLETED)
    return experiment


def get_multi_type_experiment(
    add_trial_type: bool = True, add_trials: bool = False, num_arms: int = 10
) -> MultiTypeExperiment:
    oc = OptimizationConfig(Objective(BraninMetric("m1", ["x1", "x2"]), minimize=True))
    experiment = MultiTypeExperiment(
        name="test_exp",
        search_space=get_branin_search_space(),
        default_trial_type="type1",
        default_runner=SyntheticRunner(dummy_metadata="dummy1"),
        optimization_config=oc,
        status_quo=Arm(parameters={"x1": 0.0, "x2": 0.0}),
    )
    experiment._properties = {"owners": [DEFAULT_USER]}
    experiment.add_trial_type(
        trial_type="type2", runner=SyntheticRunner(dummy_metadata="dummy2")
    )
    # Switch the order of variables so metric gives different results
    experiment.add_tracking_metric(
        BraninMetric("m2", ["x2", "x1"]), trial_type="type2", canonical_name="m1"
    )

    if add_trials and add_trial_type:
        generator = get_sobol(experiment.search_space)
        gr = generator.gen(num_arms)
        t1 = experiment.new_batch_trial(
            generator_run=gr, trial_type="type1", should_add_status_quo_arm=True
        )
        t2 = experiment.new_batch_trial(
            generator_run=gr, trial_type="type2", should_add_status_quo_arm=True
        )
        t1.run()
        t2.run()

    return experiment


def get_factorial_experiment(
    has_optimization_config: bool = True,
    with_batch: bool = False,
    with_status_quo: bool = False,
) -> Experiment:
    exp = Experiment(
        name="factorial_test_experiment",
        search_space=get_factorial_search_space(),
        optimization_config=(
            OptimizationConfig(
                objective=Objective(metric=get_factorial_metric(), minimize=False)
            )
            if has_optimization_config
            else None
        ),
        runner=SyntheticRunner(),
        is_test=True,
        tracking_metrics=[get_factorial_metric("secondary_metric")],
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    if with_status_quo:
        exp.status_quo = Arm(
            parameters={
                "factor1": "level11",
                "factor2": "level21",
                "factor3": "level31",
            }
        )

    if with_batch:
        factorial_generator = get_factorial(experiment=exp)
        # compute cardinality of discrete search space
        n = prod(
            len(assert_is_instance(p, ChoiceParameter).values)
            for p in exp.search_space.parameters.values()
        )
        factorial_run = factorial_generator.gen(n=n)
        exp.new_batch_trial(should_add_status_quo_arm=False).add_generator_run(
            factorial_run
        )

    return exp


def get_experiment_with_trial() -> Experiment:
    trial = get_trial()
    return trial.experiment


def get_experiment_with_batch_trial(
    constrain_search_space: bool = True, with_status_quo: bool = True
) -> Experiment:
    batch_trial = get_batch_trial(
        constrain_search_space=constrain_search_space, with_status_quo=with_status_quo
    )
    return batch_trial.experiment


def get_experiment_with_batch_and_single_trial() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.new_trial(generator_run=GeneratorRun(arms=[get_arm()]))
    return batch_trial.experiment


def get_experiment_with_trial_with_ttl() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.new_trial(
        generator_run=GeneratorRun(arms=[get_arm()]), ttl_seconds=1
    )
    return batch_trial.experiment


def get_experiment_with_data() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.attach_data(data=get_data())
    batch_trial.experiment.attach_data(data=get_data())
    batch_trial.experiment.attach_data(data=get_data())
    return batch_trial.experiment


def get_experiment_with_map_data() -> Experiment:
    experiment = get_experiment_with_map_data_type()
    experiment._properties = {"owners": [DEFAULT_USER]}
    experiment.new_trial()
    experiment.add_tracking_metric(MapMetric("ax_test_metric"))
    experiment.attach_data(data=get_map_data())
    return experiment


def get_experiment_with_multi_objective() -> Experiment:
    optimization_config = get_multi_objective_optimization_config()

    exp = Experiment(
        name="test_experiment_multi_objective",
        search_space=get_branin_search_space(),
        optimization_config=optimization_config,
        description="test experiment with multi objective",
        runner=SyntheticRunner(),
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    return exp


def get_status_quo_branin(
    with_fidelity_parameter: bool = False,
    with_str_choice_param: bool = False,
    with_derived_parameter: bool = False,
    with_fixed_parameter: bool = False,
    status_quo_unknown_parameters: bool = False,
) -> Arm:
    params: dict[str, TParamValue] = {"x1": 0.0, "x2": 0.0}
    if with_fixed_parameter:
        params["z"] = True
    if with_str_choice_param:
        params["str_param"] = "foo"
    if with_fidelity_parameter:
        params["fidelity"] = 1.0
    if with_derived_parameter:
        params["derived"] = 0.0
    if status_quo_unknown_parameters:
        params = dict.fromkeys(params)

    return Arm(name="status_quo", parameters=params)


def get_branin_experiment_with_multi_objective(
    has_optimization_config: bool = True,
    has_objective_thresholds: bool = False,
    with_batch: bool = False,
    with_status_quo: bool = False,
    status_quo_unknown_parameters: bool = False,
    with_fidelity_parameter: bool = False,
    num_objectives: int = 2,
    with_trial: bool = False,
    num_trial: int = 1,
    with_completed_trial: bool = False,
    with_completed_batch: bool = False,
    with_relative_constraint: bool = False,
    with_absolute_constraint: bool = False,
    with_choice_parameter: bool = False,
    with_fixed_parameter: bool = False,
    with_derived_parameter: bool = False,
) -> Experiment:
    optimization_config = (
        get_branin_multi_objective_optimization_config(
            has_objective_thresholds=has_objective_thresholds,
            num_objectives=num_objectives,
            with_relative_constraint=with_relative_constraint,
            with_absolute_constraint=with_absolute_constraint,
        )
        if has_optimization_config
        else None
    )
    exp = Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(
            with_fidelity_parameter=with_fidelity_parameter,
            with_choice_parameter=with_choice_parameter,
            with_fixed_parameter=with_fixed_parameter,
            with_derived_parameter=with_derived_parameter,
        ),
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
        is_test=True,
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    status_quo = (
        get_status_quo_branin(
            with_fidelity_parameter=with_fidelity_parameter,
            with_derived_parameter=with_derived_parameter,
            with_fixed_parameter=with_fixed_parameter,
            status_quo_unknown_parameters=status_quo_unknown_parameters,
        )
        if with_status_quo
        else None
    )

    exp.status_quo = status_quo
    outcome_names = list(exp.metrics)
    if with_batch or with_completed_batch:
        sobol_generator = get_sobol(search_space=exp.search_space, seed=TEST_SOBOL_SEED)
        sobol_run = sobol_generator.gen(n=5)
        trial = exp.new_batch_trial(
            generator_run=sobol_run, should_add_status_quo_arm=with_status_quo
        )

        if with_completed_batch:
            assert has_optimization_config
            trial.mark_running(no_runner_required=True)
            exp.attach_data(
                get_branin_data_multi_objective(
                    trial_indices=[trial.index],
                    arm_names=[arm.name for arm in trial.arms],
                    outcomes=outcome_names,
                )
            )  # Add data for one trial
            trial.mark_completed()

    if with_trial or with_completed_trial:
        sobol_generator = get_sobol(search_space=exp.search_space)
        for _ in range(num_trial):
            sobol_run = sobol_generator.gen(n=1)
            trial = exp.new_trial(generator_run=sobol_run)

            if with_completed_trial:
                assert has_optimization_config
                trial.mark_running(no_runner_required=True)
                exp.attach_data(
                    get_branin_data_multi_objective(
                        trial_indices=[trial.index], outcomes=outcome_names
                    )
                )  # Add data for one trial
                trial.mark_completed()

    return exp


def get_branin_with_multi_task(with_multi_objective: bool = False) -> Experiment:
    exp = Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(),
        optimization_config=(
            get_branin_multi_objective_optimization_config(
                has_objective_thresholds=True,
            )
            if with_multi_objective
            else get_branin_optimization_config()
        ),
        runner=SyntheticRunner(),
        is_test=True,
    )
    exp._properties = {"owners": [DEFAULT_USER]}

    exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0}, name="status_quo")

    sobol_generator = get_sobol(search_space=exp.search_space, seed=TEST_SOBOL_SEED)
    sobol_run = sobol_generator.gen(n=5)
    exp.new_batch_trial(should_add_status_quo_arm=True).add_generator_run(sobol_run)
    none_throws(exp.trials.get(0)).run()
    exp.new_batch_trial(should_add_status_quo_arm=True).add_generator_run(sobol_run)
    none_throws(exp.trials.get(1)).run()

    return exp


def get_experiment_with_scalarized_objective_and_outcome_constraint() -> Experiment:
    objective = get_scalarized_objective()
    outcome_constraints = [
        get_outcome_constraint(),
        get_scalarized_outcome_constraint(),
    ]
    optimization_config = OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )
    experiment = Experiment(
        name="test_experiment_scalarized_objective and outcome constraint",
        search_space=get_search_space(),
        optimization_config=optimization_config,
        status_quo=get_status_quo(),
        description="test experiment with scalarized objective and outcome constraint",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )
    experiment._properties = {"owners": [DEFAULT_USER]}
    return experiment


def get_hierarchical_search_space_experiment(
    num_observations: int = 0,
    use_map_data: bool = False,
) -> Experiment:
    """
    Create an experiment with a hierarchical search space and optional observations.

    Args:
        num_observations: The number of trials in the experiment.
        use_map_data: Whether to use `MapData` or `Data` when constructing the
            experiment. This flag is for testing the transform `MapKeyToFloat`, which
            is applied to the search space only if the experiment has map data.

    Returns:
        An experiment with a hierarchical search space and some optional observations.

    NOTE: We have fixed the random seeds in the Sobol generator and `torch.rand`.
    Otherwise, `MapKeyToFloatTransformTest` is flaky.
    """
    experiment = Experiment(
        name="test_experiment_hss",
        description="test experiment with hierarchical search space",
        search_space=get_hierarchical_search_space(),
        optimization_config=get_optimization_config(),
    )
    experiment._properties = {"owners": [DEFAULT_USER]}
    sobol_generator = get_sobol(
        search_space=experiment.search_space, seed=TEST_SOBOL_SEED
    )
    for i in range(num_observations):
        trial = experiment.new_trial(generator_run=sobol_generator.gen(1))
        trial.mark_running(no_runner_required=True)

        torch_rnd_generator = torch.Generator().manual_seed(TORCH_RAND_SEED)
        outcomes = torch.rand(2, generator=torch_rnd_generator).tolist()

        dict_step = {"step": i} if use_map_data else {}
        df = pd.DataFrame.from_records(
            [
                {
                    **{
                        "arm_name": f"{i}_0",
                        "metric_name": f"m{j + 1}",
                        "mean": o,
                        "sem": None,
                        "trial_index": i,
                        "metric_signature": f"m{j + 1}",
                    },
                    **dict_step,
                }
                for j, o in enumerate(outcomes)
            ]
        )
        data = MapData(df=df) if use_map_data else Data(df=df)

        experiment.attach_data(data)
        trial.mark_completed()

    return experiment


def get_experiment_with_observations(
    observations: Sequence[Sequence[float]],
    minimize: bool = False,
    scalarized: bool = False,
    constrained: bool = False,
    with_tracking_metrics: bool = False,
    search_space: SearchSpace | None = None,
    parameterizations: Sequence[Mapping[str, TParamValue]] | None = None,
    sems: list[list[float]] | None = None,
    optimization_config: OptimizationConfig | None = None,
    candidate_metadata: Sequence[TCandidateMetadata] | None = None,
    additional_data_columns: Sequence[Mapping[str, Any]] | None = None,
    signature_suffix: bool = False,
    status_quo: Arm | None = None,
) -> Experiment:
    if observations:
        multi_objective = (len(observations[0]) - constrained) > 1
    else:
        multi_objective = False
    if multi_objective and optimization_config is None:
        metrics = [
            Metric(
                name="m1",
                lower_is_better=minimize,
                signature_override="m1_sig" if signature_suffix else None,
            ),
            Metric(
                name="m2",
                lower_is_better=False,
                signature_override="m2_sig" if signature_suffix else None,
            ),
        ]
        if scalarized:
            optimization_config = OptimizationConfig(
                objective=ScalarizedObjective(metrics)
            )
            if constrained:
                raise NotImplementedError
        else:
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[Objective(metric=metric) for metric in metrics]
                ),
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=metrics[i],
                        bound=0.0,
                        relative=False,
                        op=(
                            ComparisonOp.LEQ
                            if metrics[i].lower_is_better
                            else ComparisonOp.GEQ
                        ),
                    )
                    for i in [0, 1]
                ],
                outcome_constraints=(
                    [
                        OutcomeConstraint(
                            metric=Metric(
                                name="m3",
                                signature_override="m3_sig"
                                if signature_suffix
                                else None,
                            ),
                            op=ComparisonOp.GEQ,
                            bound=0.0,
                            relative=False,
                        )
                    ]
                    if constrained
                    else None
                ),
            )
    elif optimization_config is None:
        if scalarized:
            raise NotImplementedError
        objective = Objective(
            metric=Metric(
                name="m1",
                signature_override="m1_sig" if signature_suffix else None,
            ),
            minimize=minimize,
        )
        if constrained:
            constraint = OutcomeConstraint(
                metric=Metric(
                    name="m2",
                    signature_override="m2_sig" if signature_suffix else None,
                ),
                op=ComparisonOp.GEQ,
                bound=0.0,
                relative=False,
            )
            optimization_config = OptimizationConfig(
                objective=objective, outcome_constraints=[constraint]
            )
        else:
            optimization_config = OptimizationConfig(objective=objective)
    search_space = search_space or get_search_space_for_range_values(min=0.0, max=1.0)
    exp = Experiment(
        search_space=search_space,
        optimization_config=optimization_config,
        tracking_metrics=(
            [
                Metric(
                    name=f"m{len(observations[0])}",
                    lower_is_better=False,
                    signature_override=f"m{len(observations[0])}_sig"
                    if signature_suffix
                    else None,
                )
            ]
            if with_tracking_metrics
            else None
        ),
        runner=SyntheticRunner(),
        is_test=True,
        status_quo=status_quo,
    )
    metrics = sorted(exp.metrics)
    metric_to_signature_map = {name: m.signature for name, m in exp.metrics.items()}
    exp._properties = {"owners": [DEFAULT_USER]}
    sobol_generator = get_sobol(search_space=search_space)
    for i, obs_i in enumerate(observations):
        sems_i = sems[i] if sems is not None else [None] * len(obs_i)
        if parameterizations is not None:
            arm = Arm(parameters=parameterizations[i])
        else:
            arm = sobol_generator.gen(1).arms[0]
        if candidate_metadata is not None:
            metadata = {arm.signature: candidate_metadata[i]}
        else:
            metadata = None
        if additional_data_columns is not None:
            additional_cols = additional_data_columns[i]
        else:
            additional_cols = {}
        trial = exp.new_trial(
            generator_run=GeneratorRun(
                arms=[arm], candidate_metadata_by_arm_signature=metadata
            )
        )

        data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": trial.arms[0].name,
                        "metric_name": m,
                        "mean": o,
                        "sem": s,
                        "trial_index": trial.index,
                        **additional_cols,
                        "metric_signature": metric_to_signature_map[m],
                    }
                    for m, o, s in zip(metrics, obs_i, sems_i, strict=True)
                ]
            )
        )
        exp.attach_data(data)
        trial.run().complete()
    return exp


def get_high_dimensional_branin_experiment(
    with_batch: bool = False, with_status_quo: bool = False
) -> Experiment:
    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=-5.0,
                upper=10.0,
            )
            for i in range(25)
        ]
        + [
            RangeParameter(
                name=f"x{i + 25}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=15.0,
            )
            for i in range(25)
        ],
    )

    sq_parameters = {f"x{i}": 1.0 if i < 25 else 2.0 for i in range(50)}
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="objective",
                param_names=["x19", "x44"],
            ),
            minimize=True,
        )
    )

    exp = Experiment(
        name="high_dimensional_branin_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
        status_quo=Arm(sq_parameters) if with_status_quo else None,
    )
    exp._properties = {"owners": [DEFAULT_USER]}
    if with_batch:
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=15)
        exp.new_batch_trial().add_generator_run(sobol_run)
    return exp


def get_auxiliary_experiment() -> AuxiliaryExperiment:
    return AuxiliaryExperiment(experiment=get_experiment_with_data())


def get_online_experiments() -> list[Experiment]:
    """
    Returns a List of Branin Experiments which resemble those we see in an online
    context. This means BatchTrial experiments with both single- and multi-objective
    optimization configs and with data attached and at least one trial in a CANDIDATE
    state.

    We also include combinations with and without choice parameters, fixed_parameters,
    absolute parameter constraints, and relative parameter constraints.
    """

    single_objective = [
        get_branin_experiment(
            with_batch=True,
            num_batch_trial=2,
            num_arms_per_trial=10,
            with_completed_batch=True,
            with_status_quo=True,
            status_quo_unknown_parameters=True,
            with_derived_parameter=True,
            with_choice_parameter=with_choice_parameter,
            with_parameter_constraint=with_parameter_constraint,
            with_relative_constraint=True,
        )
        for (
            with_choice_parameter,
            with_parameter_constraint,
        ) in itertools.product([True, False], repeat=2)
    ]

    multi_objective = [
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_batch=True,
            with_status_quo=True,
            with_completed_batch=True,
            has_objective_thresholds=has_objective_thresholds,
            with_choice_parameter=with_choice_parameter,
            with_fixed_parameter=with_fixed_parameter,
            with_derived_parameter=True,
            with_relative_constraint=True,
            with_absolute_constraint=False,
        )
        for (
            has_objective_thresholds,
            with_choice_parameter,
            with_fixed_parameter,
        ) in itertools.product([True, False], repeat=3)
    ]

    experiments = [*single_objective, *multi_objective]
    _configure_online_experiments(experiments=experiments)

    return experiments


def get_online_experiments_subset() -> list[Experiment]:
    """
    Set of 4 experiments includes: 1 single objective exp with choice parameter,
    parameter constraint, and relative constriant. 3 multi-objective experiments
    with (a) choice param, fixed param, relative and absolute constraint, (b)
    fixed param and relative constraint (c) no constraints but both fixed and
    choice param
    """
    experiments = []
    experiments.append(
        get_branin_experiment(
            with_batch=True,
            num_batch_trial=2,
            num_arms_per_trial=10,
            with_completed_batch=True,
            with_status_quo=True,
            status_quo_unknown_parameters=True,
            with_choice_parameter=True,
            with_parameter_constraint=True,
            with_relative_constraint=True,
            with_derived_parameter=True,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_batch=True,
            with_status_quo=True,
            with_completed_batch=True,
            has_objective_thresholds=True,
            with_choice_parameter=True,
            with_fixed_parameter=True,
            with_relative_constraint=True,
            with_absolute_constraint=True,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_batch=True,
            with_status_quo=True,
            with_completed_batch=True,
            has_objective_thresholds=True,
            with_choice_parameter=False,
            with_fixed_parameter=True,
            with_relative_constraint=True,
            with_absolute_constraint=False,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_batch=True,
            with_status_quo=True,
            with_completed_batch=True,
            has_objective_thresholds=True,
            with_choice_parameter=True,
            with_fixed_parameter=True,
            with_relative_constraint=False,
            with_absolute_constraint=False,
        )
    )
    _configure_online_experiments(experiments=experiments)
    return experiments


def _configure_online_experiments(experiments: list[Experiment]) -> None:
    for experiment in experiments:
        sobol_generator = get_sobol(search_space=experiment.search_space)

        # Add a candidate to each Experiment
        sobol_run = sobol_generator.gen(n=len(experiment.trials[0].arms))
        trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
        trial.add_generator_run(sobol_run)

        # Add a RUNNING trial to each Experiment
        sobol_run = sobol_generator.gen(n=len(experiment.trials[0].arms))
        trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
        trial.add_generator_run(sobol_run)
        trial.mark_running(no_runner_required=True)

        # Add a FAILED trial to each Experiment
        sobol_run = sobol_generator.gen(n=len(experiment.trials[0].arms))
        trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
        trial.add_generator_run(sobol_run)
        trial.mark_running(no_runner_required=True)
        trial.mark_failed()

        # Add an ABANDONED trial to each Experiment
        sobol_run = sobol_generator.gen(n=len(experiment.trials[0].arms))
        trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
        trial.add_generator_run(sobol_run)
        trial.mark_abandoned()

        # Add a custom arm to each Experiment
        sobol_run = sobol_generator.gen(n=len(experiment.trials[0].arms))
        trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
        # Detatch the arms from the GeneratorRun so they appear as custom arms
        trial.add_arms_and_weights(arms=sobol_run.arms)


def get_offline_experiments() -> list[Experiment]:
    """
    Returns a List of Experiments which resemble those we see in an offline context.
    This means single-arm Trial experiments with both single- and multi-objective
    optimization configs, with data attached.

    We also include combinations with and without choice parameters, fixed_parameters,
    absolute parameter constraints, and relative parameter constraints.
    """
    single_objective = [
        get_branin_experiment(
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            with_choice_parameter=with_choice_parameter,
            with_parameter_constraint=with_parameter_constraint,
            with_derived_parameter=True,
        )
        for (
            with_choice_parameter,
            with_parameter_constraint,
        ) in itertools.product([True, False], repeat=2)
    ]

    multi_objective = [
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            has_objective_thresholds=has_objective_thresholds,
            with_absolute_constraint=with_absolute_constraint,
            with_choice_parameter=with_choice_parameter,
            with_fixed_parameter=with_fixed_parameter,
            with_derived_parameter=True,
        )
        for (
            has_objective_thresholds,
            with_absolute_constraint,
            with_choice_parameter,
            with_fixed_parameter,
        ) in itertools.product([True, False], repeat=4)
    ]

    experiments = [*single_objective, *multi_objective]
    _configure_offline_experiments(experiments=experiments)

    return experiments


def get_offline_experiments_subset() -> list[Experiment]:
    """
    Set of 4 experiments that include:
    1. Single objective with choice param and param constraint
    2. Mulit-objective with objective threshold, absolute constraint, choice param,
        and fixed param
    3. Mulit-objective with no thresholds, constraint, or special params
    4. Mulit-objective with objective threshold and fixed param
    """
    experiments = []
    experiments.append(
        get_branin_experiment(
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            with_choice_parameter=True,
            with_parameter_constraint=True,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            has_objective_thresholds=True,
            with_absolute_constraint=True,
            with_choice_parameter=True,
            with_fixed_parameter=True,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            has_objective_thresholds=False,
            with_absolute_constraint=False,
            with_choice_parameter=False,
            with_fixed_parameter=False,
        )
    )
    experiments.append(
        get_branin_experiment_with_multi_objective(
            num_objectives=3,
            with_trial=True,
            num_trial=10,
            with_completed_trial=True,
            with_status_quo=False,
            has_objective_thresholds=True,
            with_absolute_constraint=False,
            with_choice_parameter=False,
            with_fixed_parameter=True,
        )
    )
    _configure_offline_experiments(experiments=experiments)
    return experiments


def _configure_offline_experiments(experiments: list[Experiment]) -> None:
    for experiment in experiments:
        sobol_generator = get_sobol(search_space=experiment.search_space)

        # Add a candidate to each Experiment
        trial = experiment.new_trial(generator_run=sobol_generator.gen(n=1))

        # Add a RUNNING trial to each Experiment
        sobol_generator = get_sobol(search_space=experiment.search_space)

        trial.mark_running(no_runner_required=True)

        # Add a FAILED trial to each Experiment
        trial = experiment.new_trial(generator_run=sobol_generator.gen(n=1))
        trial.mark_running(no_runner_required=True)
        trial.mark_failed()

        # Add an ABANDONED trial to each Experiment
        trial = experiment.new_trial(generator_run=sobol_generator.gen(n=1))
        trial.mark_abandoned()

        # Add a custom arm to each Experiment
        sobol_run = sobol_generator.gen(n=1)
        trial = experiment.new_trial()
        # Detatch the arms from the GeneratorRun so they appear as custom arms
        trial.add_arm(arm=sobol_run.arms[0])


##############################
# Search Spaces
##############################


def get_search_space(constrain_search_space: bool = True) -> SearchSpace:
    parameters: list[Parameter] = [
        get_range_parameter(),
        get_range_parameter2(),
        get_choice_parameter(),
        get_fixed_parameter(),
        get_derived_parameter(),
    ]
    parameter_constraints = []
    if constrain_search_space:
        parameter_constraints = [
            get_order_constraint(),
            get_parameter_constraint(),
            get_sum_constraint1(),
        ]
    return SearchSpace(
        parameters=parameters,
        parameter_constraints=parameter_constraints,
    )


def get_branin_search_space(
    with_fidelity_parameter: bool = False,
    with_choice_parameter: bool = False,
    with_str_choice_param: bool = False,
    with_derived_parameter: bool = False,
    with_parameter_constraint: bool = False,
    with_fixed_parameter: bool = False,
) -> SearchSpace:
    parameters: list[Parameter] = [
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        (
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                values=[float(x) for x in range(0, 16)],
            )
            if with_choice_parameter
            else RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
            )
        ),
    ]
    if with_str_choice_param:
        parameters.append(
            ChoiceParameter(
                name="str_param",
                parameter_type=ParameterType.STRING,
                values=["foo", "bar", "baz"],
            )
        )
    if with_fidelity_parameter:
        parameters.append(
            RangeParameter(
                name="fidelity",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
                is_fidelity=True,
                target_value=1.0,
            )
        )

    if with_fixed_parameter:
        parameters.append(get_fixed_parameter())
    if with_derived_parameter:
        parameters.append(
            DerivedParameter(
                name="derived",
                parameter_type=ParameterType.FLOAT,
                expression_str="x1 + x2",
            )
        )

    if with_parameter_constraint:
        constraints = [ParameterConstraint(inequality="x1 <= 15")]
    else:
        constraints = None

    return SearchSpace(parameters=parameters, parameter_constraints=constraints)


def get_factorial_search_space() -> SearchSpace:
    return SearchSpace(
        parameters=[
            ChoiceParameter(
                name="factor1",
                parameter_type=ParameterType.STRING,
                values=["level11", "level12", "level13"],
            ),
            ChoiceParameter(
                name="factor2",
                parameter_type=ParameterType.STRING,
                values=["level21", "level22"],
            ),
            ChoiceParameter(
                name="factor3",
                parameter_type=ParameterType.STRING,
                values=["level31", "level32", "level33", "level34"],
            ),
        ]
    )


def get_large_factorial_search_space(
    num_levels: int = 10, num_parameters: int = 6
) -> SearchSpace:
    return SearchSpace(
        parameters=[
            ChoiceParameter(
                name=f"factor{j}",
                parameter_type=ParameterType.STRING,
                values=[f"level1{i}" for i in range(num_levels)],
            )
            for j in range(num_parameters)
        ]
    )


def get_large_ordinal_search_space(
    n_ordinal_choice_parameters: int,
    n_continuous_range_parameters: int,
) -> SearchSpace:
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(n_continuous_range_parameters)
        ]
        + [
            ChoiceParameter(
                name=f"y{i}",
                parameter_type=ParameterType.INT,
                values=[2, 4, 8, 16],
            )
            for i in range(n_ordinal_choice_parameters)
        ]
    )


def get_search_space_for_value(val: float = 3.0) -> SearchSpace:
    return SearchSpace([FixedParameter("x", ParameterType.FLOAT, val)])


def get_search_space_for_range_value(min: float = 3.0, max: float = 6.0) -> SearchSpace:
    return SearchSpace([RangeParameter("x", ParameterType.FLOAT, min, max)])


def get_search_space_for_range_values(
    min: float = 3.0, max: float = 6.0, parameter_names: list[str] | None = None
) -> SearchSpace:
    if parameter_names is None:
        parameter_names = ["x", "y"]
    return SearchSpace(
        [
            RangeParameter(name, ParameterType.FLOAT, min, max)
            for name in parameter_names
        ]
    )


def get_discrete_search_space() -> SearchSpace:
    return SearchSpace(
        [
            RangeParameter("x", ParameterType.INT, 0, 3),
            RangeParameter("y", ParameterType.INT, 5, 7),
            ChoiceParameter("z", ParameterType.STRING, ["red", "panda", "bear"]),
        ]
    )


def get_small_discrete_search_space() -> SearchSpace:
    return SearchSpace(
        [
            RangeParameter("x", ParameterType.INT, 0, 1),
            ChoiceParameter("y", ParameterType.STRING, ["red", "panda"]),
        ]
    )


def get_search_space_with_choice_parameters(
    num_ordered_parameters: int = 2,
    num_unordered_choices: int = 5,
) -> SearchSpace:
    parameters = []
    for i in range(num_ordered_parameters):
        parameters.append(
            ChoiceParameter(
                name=f"ordered_{i}",
                parameter_type=ParameterType.INT,
                values=list(range(10)),
                is_ordered=True,
            )
        )
    parameters.append(
        ChoiceParameter(
            name="unordered",
            parameter_type=ParameterType.INT,
            values=list(range(num_unordered_choices)),
            is_ordered=False,
        )
    )
    return SearchSpace(parameters=parameters)


def get_hierarchical_search_space(
    with_fixed_parameter: bool = False,
) -> SearchSpace:
    parameters: list[Parameter] = [
        get_model_parameter(with_fixed_parameter=with_fixed_parameter),
        get_lr_parameter(),
        get_l2_reg_weight_parameter(),
        get_num_boost_rounds_parameter(),
    ]
    if with_fixed_parameter:
        parameters.append(get_fixed_parameter())
    return SearchSpace(parameters=parameters)


##############################
# Trials
##############################


def get_batch_trial(
    abandon_arm: bool = True,
    experiment: Experiment | None = None,
    constrain_search_space: bool = True,
    with_status_quo: bool = True,
) -> BatchTrial:
    experiment = experiment or get_experiment(
        constrain_search_space=constrain_search_space, with_status_quo=with_status_quo
    )
    batch = experiment.new_batch_trial(should_add_status_quo_arm=with_status_quo)
    arms = get_arms_from_dict(get_arm_weights1())
    weights = get_weights_from_dict(get_arm_weights1())
    batch.add_arms_and_weights(arms=arms, weights=weights)
    if abandon_arm:
        batch.mark_arm_abandoned(batch.arms[2].name, "abandoned reason")
    experiment.runner = SyntheticRunner()
    batch.should_add_status_quo_arm = True
    return batch


def get_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial(ttl_seconds=72)
    arm = get_arms_from_dict(get_arm_weights1())[0]
    trial.add_arm(arm)
    experiment.runner = SyntheticRunner()
    trial.update_run_metadata({"workflow_run_id": [12345]})
    return trial


class TestTrial(BaseTrial):
    "Trial class to test unsupported trial type error"

    _arms: list[Arm] = []

    def __repr__(self) -> str:
        return "test"

    def _get_candidate_metadata(self, arm_name: str) -> dict[str, Any] | None:
        return None

    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> dict[str, dict[str, Any] | None]:
        return {"test": None}

    def abandoned_arms(self) -> str:
        return "test"

    @property
    def arms(self) -> list[Arm]:
        return self._arms

    @arms.setter
    def arms(self, val: list[Arm]) -> None:
        self._arms = val

    def arms_by_name(self) -> str:
        return "test"

    def generator_runs(self) -> str:
        return "test"

    def add_generator_run(self, generator_run: GeneratorRun) -> Self:
        """Add a generator run to the trial.

        The arms and weights from the generator run will be merged with
        the existing arms and weights on the trial, and the generator run
        object will be linked to the trial for tracking.

        Args:
            generator_run: The generator run to be added.

        Returns:
            The trial instance.
        """
        return self

    def add_arm(
        self, arm: Arm, candidate_metadata: dict[str, Any] | None = None
    ) -> Self:
        """Add arm to the trial.

        Returns:
            The trial instance.
        """
        return self


def get_non_failed_arm_names(experiment: Experiment) -> set[str]:
    """Get the names of all arms from non-failed trials."""
    return {
        arm.name
        for trial in experiment.trials.values()
        if trial.status != TrialStatus.FAILED
        for arm in trial.arms
    }


##############################
# Parameters
##############################


def get_range_parameter() -> RangeParameter:
    return RangeParameter(
        name="w",
        parameter_type=ParameterType.FLOAT,
        lower=0.5,
        upper=5.5,
        log_scale=False,
        digits=5,
    )


def get_range_parameter2() -> RangeParameter:
    return RangeParameter(name="x", parameter_type=ParameterType.INT, lower=1, upper=10)


def get_choice_parameter() -> ChoiceParameter:
    return ChoiceParameter(
        name="y",
        parameter_type=ParameterType.STRING,
        # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for 4th
        # parameter `values` to call
        # `ax.core.parameter.ChoiceParameter.__init__` but got `List[str]`.
        values=["foo", "bar", "baz"],
        sort_values=False,
        is_ordered=False,
    )


def get_ordered_choice_parameter() -> ChoiceParameter:
    return ChoiceParameter(
        name="y",
        parameter_type=ParameterType.INT,
        # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for 4th
        # parameter `values` to call
        # `ax.core.parameter.ChoiceParameter.__init__` but got `List[str]`.
        values=[1, 2, 3],
        is_ordered=True,
    )


def get_sorted_choice_parameter() -> ChoiceParameter:
    return ChoiceParameter(
        name="y",
        parameter_type=ParameterType.STRING,
        values=["2", "1", "3"],
        is_ordered=True,
        sort_values=True,
    )


def get_task_choice_parameter() -> ChoiceParameter:
    return ChoiceParameter(
        name="y",
        parameter_type=ParameterType.INT,
        values=[1, 2, 3],
        is_task=True,
        target_value=1,
    )


def get_hierarchical_choice_parameter(parameter_type: ParameterType) -> ChoiceParameter:
    if parameter_type == ParameterType.BOOL:
        values = [True, False]
    elif parameter_type == ParameterType.INT:
        values = [0, 1]
    elif parameter_type == ParameterType.FLOAT:
        values = [0.0, 1.0]
    else:
        values = ["yee", "haw"]

    return ChoiceParameter(
        name="x",
        parameter_type=parameter_type,
        values=values,  # pyre-ignore [6]
        is_ordered=True,
        dependents={values[0]: ["y"], values[1]: ["z"]},
    )


def get_fixed_parameter(with_dependents: bool = False) -> FixedParameter:
    return FixedParameter(
        name="z",
        parameter_type=ParameterType.BOOL,
        value=True,
        dependents={True: ["y"]} if with_dependents else None,  # pyre-ignore [6]
    )


def get_derived_parameter() -> DerivedParameter:
    return DerivedParameter(
        name="d", parameter_type=ParameterType.FLOAT, expression_str="2.0 * w + 1.0"
    )


def get_model_parameter(with_fixed_parameter: bool = False) -> ChoiceParameter:
    return ChoiceParameter(
        name="model",
        parameter_type=ParameterType.STRING,
        values=["Linear", "XGBoost"],
        dependents={
            "Linear": ["learning_rate", "l2_reg_weight"],
            "XGBoost": (
                ["num_boost_rounds", "z"]
                if with_fixed_parameter
                else ["num_boost_rounds"]
            ),
        },
    )


def get_lr_parameter() -> RangeParameter:
    return RangeParameter(
        name="learning_rate",
        parameter_type=ParameterType.FLOAT,
        lower=0.001,
        upper=0.1,
    )


def get_l2_reg_weight_parameter() -> RangeParameter:
    return RangeParameter(
        name="l2_reg_weight",
        parameter_type=ParameterType.FLOAT,
        lower=0.00001,
        upper=0.001,
    )


def get_num_boost_rounds_parameter() -> RangeParameter:
    return RangeParameter(
        name="num_boost_rounds",
        parameter_type=ParameterType.INT,
        lower=10,
        upper=20,
    )


##############################
# Parameter Constraints
##############################


def get_order_constraint() -> ParameterConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return ParameterConstraint(inequality=f"{x.name} <= {w.name}")


def get_parameter_constraint(
    param_x: str = "x", param_y: str = "w"
) -> ParameterConstraint:
    return ParameterConstraint(inequality=f"{param_x} - {param_y} <= 1")


def get_sum_constraint1() -> ParameterConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return ParameterConstraint(inequality=f"{x.name} + {w.name} >= 10")


def get_sum_constraint2() -> ParameterConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return ParameterConstraint(inequality=f"{x.name} + {w.name} <= 10")


##############################
# Metrics
##############################


def get_metric() -> Metric:
    return Metric(name="m1", properties={"prop": "val"})


def get_branin_metric(
    name: str = "branin", lower_is_better: bool = True
) -> BraninMetric:
    param_names = ["x1", "x2"]
    return BraninMetric(
        name=name,
        param_names=param_names,
        noise_sd=0.01,
        lower_is_better=lower_is_better,
    )


def get_hartmann_metric(name: str = "hartmann") -> Hartmann6Metric:
    param_names = [f"x{idx + 1}" for idx in range(6)]
    return Hartmann6Metric(name=name, param_names=param_names, noise_sd=0.01)


def get_factorial_metric(name: str = "success_metric") -> FactorialMetric:
    coefficients: dict[str, dict[TParamValue, float]] = {
        "factor1": {"level11": 0.1, "level12": 0.2, "level13": 0.3},
        "factor2": {"level21": 0.1, "level22": 0.2},
        "factor3": {"level31": 0.1, "level32": 0.2, "level33": 0.3, "level34": 0.4},
    }
    return FactorialMetric(name=name, coefficients=coefficients, batch_size=int(1e4))


##############################
# Outcome Constraints
##############################


def get_objective_threshold(
    metric_name: str = "m1",
    bound: float = -0.25,
    comparison_op: ComparisonOp = ComparisonOp.GEQ,
) -> ObjectiveThreshold:
    return ObjectiveThreshold(
        metric=Metric(name=metric_name), bound=bound, op=comparison_op
    )


def get_outcome_constraint(
    metric: Metric | None = None, relative: bool = True, bound: float = -0.25
) -> OutcomeConstraint:
    if metric is None:
        metric = Metric(name="m2")
    return OutcomeConstraint(
        metric=metric, op=ComparisonOp.GEQ, bound=bound, relative=relative
    )


def get_scalarized_outcome_constraint() -> ScalarizedOutcomeConstraint:
    return ScalarizedOutcomeConstraint(
        metrics=[Metric(name="oc_m3"), Metric(name="oc_m4")],
        weights=[0.2, 0.8],
        op=ComparisonOp.GEQ,
        bound=-0.25,
    )


def get_branin_outcome_constraint(name: str = "branin") -> OutcomeConstraint:
    return OutcomeConstraint(
        metric=get_branin_metric(name=name), op=ComparisonOp.LEQ, bound=0.0
    )


##############################
# Objectives
##############################


def get_objective(minimize: bool = False) -> Objective:
    return Objective(metric=Metric(name="m1"), minimize=minimize)


def get_map_objective(minimize: bool = False) -> Objective:
    return Objective(metric=MapMetric(name="m1"), minimize=minimize)


def get_multi_objective() -> MultiObjective:
    return MultiObjective(
        objectives=[
            Objective(metric=Metric(name="m1"), minimize=False),
            Objective(metric=Metric(name="m3", lower_is_better=True), minimize=True),
        ],
    )


def get_custom_multi_objective() -> MultiObjective:
    return MultiObjective(
        objectives=[
            Objective(
                metric=CustomTestMetric(name="m1", test_attribute="test"),
                minimize=False,
            ),
            Objective(
                metric=CustomTestMetric(
                    name="m3", lower_is_better=True, test_attribute="test"
                ),
                minimize=True,
            ),
        ],
    )


def get_many_branin_objective_opt_config(
    n_objectives: int,
) -> MultiObjectiveOptimizationConfig:
    return MultiObjectiveOptimizationConfig(
        objective=MultiObjective(
            objectives=[get_branin_objective(name=f"m{i}") for i in range(n_objectives)]
        )
    )


def get_scalarized_objective() -> Objective:
    return ScalarizedObjective(
        metrics=[Metric(name="m1"), Metric(name="m3")],
        weights=[1.0, 2.0],
        minimize=False,
    )


def get_branin_objective(name: str = "branin", minimize: bool = False) -> Objective:
    return Objective(
        metric=get_branin_metric(name=name, lower_is_better=minimize), minimize=minimize
    )


def get_branin_multi_objective(num_objectives: int = 2) -> MultiObjective:
    objectives = [
        Objective(metric=get_branin_metric(name="branin_a"), minimize=True),
        Objective(metric=get_branin_metric(name="branin_b"), minimize=True),
    ]
    if num_objectives == 3:
        objectives.append(
            Objective(metric=get_branin_metric(name="branin_c"), minimize=True)
        )
    return MultiObjective(objectives=objectives)


##############################
# Optimization Configs
##############################


def get_optimization_config(
    outcome_constraint: bool = True, relative: bool = True
) -> OptimizationConfig:
    objective = get_objective()
    outcome_constraints = (
        [get_outcome_constraint(relative=relative)] if outcome_constraint else []
    )
    return OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )


def get_map_optimization_config() -> OptimizationConfig:
    objective = get_map_objective()
    return OptimizationConfig(objective=objective)


def get_multi_objective_optimization_config(
    custom_metric: bool = False,
    relative: bool = True,
    outcome_constraint: bool = True,
) -> MultiObjectiveOptimizationConfig:
    objective = get_custom_multi_objective() if custom_metric else get_multi_objective()
    outcome_constraints = (
        [get_outcome_constraint(relative=relative)] if outcome_constraint else []
    )
    objective_thresholds = [
        get_objective_threshold(metric_name="m1"),
        get_objective_threshold(metric_name="m3", comparison_op=ComparisonOp.LEQ),
    ]
    return MultiObjectiveOptimizationConfig(
        objective=objective,
        outcome_constraints=outcome_constraints,
        objective_thresholds=objective_thresholds,
    )


def get_optimization_config_no_constraints(
    minimize: bool = False,
) -> OptimizationConfig:
    return OptimizationConfig(
        objective=Objective(metric=Metric("test_metric"), minimize=minimize)
    )


def get_branin_optimization_config(
    minimize: bool = False,
    with_relative_constraint: bool = False,
    with_absolute_constraint: bool = False,
) -> OptimizationConfig:
    outcome_constraint = []
    if with_relative_constraint:
        outcome_constraint.append(
            get_outcome_constraint(
                metric=get_branin_metric(name="branin_d"), relative=True
            )
        )
    if with_absolute_constraint:
        outcome_constraint.append(
            get_outcome_constraint(
                metric=get_branin_metric(name="branin_e"), relative=False
            )
        )
    return OptimizationConfig(
        objective=get_branin_objective(minimize=minimize),
        outcome_constraints=outcome_constraint,
    )


def _validate_num_outcomes(num_outcomes: int) -> None:
    if 2 > num_outcomes or num_outcomes > 5:
        raise NotImplementedError("Only 2-5 outcomes are supported.")


def get_branin_multi_objective_optimization_config(
    has_objective_thresholds: bool = False,
    num_objectives: int = 2,
    with_relative_constraint: bool = False,
    with_absolute_constraint: bool = False,
) -> MultiObjectiveOptimizationConfig:
    num_constraint_outcomes = with_relative_constraint + with_absolute_constraint
    _validate_num_outcomes(num_outcomes=num_objectives + num_constraint_outcomes)
    # minimum Branin value is 0.397887
    if has_objective_thresholds:
        objective_thresholds = [
            ObjectiveThreshold(
                metric=get_branin_metric(name="branin_a"),
                bound=10.0,
                op=ComparisonOp.LEQ,
                relative=False,
            ),
            ObjectiveThreshold(
                metric=get_branin_metric(name="branin_b"),
                bound=20.0,
                op=ComparisonOp.LEQ,
                relative=False,
            ),
        ]
        if num_objectives == 3:
            objective_thresholds.append(
                ObjectiveThreshold(
                    metric=get_branin_metric(name="branin_c"),
                    bound=5.0,
                    op=ComparisonOp.LEQ,
                    relative=False,
                )
            )
    else:
        objective_thresholds = None
    outcome_constraints = []
    if with_relative_constraint:
        outcome_constraints.append(
            get_outcome_constraint(
                metric=get_branin_metric(name="branin_d"), relative=True
            )
        )

    if with_absolute_constraint:
        outcome_constraints.append(
            get_outcome_constraint(
                metric=get_branin_metric(name="branin_e"), relative=False
            )
        )
    return MultiObjectiveOptimizationConfig(
        objective=get_branin_multi_objective(num_objectives=num_objectives),
        objective_thresholds=objective_thresholds,
        outcome_constraints=outcome_constraints,
    )


##############################
# Arms
##############################


def get_arm() -> Arm:
    # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
    # parameter `parameters` to call `ax.core.arm.Arm.__init__` but got
    # `Dict[str, typing.Union[float, str]]`.
    return Arm(parameters={"w": 0.75, "x": 1, "y": "foo", "z": True, "d": 2.5})


def get_status_quo() -> Arm:
    return Arm(
        # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
        # parameter `parameters` to call `ax.core.arm.Arm.__init__`
        # but got `Dict[str, typing.Union[float, str]]`.
        parameters={"w": 0.2, "x": 1, "y": "bar", "z": False, "d": 1.4},
        name="status_quo",
    )


def get_arm_weights1() -> MutableMapping[Arm, float]:
    parameters_dicts: list[TParameterization] = [
        {"w": 0.85, "x": 1, "y": "baz", "z": False, "d": 2.7},
        {"w": 0.75, "x": 1, "y": "foo", "z": True, "d": 2.5},
        {"w": 1.4, "x": 2, "y": "bar", "z": True, "d": 3.8},
    ]
    arms = [Arm(param_dict) for param_dict in parameters_dicts]
    weights = [0.25, 0.3, 0.25, 0.2]
    return OrderedDict(zip(arms, weights))


def get_arm_weights2() -> MutableMapping[Arm, float]:  # update
    parameters_dicts: list[TParameterization] = [
        {"w": 0.96, "x": 3, "y": "hello", "z": True, "d": 2.92},
        {"w": 0.16, "x": 4, "y": "dear", "z": True, "d": 1.32},
        {"w": 3.1, "x": 5, "y": "world", "z": False, "d": 7.2},
    ]
    arms = [Arm(param_dict) for param_dict in parameters_dicts]
    weights = [0.25, 0.3, 0.25, 0.2]
    return OrderedDict(zip(arms, weights))


def get_arms_from_dict(arm_weights_dict: MutableMapping[Arm, float]) -> list[Arm]:
    return list(arm_weights_dict.keys())


def get_weights_from_dict(arm_weights_dict: MutableMapping[Arm, float]) -> list[float]:
    return list(arm_weights_dict.values())


def get_arms() -> list[Arm]:
    return list(get_arm_weights1().keys())


def get_weights() -> list[float]:
    return list(get_arm_weights1().values())


def get_branin_arms(n: int, seed: int) -> list[Arm]:
    np.random.seed(seed)
    x1_raw = np.random.rand(n)
    x2_raw = np.random.rand(n)
    return [
        Arm(parameters={"x1": -5 + x1_raw[i] * 15, "x2": x2_raw[i] * 15})
        for i in range(n)
    ]


def get_abandoned_arm() -> AbandonedArm:
    return AbandonedArm(name="0_0", reason="foobar", time=datetime.now())


##############################
# Generator Runs
##############################


def get_generator_run() -> GeneratorRun:
    arms = get_arms_from_dict(get_arm_weights1())
    weights = get_weights_from_dict(get_arm_weights1())
    optimization_config = get_optimization_config()
    search_space = get_search_space()
    arm_predictions = get_model_predictions_per_arm()
    return GeneratorRun(
        arms=arms,
        weights=weights,
        optimization_config=optimization_config,
        search_space=search_space,
        model_predictions=get_model_predictions(),
        best_arm_predictions=(arms[0], arm_predictions[arms[0].signature]),
        fit_time=10.0,
        gen_time=5.0,
        generator_key="Sobol",
        generator_kwargs={"scramble": False, "torch_device": torch.device("cpu")},
        adapter_kwargs={"transforms": Cont_X_trans},
        candidate_metadata_by_arm_signature={
            a.signature: {"md_key": f"md_val_{a.signature}"} for a in arms
        },
    )


def get_generator_run2() -> GeneratorRun:
    arms = get_arms_from_dict(get_arm_weights1())
    weights = get_weights_from_dict(get_arm_weights1())
    return GeneratorRun(arms=arms, weights=weights)


##############################
# Runners
##############################


def get_synthetic_runner() -> SyntheticRunner:
    return SyntheticRunner(dummy_metadata="foobar")


##############################
# Data
##############################


def get_data(
    metric_name: str = "ax_test_metric",
    trial_index: int = 0,
    num_non_sq_arms: int = 4,
    include_sq: bool = True,
) -> Data:
    arm_names = ["status_quo"] if include_sq else []
    arm_names += [f"{trial_index}_{i}" for i in range(num_non_sq_arms)]
    num_arms = num_non_sq_arms + 1 if include_sq else num_non_sq_arms
    df_dict = {
        "trial_index": trial_index,
        "metric_name": metric_name,
        "arm_name": arm_names,
        "mean": ([1, 3, 2, 2.25, 1.75] * ((num_arms + 4) // 5))[:num_arms],
        "sem": ([0, 0.5, 0.25, 0.40, 0.15] * ((num_arms + 4) // 5))[:num_arms],
        "n": ([100, 100, 100, 100, 100] * ((num_arms + 4) // 5))[:num_arms],
        "metric_signature": metric_name,
    }
    return Data(df=pd.DataFrame.from_records(df_dict))


def get_non_monolithic_branin_moo_data() -> Data:
    now = datetime.now()
    return Data(
        df=pd.DataFrame.from_records(
            [
                {
                    "arm_name": "status_quo",
                    "trial_index": "0",
                    "metric_name": "branin_a",  # Obj. metric for experiment.
                    "mean": 2.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=3),
                    "end_time": now,
                    "metric_signature": "branin_a",
                },
                {
                    "arm_name": "0_0",
                    "trial_index": "0",
                    "metric_name": "branin_a",  # Obj. metric for experiment.
                    "mean": 1.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=3),
                    "end_time": now,
                    "metric_signature": "branin_a",
                },
                {
                    "arm_name": "status_quo",
                    "trial_index": "0",
                    "metric_name": "branin_b",  # Obj. metric for experiment.
                    "mean": 2.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=2),
                    "end_time": now - timedelta(days=1),
                    "metric_signature": "branin_b",
                },
                {
                    "arm_name": "0_0",
                    "trial_index": "0",
                    "metric_name": "branin_b",  # Obj. metric for experiment.
                    "mean": 1.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=2),
                    "end_time": now - timedelta(days=1),
                    "metric_signature": "branin_b",
                },
            ]
        )
    )


def get_map_data(trial_index: int = 0) -> MapData:
    evaluations = {
        "status_quo": [
            (1, {"ax_test_metric": (1.0, 0.5)}),
            (2, {"ax_test_metric": (2.0, 0.5)}),
            (3, {"ax_test_metric": (3.0, 0.5)}),
            (4, {"ax_test_metric": (4.0, 0.5)}),
        ],
        "0_0": [
            (1, {"ax_test_metric": (3.7, 0.5)}),
            (2, {"ax_test_metric": (3.8, 0.5)}),
            (3, {"ax_test_metric": (3.9, 0.5)}),
            (4, {"ax_test_metric": (4.0, 0.5)}),
        ],
        "0_1": [
            (1, {"ax_test_metric": (3.0, 0.5)}),
            (2, {"ax_test_metric": (5.0, 0.5)}),
            (3, {"ax_test_metric": (6.0, 0.5)}),
            (4, {"ax_test_metric": (1.0, 0.5)}),
        ],
    }
    return assert_is_instance(
        raw_evaluations_to_data(
            raw_data=evaluations,
            trial_index=trial_index,
            metric_name_to_signature={
                "ax_test_metric": "ax_test_metric",
                "epoch": "epoch",
            },
        ),
        MapData,
    )


def get_branin_data(
    trial_indices: Iterable[int] | None = None,
    trials: Iterable[Trial] | None = None,
    metrics: Iterable[str] | None = None,
) -> Data:
    if trial_indices and trials:
        raise ValueError("Expected `trial_indices` or `trials`, not both.")
    if metrics is None:
        metrics = ["branin"]
    if trials:
        df_dicts = [
            {
                "trial_index": trial.index,
                "metric_name": metric,
                "arm_name": none_throws(assert_is_instance(trial, Trial).arm).name,
                "mean": branin(
                    float(none_throws(none_throws(trial.arm).parameters["x1"])),
                    float(none_throws(none_throws(trial.arm).parameters["x2"])),
                ),
                "sem": 0.0,
                "metric_signature": metric,
            }
            for trial in trials
            for metric in metrics
        ]
    else:
        df_dicts = [
            {
                "trial_index": trial_index,
                "metric_name": metric,
                "arm_name": f"{trial_index}_0",
                "mean": 5.0,
                "sem": 0.0,
                "metric_signature": metric,
            }
            for trial_index in (trial_indices or [0])
            for metric in metrics
        ]
    return Data(df=pd.DataFrame.from_records(df_dicts))


def get_branin_data_batch(
    batch: BatchTrial,
    fill_vals: dict[str, float] | None = None,
    metrics: list[str] | None = None,
) -> Data:
    means = []
    fill_vals = fill_vals or {}
    metrics = metrics or ["branin"]
    for arm in batch.arms:
        params = arm.parameters
        for k, v in fill_vals.items():
            if params.get(k, None) is None:
                params[k] = v
        if params["x1"] is None or params["x2"] is None:
            means.append(5.0)
        else:
            means.append(
                branin(
                    float(none_throws(params["x1"])),
                    float(none_throws(params["x2"])),
                )
            )
    records = [
        {
            "trial_index": batch.index,
            "arm_name": batch.arms[i].name,
            "metric_name": metric,
            "mean": means[i],
            "sem": 0.1,
            "metric_signature": metric,
        }
        for i in range(len(means))
        for metric in metrics
    ]
    return Data(pd.DataFrame.from_records(records))


def get_branin_data_multi_objective(
    trial_indices: Iterable[int] | None = None,
    arm_names: Iterable[str] | None = None,
    outcomes: Sequence[str] | None = None,
) -> Data:
    if outcomes is None:
        outcomes = ["branin_a", "branin_b"]
    else:
        _validate_num_outcomes(num_outcomes=len(outcomes))

    df_dicts = [
        {
            "trial_index": trial_index,
            "metric_name": outcome,
            "arm_name": arm_name,
            "mean": 5.0,
            "sem": 0.0,
            "metric_signature": outcome,
        }
        for trial_index in (trial_indices or [0])
        for arm_name in arm_names or [f"{trial_index}_0"]
        for outcome in outcomes
    ]
    return Data(df=pd.DataFrame.from_records(df_dicts))


def get_percentile_early_stopping_strategy() -> PercentileEarlyStoppingStrategy:
    return PercentileEarlyStoppingStrategy(
        percentile_threshold=0.25,
        min_progression=0.2,
        min_curves=10,
        normalize_progressions=True,
    )


def get_percentile_early_stopping_strategy_with_non_objective_metric_signature() -> (
    PercentileEarlyStoppingStrategy
):
    return PercentileEarlyStoppingStrategy(
        metric_signatures=["foo"],
        percentile_threshold=0.25,
        min_progression=0.2,
        min_curves=10,
        normalize_progressions=True,
    )


def get_threshold_early_stopping_strategy() -> ThresholdEarlyStoppingStrategy:
    return ThresholdEarlyStoppingStrategy(
        metric_threshold=0.1,
        min_progression=0.2,
        normalize_progressions=True,
    )


def get_and_early_stopping_strategy() -> AndEarlyStoppingStrategy:
    return AndEarlyStoppingStrategy(
        left=get_percentile_early_stopping_strategy(),
        right=get_threshold_early_stopping_strategy(),
    )


def get_or_early_stopping_strategy() -> OrEarlyStoppingStrategy:
    return OrEarlyStoppingStrategy(
        left=get_percentile_early_stopping_strategy(),
        right=get_threshold_early_stopping_strategy(),
    )


class DummyEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    def __init__(self, early_stop_trials: dict[int, str | None] | None = None) -> None:
        super().__init__()
        self.early_stop_trials: dict[int, str | None] = early_stop_trials or {}

    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        return False

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        return self.early_stop_trials


def get_improvement_global_stopping_strategy() -> ImprovementGlobalStoppingStrategy:
    return ImprovementGlobalStoppingStrategy(
        min_trials=30,
        window_size=10,
        improvement_bar=0.05,
        inactive_when_pending_trials=True,
    )


class DummyGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """
    A dummy Global Stopping Strategy which stops the optimization after
    a pre-specified number of trials are completed.
    """

    def __init__(self, min_trials: int, trial_to_stop: int) -> None:
        super().__init__(min_trials=min_trials)
        self.trial_to_stop = trial_to_stop

    def _should_stop_optimization(
        self, experiment: Experiment, **kwargs: dict[str, Any]
    ) -> tuple[bool, str]:
        num_completed_trials = len(experiment.trials_by_status[TrialStatus.COMPLETED])

        if num_completed_trials >= max([self.min_trials, self.trial_to_stop]):
            return True, "Stop the optimization."
        else:
            return False, ""


##############################
# Instances of types from core/types.py
##############################


def get_model_mean() -> TModelMean:
    mean: TModelMean = {"test_metric_1": [1, 2, 3], "test_metric_2": [3, 4, 5]}
    return mean


def get_model_covariance() -> TModelCov:
    covariance: TModelCov = {
        "test_metric_1": {"test_metric_1": [5, 6, 7], "test_metric_2": [7, 8, 9]},
        "test_metric_2": {"test_metric_1": [9, 10, 11], "test_metric_2": [11, 12, 13]},
    }
    return covariance


def get_model_predictions() -> TModelPredict:
    model_predictions: TModelPredict = (get_model_mean(), get_model_covariance())
    return model_predictions


def get_model_predictions_per_arm() -> dict[str, TModelPredictArm]:
    arms = list(get_arm_weights1().keys())
    means = get_model_mean()
    covariances = get_model_covariance()
    metric_names = list(means.keys())
    m_1, m_2 = metric_names[0], metric_names[1]
    return {
        arms[i].signature: (
            {m_1: means[m_1][i], m_2: means[m_2][i]},
            {
                m_1: {m_1: covariances[m_1][m_1][i], m_2: covariances[m_1][m_2][i]},
                m_2: {m_1: covariances[m_2][m_1][i], m_2: covariances[m_2][m_2][i]},
            },
        )
        for i in range(len(arms))
    }


##############################
# Modular BoTorch Model Components
##############################


def get_botorch_model() -> BoTorchGenerator:
    return BoTorchGenerator(
        surrogate=get_surrogate(), acquisition_class=get_acquisition_type()
    )


def get_botorch_model_with_default_acquisition_class() -> BoTorchGenerator:
    return BoTorchGenerator(
        surrogate=get_surrogate(),
        acquisition_class=Acquisition,
        botorch_acqf_class=get_acquisition_function_type(),
    )


def get_botorch_model_with_surrogate_spec(
    with_covar_module: bool = True,
) -> BoTorchGenerator:
    if with_covar_module:
        config = ModelConfig(
            model_options={"some_option": "some_value"},
            covar_module_class=DefaultRBFKernel,
            covar_module_options={"inactive_features": []},
        )
    else:
        config = ModelConfig(
            model_options={"some_option": "some_value"},
        )
    return BoTorchGenerator(surrogate_spec=SurrogateSpec(model_configs=[config]))


def get_surrogate() -> Surrogate:
    return Surrogate(
        surrogate_spec=SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=get_model_type(),
                    mll_class=get_mll_type(),
                )
            ]
        )
    )


def get_surrogate_spec_with_default() -> SurrogateSpec:
    return SurrogateSpec(
        model_configs=[
            ModelConfig(
                botorch_model_class=SingleTaskGP,
                covar_module_class=ScaleMaternKernel,
                covar_module_options={
                    "ard_num_dims": DEFAULT,
                    "lengthscale_prior": GammaPrior(6.0, 3.0),
                    "outputscale_prior": GammaPrior(2.0, 0.15),
                    "batch_shape": DEFAULT,
                },
            )
        ]
    )


def get_surrogate_spec_with_lognormal() -> SurrogateSpec:
    return SurrogateSpec(
        model_configs=[
            ModelConfig(
                botorch_model_class=SingleTaskGP,
                covar_module_class=RBFKernel,
                covar_module_options={
                    "ard_num_dims": DEFAULT,
                    "lengthscale_prior": LogNormalPrior(-4.0, 1.0),
                    "batch_shape": DEFAULT,
                },
            )
        ]
    )


def get_surrogate_spec_with_inputs(
    model_class: type[Model] | None = None,
    covar_module_class: type[Kernel] | None = None,
) -> SurrogateSpec:
    return SurrogateSpec(
        model_configs=[
            ModelConfig(
                botorch_model_class=model_class, covar_module_class=covar_module_class
            )
        ]
    )


def get_acquisition_type() -> type[Acquisition]:
    return Acquisition


def get_model_type() -> type[Model]:
    return SingleTaskGP


def get_mll_type() -> type[MarginalLogLikelihood]:
    return ExactMarginalLogLikelihood


def get_acquisition_function_type() -> type[AcquisitionFunction]:
    return qExpectedImprovement


def get_winsorization_config() -> WinsorizationConfig:
    return WinsorizationConfig(
        lower_quantile_margin=0.2,
        upper_quantile_margin=0.3,
        lower_boundary=20,
        upper_boundary=50,
    )


def get_gamma_prior() -> GammaPrior:
    return GammaPrior(concentration=0.9, rate=10.0)


def get_interval() -> Interval:
    return Interval(lower_bound=1e-6, upper_bound=0.1)


def get_chained_input_transform() -> ChainedInputTransform:
    bounds = torch.tensor([[0, 0], [3, 5]], dtype=torch.double)
    return ChainedInputTransform(
        round=Round(
            integer_indices=[1],
            transform_on_eval=True,
            transform_on_train=False,
        ),
        normalize=Normalize(d=2, bounds=bounds),
    )


##############################
# Orchestrator
##############################


def get_default_orchestrator_options() -> OrchestratorOptions:
    return OrchestratorOptions()


def get_orchestrator_options_batch_trial() -> OrchestratorOptions:
    return OrchestratorOptions(trial_type=TrialType.BATCH_TRIAL)


##############################
# Other
##############################


def get_pathlib_path() -> Path:
    return Path("some/meaningless/path")


def get_dataset(
    num_samples: int = 2,
    d: int = 2,
    m: int = 2,
    has_observation_noise: bool = False,
    feature_names: list[str] | None = None,
    outcome_names: list[str] | None = None,
    tkwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> SupervisedDataset:
    """Constructs a SupervisedDataset based on the given arguments.

    Args:
        num_samples: The number of samples in the dataset.
        d: The dimension of the features.
        m: The number of outcomes.
        has_observation_noise: If True, includes Yvar in the dataset.
        feature_names: A list of feature names. Defaults to x0, x1...
        outcome_names: A list of outcome names. Defaults to y0, y1...
        tkwargs: Optional dictionary of tensor kwargs, such as dtype and device.
        seed: An optional seed used to generate the data.
    """
    if seed is not None:
        set_rng_seed(seed)
    feature_names = feature_names or [f"x{i}" for i in range(d)]
    outcome_names = outcome_names or [f"y{i}" for i in range(m)]
    tkwargs = tkwargs or {}
    return SupervisedDataset(
        X=torch.rand(num_samples, d, **tkwargs),
        Y=torch.rand(num_samples, m, **tkwargs),
        Yvar=(
            torch.rand(num_samples, m, **tkwargs) * 0.01
            if has_observation_noise
            else None
        ),
        feature_names=feature_names,
        outcome_names=outcome_names,
    )


def get_online_sobol_mbm_generation_strategy() -> GenerationStrategy:
    """Constructs a GenerationStrategy with Sobol and MBM nodes for simulating
    online optimization.
    """
    # Set up the node-based generation strategy for testing.
    # TODO: @mgarrard make this more realistic of an actual online gs
    step_generator_kwargs = {"silently_filter_kwargs": True}
    sobol_criterion = [
        MinTrials(
            threshold=1,
            transition_to="MBM_node",
            block_gen_if_met=True,
            only_in_statuses=None,
            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
        ),
        MinTrials(
            threshold=1,
            transition_to="MBM_node",
            block_gen_if_met=True,
            only_in_statuses=[
                TrialStatus.RUNNING,
                TrialStatus.COMPLETED,
                TrialStatus.EARLY_STOPPED,
            ],
        ),
    ]
    sobol_generator_spec = GeneratorSpec(
        generator_enum=Generators.SOBOL,
        generator_kwargs=step_generator_kwargs,
        generator_gen_kwargs={},
    )
    mbm_generator_spec = GeneratorSpec(
        generator_enum=Generators.BOTORCH_MODULAR,
        generator_kwargs=step_generator_kwargs,
        generator_gen_kwargs={},
    )
    sobol_node = GenerationNode(
        name="sobol_node",
        transition_criteria=sobol_criterion,
        generator_specs=[sobol_generator_spec],
        input_constructors={InputConstructorPurpose.N: NodeInputConstructors.ALL_N},
    )
    mbm_node = GenerationNode(
        name="MBM_node",
        transition_criteria=[],
        generator_specs=[mbm_generator_spec],
        input_constructors={InputConstructorPurpose.N: NodeInputConstructors.ALL_N},
    )
    return GenerationStrategy(
        name="Sobol+MBM_Nodes",
        nodes=[sobol_node, mbm_node],
    )


##############################
# Custom runner and metric
##############################


class CustomTestRunner(Runner):
    def __init__(self, test_attribute: str) -> None:
        self.test_attribute = test_attribute

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        return {"foo": "bar"}


class CustomTestMetric(Metric):
    def __init__(
        self,
        name: str,
        test_attribute: str,
        lower_is_better: bool | None = None,
    ) -> None:
        self.test_attribute = test_attribute
        super().__init__(name=name, lower_is_better=lower_is_better)
