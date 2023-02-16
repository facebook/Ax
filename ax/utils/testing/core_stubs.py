#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from datetime import datetime, timedelta

from logging import Logger
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
)

import numpy as np
import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import DataType, Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData, MapKeyInfo
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
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.parameter_distribution import ParameterDistribution
from ax.core.risk_measures import RiskMeasure
from ax.core.runner import Runner
from ax.core.search_space import HierarchicalSearchSpace, RobustSearchSpace, SearchSpace
from ax.core.trial import Trial
from ax.core.types import (
    ComparisonOp,
    TModelCov,
    TModelMean,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
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
from ax.exceptions.core import UserInputError
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.metrics.branin import AugmentedBraninMetric, BraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import AugmentedHartmann6Metric, Hartmann6Metric
from ax.modelbridge.factory import Cont_X_trans, get_factorial, get_sobol
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.winsorization_config import WinsorizationConfig
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.measurement.synthetic_functions import branin
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from gpytorch.constraints import Interval
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

logger: Logger = get_logger(__name__)

TEST_SOBOL_SEED = 1234

##############################
# Experiments
##############################


def get_experiment(with_status_quo: bool = True) -> Experiment:
    return Experiment(
        name="test",
        search_space=get_search_space(),
        optimization_config=get_optimization_config(),
        status_quo=get_status_quo() if with_status_quo else None,
        description="test description",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )


def get_experiment_with_map_data_type() -> Experiment:
    return Experiment(
        name="test_map_data",
        search_space=get_search_space(),
        optimization_config=get_map_optimization_config(),
        status_quo=get_status_quo(),
        description="test description",
        tracking_metrics=[MapMetric(name="tracking")],
        is_test=True,
        default_data_type=DataType.MAP_DATA,
    )


def get_experiment_with_custom_runner_and_metric() -> Experiment:

    # Create experiment with custom runner and metric
    experiment = Experiment(
        name="test",
        search_space=get_search_space(),
        optimization_config=get_optimization_config(),
        description="test description",
        tracking_metrics=[
            CustomTestMetric(name="custom_test_metric", test_attribute="test")
        ],
        runner=CustomTestRunner(test_attribute="test"),
        is_test=True,
    )

    # Create a trial, set its runner and complete it.
    sobol_generator = get_sobol(search_space=experiment.search_space)
    sobol_run = sobol_generator.gen(n=1)
    trial = experiment.new_trial(generator_run=sobol_run)
    trial.runner = experiment.runner
    trial.mark_running()
    experiment.attach_data(get_data(metric_name="custom_test_metric"))
    trial.mark_completed()

    return experiment


def get_branin_experiment(
    has_optimization_config: bool = True,
    with_batch: bool = False,
    with_trial: bool = False,
    with_status_quo: bool = False,
    with_fidelity_parameter: bool = False,
    with_choice_parameter: bool = False,
    with_str_choice_param: bool = False,
    search_space: Optional[SearchSpace] = None,
    minimize: bool = False,
    named: bool = True,
    with_completed_trial: bool = False,
) -> Experiment:
    search_space = search_space or get_branin_search_space(
        with_fidelity_parameter=with_fidelity_parameter,
        with_choice_parameter=with_choice_parameter,
        with_str_choice_param=with_str_choice_param,
    )
    exp = Experiment(
        name="branin_test_experiment" if named else None,
        search_space=search_space,
        optimization_config=get_branin_optimization_config(minimize=minimize)
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
        is_test=True,
    )

    if with_status_quo:
        exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})

    if with_batch:
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=15)
        exp.new_batch_trial(optimize_for_power=with_status_quo).add_generator_run(
            sobol_run
        )

    if with_trial or with_completed_trial:
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=1)
        trial = exp.new_trial(generator_run=sobol_run)

        if with_completed_trial:
            trial.mark_running(no_runner_required=True)
            exp.attach_data(get_branin_data(trials=[trial]))  # Add data for one trial
            trial.mark_completed()

    return exp


def get_robust_branin_experiment(
    risk_measure: Optional[RiskMeasure] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    num_sobol_trials: int = 2,
) -> Experiment:
    x1_dist = ParameterDistribution(
        parameters=["x1"], distribution_class="norm", distribution_parameters={}
    )
    search_space = RobustSearchSpace(
        parameters=[
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
            ),
        ],
        parameter_distributions=[x1_dist],
        num_samples=16,
    )

    risk_measure = risk_measure or RiskMeasure(
        risk_measure="CVaR",
        options={"n_w": 16, "alpha": 0.8},
    )

    optimization_config = optimization_config or OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="branin_metric", param_names=["x1", "x2"], lower_is_better=True
            ),
            minimize=True,
        ),
        risk_measure=risk_measure,
    )

    exp = Experiment(
        name="branin_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    sobol = get_sobol(search_space=exp.search_space)
    for _ in range(num_sobol_trials):
        exp.new_trial(generator_run=sobol.gen(1)).run().mark_completed()
    return exp


def get_branin_experiment_with_timestamp_map_metric(
    with_status_quo: bool = False,
    rate: Optional[float] = None,
) -> Experiment:
    exp = Experiment(
        name="branin_with_timestamp_map_metric",
        search_space=get_branin_search_space(),
        optimization_config=OptimizationConfig(
            objective=Objective(
                metric=BraninTimestampMapMetric(
                    name="branin_map",
                    param_names=["x1", "x2"],
                    rate=rate,
                    lower_is_better=True,
                ),
                minimize=True,
            )
        ),
        tracking_metrics=[BraninMetric(name="branin", param_names=["x1", "x2"])],
        runner=SyntheticRunner(),
        default_data_type=DataType.MAP_DATA,
    )

    if with_status_quo:
        exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})

    return exp


def get_test_map_data_experiment(
    num_trials: int, num_fetches: int, num_complete: int
) -> Experiment:
    experiment = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
    for i in range(num_trials):
        trial = experiment.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
        trial.run()
    for _ in range(num_fetches):
        # each time we call fetch, we grab another timestamp
        experiment.fetch_data()
    for i in range(num_complete):
        experiment.trials[i].mark_as(status=TrialStatus.COMPLETED)
    experiment.attach_data(data=experiment.fetch_data())
    return experiment


def get_multi_type_experiment(
    add_trial_type: bool = True, add_trials: bool = False, num_arms: int = 10
) -> MultiTypeExperiment:
    oc = OptimizationConfig(Objective(BraninMetric("m1", ["x1", "x2"])))
    experiment = MultiTypeExperiment(
        name="test_exp",
        search_space=get_branin_search_space(),
        default_trial_type="type1",
        default_runner=SyntheticRunner(dummy_metadata="dummy1"),
        optimization_config=oc,
    )
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
        t1 = experiment.new_batch_trial(generator_run=gr, trial_type="type1")
        t2 = experiment.new_batch_trial(generator_run=gr, trial_type="type2")
        t1.set_status_quo_with_weight(status_quo=t1.arms[0], weight=0.5)
        t2.set_status_quo_with_weight(status_quo=t2.arms[0], weight=0.5)
        t1.run()
        t2.run()

    return experiment


def get_multi_type_experiment_with_multi_objective(
    add_trials: bool = False,
) -> MultiTypeExperiment:
    oc = get_branin_multi_objective_optimization_config()
    experiment = MultiTypeExperiment(
        name="test_exp",
        search_space=get_branin_search_space(),
        default_trial_type="type1",
        default_runner=SyntheticRunner(dummy_metadata="dummy1"),
        optimization_config=oc,
    )
    experiment.add_trial_type(
        trial_type="type2", runner=SyntheticRunner(dummy_metadata="dummy2")
    )

    if add_trials:
        generator = get_sobol(experiment.search_space)
        gr = generator.gen(10)
        t1 = experiment.new_batch_trial(generator_run=gr, trial_type="type1")
        t2 = experiment.new_batch_trial(generator_run=gr, trial_type="type2")
        t1.set_status_quo_with_weight(status_quo=t1.arms[0], weight=0.5)
        t2.set_status_quo_with_weight(status_quo=t2.arms[0], weight=0.5)
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
        optimization_config=OptimizationConfig(
            objective=Objective(metric=get_factorial_metric())
        )
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
        is_test=True,
        tracking_metrics=[get_factorial_metric("secondary_metric")],
    )

    if with_status_quo:
        exp.status_quo = Arm(
            parameters={
                "factor1": "level11",
                "factor2": "level21",
                "factor3": "level31",
            }
        )

    if with_batch:
        factorial_generator = get_factorial(search_space=exp.search_space)
        factorial_run = factorial_generator.gen(n=-1)
        exp.new_batch_trial(optimize_for_power=with_status_quo).add_generator_run(
            factorial_run
        )

    return exp


def get_experiment_with_repeated_arms(num_repeated_arms: int) -> Experiment:
    batch_trial = get_batch_trial_with_repeated_arms(num_repeated_arms)
    return batch_trial.experiment


def get_experiment_with_trial() -> Experiment:
    trial = get_trial()
    return trial.experiment


def get_experiment_with_batch_trial() -> Experiment:
    batch_trial = get_batch_trial()
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

    return exp


def get_branin_experiment_with_multi_objective(
    has_optimization_config: bool = True,
    has_objective_thresholds: bool = False,
    with_batch: bool = False,
    with_status_quo: bool = False,
    with_fidelity_parameter: bool = False,
    num_objectives: int = 2,
) -> Experiment:
    exp = Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(
            with_fidelity_parameter=with_fidelity_parameter
        ),
        optimization_config=get_branin_multi_objective_optimization_config(
            has_objective_thresholds=has_objective_thresholds,
            num_objectives=num_objectives,
        )
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
        is_test=True,
    )

    if with_status_quo:
        # Experiment chooses the name "status_quo" by default
        exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})

    if with_batch:
        sobol_generator = get_sobol(search_space=exp.search_space, seed=TEST_SOBOL_SEED)
        sobol_run = sobol_generator.gen(n=5)
        exp.new_batch_trial(optimize_for_power=with_status_quo).add_generator_run(
            sobol_run
        )

    return exp


def get_branin_with_multi_task(with_multi_objective: bool = False) -> Experiment:
    exp = Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(),
        optimization_config=get_branin_multi_objective_optimization_config(
            has_objective_thresholds=True,
        )
        if with_multi_objective
        else get_branin_optimization_config(),
        runner=SyntheticRunner(),
        is_test=True,
    )

    exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0}, name="status_quo")

    sobol_generator = get_sobol(search_space=exp.search_space, seed=TEST_SOBOL_SEED)
    sobol_run = sobol_generator.gen(n=5)
    exp.new_batch_trial(optimize_for_power=True).add_generator_run(sobol_run)
    not_none(exp.trials.get(0)).run()
    exp.new_batch_trial(optimize_for_power=True).add_generator_run(sobol_run)
    not_none(exp.trials.get(1)).run()

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
    return Experiment(
        name="test_experiment_scalarized_objective and outcome constraint",
        search_space=get_search_space(),
        optimization_config=optimization_config,
        status_quo=get_status_quo(),
        description="test experiment with scalarized objective and outcome constraint",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )


def get_hierarchical_search_space_experiment() -> Experiment:
    return Experiment(
        name="test_experiment_hss",
        description="test experiment with hierarchical search space",
        search_space=get_hierarchical_search_space(),
        optimization_config=get_optimization_config(),
    )


def get_experiment_with_observations(
    observations: List[List[float]],
    minimize: bool = False,
    scalarized: bool = False,
    constrained: bool = False,
) -> Experiment:
    multi_objective = (len(observations[0]) - constrained) > 1
    if multi_objective:
        metrics = [
            Metric(name="m1", lower_is_better=minimize),
            Metric(name="m2", lower_is_better=False),
        ]
        if scalarized:
            optimization_config = OptimizationConfig(
                objective=ScalarizedObjective(metrics)
            )
            if constrained:  # pragma: no cover
                raise NotImplementedError
        else:
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(metrics=metrics),
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=metrics[i],
                        bound=0.0,
                        relative=False,
                        op=ComparisonOp.LEQ
                        if metrics[i].lower_is_better
                        else ComparisonOp.GEQ,
                    )
                    for i in [0, 1]
                ],
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric(name="m3"),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=False,
                    )
                ]
                if constrained
                else None,
            )
    else:
        if scalarized:  # pragma: no cover
            raise NotImplementedError
        objective = Objective(metric=Metric(name="m1"), minimize=minimize)
        if constrained:
            constraint = OutcomeConstraint(
                metric=Metric(name="m2"),
                op=ComparisonOp.GEQ,
                bound=0.0,
                relative=False,
            )
            optimization_config = OptimizationConfig(
                objective=objective, outcome_constraints=[constraint]
            )
        else:
            optimization_config = OptimizationConfig(objective=objective)
    exp = Experiment(
        search_space=get_search_space_for_range_values(min=0.0, max=1.0),
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
        is_test=True,
    )
    for i, obs in enumerate(observations):
        # Create a dummy trial to add the observation.
        trial = exp.new_trial()
        trial.add_arm(Arm({"x": i / 100.0, "y": i / 100.0}))
        data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": f"{i}_0",
                        "metric_name": f"m{j + 1}",
                        "mean": o,
                        "sem": None,
                        "trial_index": i,
                    }
                    for j, o in enumerate(obs)
                ]
            )
        )
        exp.attach_data(data)
        trial.run().complete()
    return exp


def get_high_dimensional_branin_experiment() -> Experiment:
    search_space = SearchSpace(
        # pyre-fixme[6]: In call `SearchSpace.__init__`, for 1st parameter `parameters`
        # expected `List[Parameter]` but got `List[RangeParameter]`.
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

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="objective",
                param_names=["x19", "x44"],
            ),
            minimize=True,
        )
    )

    return Experiment(
        name="high_dimensional_branin_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )


##############################
# Search Spaces
##############################


def get_search_space() -> SearchSpace:
    parameters: List[Parameter] = [
        get_range_parameter(),
        get_range_parameter2(),
        get_choice_parameter(),
        get_fixed_parameter(),
    ]
    return SearchSpace(
        parameters=parameters,
        parameter_constraints=[
            get_order_constraint(),
            get_parameter_constraint(),
            get_sum_constraint1(),
        ],
    )


def get_branin_search_space(
    with_fidelity_parameter: bool = False,
    with_choice_parameter: bool = False,
    with_str_choice_param: bool = False,
) -> SearchSpace:
    parameters = [
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            values=[float(x) for x in range(0, 16)],
        )
        if with_choice_parameter
        else RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
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

    return SearchSpace(parameters=cast(List[Parameter], parameters))


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
        parameters=[  # pyre-ignore[6]
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


def get_hartmann_search_space(with_fidelity_parameter: bool = False) -> SearchSpace:
    parameters = [
        RangeParameter(
            name=f"x{idx+1}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for idx in range(6)
    ]
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

    return SearchSpace(parameters=cast(List[Parameter], parameters))


def get_search_space_for_value(val: float = 3.0) -> SearchSpace:
    return SearchSpace([FixedParameter("x", ParameterType.FLOAT, val)])


def get_search_space_for_range_value(min: float = 3.0, max: float = 6.0) -> SearchSpace:
    return SearchSpace([RangeParameter("x", ParameterType.FLOAT, min, max)])


def get_search_space_for_range_values(
    min: float = 3.0, max: float = 6.0
) -> SearchSpace:
    return SearchSpace(
        [
            RangeParameter("x", ParameterType.FLOAT, min, max),
            RangeParameter("y", ParameterType.FLOAT, min, max),
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


def get_hierarchical_search_space(
    with_fixed_parameter: bool = False,
) -> HierarchicalSearchSpace:
    parameters: List[Parameter] = [
        get_model_parameter(with_fixed_parameter=with_fixed_parameter),
        get_lr_parameter(),
        get_l2_reg_weight_parameter(),
        get_num_boost_rounds_parameter(),
    ]
    if with_fixed_parameter:
        parameters.append(get_fixed_parameter())
    return HierarchicalSearchSpace(parameters=parameters)


def get_robust_search_space(
    lb: float = 0.0,
    ub: float = 5.0,
    multivariate: bool = False,
    use_discrete: bool = False,
    num_samples: int = 4,  # dummy
) -> RobustSearchSpace:

    parameters = [
        RangeParameter("x", ParameterType.FLOAT, lb, ub),
        RangeParameter("y", ParameterType.FLOAT, lb, ub),
        RangeParameter("z", ParameterType.INT, lb, ub),
        ChoiceParameter("c", ParameterType.STRING, ["red", "panda"]),
    ]
    if multivariate:
        if use_discrete:  # pragma: no cover
            raise UserInputError(
                "`multivariate` and `use_discrete` are not supported together."
            )
        distributions = [
            ParameterDistribution(
                parameters=["x", "y"],
                distribution_class="multivariate_normal",
                distribution_parameters={
                    "mean": [0.1, 0.2],
                    "cov": [[0.5, 0.1], [0.1, 0.3]],
                },
            )
        ]
    else:
        distributions = []
        distributions.append(
            ParameterDistribution(
                parameters=["x"],
                distribution_class="norm",
                distribution_parameters={"loc": 1.0, "scale": ub - lb},
            )
        )
        if use_discrete:
            distributions.append(
                ParameterDistribution(
                    parameters=["z"],
                    distribution_class="binom",
                    distribution_parameters={"n": 20, "p": 0.5},
                )
            )
        else:
            distributions.append(
                ParameterDistribution(
                    parameters=["y"],
                    distribution_class="expon",
                    distribution_parameters={},
                )
            )
    return RobustSearchSpace(
        # pyre-ignore Incompatible parameter type [6]
        parameters=parameters,
        parameter_distributions=distributions,
        num_samples=num_samples,
    )


def get_robust_search_space_environmental(
    lb: float = 0.0,
    ub: float = 5.0,
) -> RobustSearchSpace:
    parameters = [
        RangeParameter("x", ParameterType.FLOAT, lb, ub),
        RangeParameter("y", ParameterType.FLOAT, lb, ub),
    ]
    environmental_variables = [
        RangeParameter("z", ParameterType.INT, lb, ub),
    ]
    distributions = [
        ParameterDistribution(
            parameters=["z"],
            distribution_class="binom",
            distribution_parameters={"n": 5, "p": 0.5},
        )
    ]
    return RobustSearchSpace(
        # pyre-ignore Incompatible parameter type [6]
        parameters=parameters,
        parameter_distributions=distributions,
        num_samples=4,
        # pyre-ignore Incompatible parameter type [6]
        environmental_variables=environmental_variables,
    )


##############################
# Trials
##############################


def get_batch_trial(
    abandon_arm: bool = True, experiment: Optional[Experiment] = None
) -> BatchTrial:
    experiment = experiment or get_experiment()
    batch = experiment.new_batch_trial()
    arms = get_arms_from_dict(get_arm_weights1())
    weights = get_weights_from_dict(get_arm_weights1())
    batch.add_arms_and_weights(arms=arms, weights=weights, multiplier=0.75)
    if abandon_arm:
        batch.mark_arm_abandoned(batch.arms[0].name, "abandoned reason")
    batch.runner = SyntheticRunner()
    batch.set_status_quo_with_weight(status_quo=arms[0], weight=0.5)
    batch._generation_step_index = 0
    return batch


def get_batch_trial_with_repeated_arms(num_repeated_arms: int) -> BatchTrial:
    """Create a batch that contains both new arms and N arms from the last
    existed trial in the experiment. Where N is equal to the input argument
    'num_repeated_arms'.
    """
    experiment = get_experiment_with_batch_trial()
    if len(experiment.trials) > 0:
        # Get last (previous) trial.
        prev_trial = experiment.trials[len(experiment.trials) - 1]
        # Take the first N arms, where N is num_repeated_arms.

        if len(prev_trial.arms) < num_repeated_arms:
            logger.warning(
                "There are less arms in the previous trial than the value of "
                "input parameter 'num_repeated_arms'. Thus all the arms from "
                "the last trial will be repeated in the new trial."
            )
        prev_arms = prev_trial.arms[:num_repeated_arms]
        if isinstance(prev_trial, BatchTrial):
            prev_weights = prev_trial.weights[:num_repeated_arms]
        else:
            prev_weights = [1] * len(prev_arms)
    else:
        raise Exception(
            "There are no previous trials in this experiment. Thus the new "
            "batch was not created as no repeated arms could be added."
        )

    # Create new (next) arms.
    next_arms = get_arms_from_dict(get_arm_weights2())
    next_weights = get_weights_from_dict(get_arm_weights2())

    # Add num_repeated_arms to the new trial.
    arms = prev_arms + next_arms
    weights = prev_weights + next_weights
    batch = experiment.new_batch_trial()
    batch.add_arms_and_weights(arms=arms, weights=weights, multiplier=1)
    batch.runner = SyntheticRunner()
    batch.set_status_quo_with_weight(status_quo=arms[0], weight=0.5)
    return batch


def get_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial(ttl_seconds=72)
    arm = get_arms_from_dict(get_arm_weights1())[0]
    trial.add_arm(arm)
    trial.runner = SyntheticRunner()
    trial._generation_step_index = 0
    trial.update_run_metadata({"workflow_run_id": [12345]})
    return trial


def get_hss_trials_with_fixed_parameter(exp: Experiment) -> Dict[int, BaseTrial]:
    return {
        0: Trial(experiment=exp).add_arm(
            arm=Arm(
                parameters={
                    "model": "Linear",
                    "learning_rate": 0.05,
                    "l2_reg_weight": 1e-4,
                    "num_boost_rounds": 15,
                    "z": True,
                },
                name="0_0",
            )
        ),
        1: Trial(experiment=exp).add_arm(
            arm=Arm(
                parameters={
                    "model": "XGBoost",
                    "learning_rate": 0.05,
                    "l2_reg_weight": 1e-4,
                    "num_boost_rounds": 15,
                    "z": True,
                },
                name="1_0",
            )
        ),
    }


class TestTrial(BaseTrial):
    "Trial class to test unsupported trial type error"

    _arms: List[Arm] = []

    def __repr__(self) -> str:
        return "test"

    def _get_candidate_metadata(self, arm_name: str) -> Optional[Dict[str, Any]]:
        return None

    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        return {"test": None}

    def abandoned_arms(self) -> str:
        return "test"

    @property
    def arms(self) -> List[Arm]:
        return self._arms

    @arms.setter
    def arms(self, val: List[Arm]) -> None:
        self._arms = val

    def arms_by_name(self) -> str:
        return "test"

    def generator_runs(self) -> str:
        return "test"


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


def get_task_choice_parameter() -> ChoiceParameter:
    return ChoiceParameter(
        name="y",
        parameter_type=ParameterType.INT,
        # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for 4th
        # parameter `values` to call
        # `ax.core.parameter.ChoiceParameter.__init__` but got `List[str]`.
        values=[1, 2, 3],
        is_task=True,
    )


def get_fixed_parameter() -> FixedParameter:
    return FixedParameter(name="z", parameter_type=ParameterType.BOOL, value=True)


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


def get_order_constraint() -> OrderConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return OrderConstraint(lower_parameter=x, upper_parameter=w)


def get_parameter_constraint(
    param_x: str = "x", param_y: str = "w"
) -> ParameterConstraint:
    return ParameterConstraint(constraint_dict={param_x: 1.0, param_y: -1.0}, bound=1.0)


def get_sum_constraint1() -> SumConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return SumConstraint(parameters=[x, w], is_upper_bound=False, bound=10.0)


def get_sum_constraint2() -> SumConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return SumConstraint(parameters=[x, w], is_upper_bound=True, bound=10.0)


##############################
# Metrics
##############################


def get_metric() -> Metric:
    return Metric(name="m1", properties={"prop": "val"})


def get_branin_metric(name: str = "branin") -> BraninMetric:
    param_names = ["x1", "x2"]
    return BraninMetric(name=name, param_names=param_names, noise_sd=0.01)


def get_augmented_branin_metric(name: str = "aug_branin") -> AugmentedBraninMetric:
    param_names = ["x1", "x2", "fidelity"]
    return AugmentedBraninMetric(name=name, param_names=param_names, noise_sd=0.01)


def get_hartmann_metric(name: str = "hartmann") -> Hartmann6Metric:
    param_names = [f"x{idx + 1}" for idx in range(6)]
    return Hartmann6Metric(name=name, param_names=param_names, noise_sd=0.01)


def get_augmented_hartmann_metric(
    name: str = "aug_hartmann",
) -> AugmentedHartmann6Metric:
    param_names = [f"x{idx + 1}" for idx in range(6)]
    param_names.append("fidelity")
    return AugmentedHartmann6Metric(name=name, param_names=param_names, noise_sd=0.01)


def get_factorial_metric(name: str = "success_metric") -> FactorialMetric:
    coefficients = {
        "factor1": {"level11": 0.1, "level12": 0.2, "level13": 0.3},
        "factor2": {"level21": 0.1, "level22": 0.2},
        "factor3": {"level31": 0.1, "level32": 0.2, "level33": 0.3, "level34": 0.4},
    }
    return FactorialMetric(
        name=name,
        # Expected `Dict[str, Dict[typing.Optional[typing.Union[bool, float, str]],
        # float]]` for 3rd parameter `coefficients` to call
        # `ax.metrics.factorial.FactorialMetric.__init__` but got `Dict[str,
        # Dict[str, float]]`.
        # pyre-fixme[6]:
        coefficients=coefficients,
        batch_size=int(1e4),
    )


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


def get_outcome_constraint() -> OutcomeConstraint:
    return OutcomeConstraint(metric=Metric(name="m2"), op=ComparisonOp.GEQ, bound=-0.25)


def get_scalarized_outcome_constraint() -> ScalarizedOutcomeConstraint:
    return ScalarizedOutcomeConstraint(
        metrics=[Metric(name="oc_m3"), Metric(name="oc_m4")],
        weights=[0.2, 0.8],
        op=ComparisonOp.GEQ,
        bound=-0.25,
    )


def get_branin_outcome_constraint() -> OutcomeConstraint:
    return OutcomeConstraint(metric=get_branin_metric(), op=ComparisonOp.LEQ, bound=0)


##############################
# Objectives
##############################


def get_objective() -> Objective:
    return Objective(metric=Metric(name="m1"), minimize=False)


def get_map_objective() -> Objective:
    return Objective(metric=MapMetric(name="m1"), minimize=False)


def get_multi_objective() -> Objective:
    return MultiObjective(
        objectives=[
            Objective(metric=Metric(name="m1")),
            Objective(metric=Metric(name="m3", lower_is_better=True), minimize=True),
        ],
    )


def get_scalarized_objective() -> Objective:
    return ScalarizedObjective(
        metrics=[Metric(name="m1"), Metric(name="m3")],
        weights=[1.0, 2.0],
        minimize=False,
    )


def get_branin_objective(minimize: bool = False) -> Objective:
    return Objective(metric=get_branin_metric(), minimize=minimize)


def get_branin_multi_objective(num_objectives: int = 2) -> Objective:
    _validate_num_objectives(num_objectives=num_objectives)
    objectives = [
        Objective(metric=get_branin_metric(name="branin_a")),
        Objective(metric=get_branin_metric(name="branin_b")),
    ]
    if num_objectives == 3:
        objectives.append(Objective(metric=get_branin_metric(name="branin_c")))
    return MultiObjective(objectives=objectives)


def get_augmented_branin_objective() -> Objective:
    return Objective(metric=get_augmented_branin_metric(), minimize=False)


def get_hartmann_objective() -> Objective:
    return Objective(metric=get_hartmann_metric(), minimize=False)


def get_augmented_hartmann_objective() -> Objective:
    return Objective(metric=get_augmented_hartmann_metric(), minimize=False)


##############################
# Optimization Configs
##############################


def get_optimization_config() -> OptimizationConfig:
    objective = get_objective()
    outcome_constraints = [get_outcome_constraint()]
    return OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )


def get_map_optimization_config() -> OptimizationConfig:
    objective = get_map_objective()
    return OptimizationConfig(objective=objective)


def get_multi_objective_optimization_config() -> OptimizationConfig:
    objective = get_multi_objective()
    outcome_constraints = [get_outcome_constraint()]
    objective_thresholds = [
        get_objective_threshold(metric_name="m1"),
        get_objective_threshold(metric_name="m3", comparison_op=ComparisonOp.LEQ),
    ]
    return MultiObjectiveOptimizationConfig(
        objective=objective,
        outcome_constraints=outcome_constraints,
        objective_thresholds=objective_thresholds,
    )


def get_optimization_config_no_constraints() -> OptimizationConfig:
    return OptimizationConfig(objective=Objective(metric=Metric("test_metric")))


def get_branin_optimization_config(minimize: bool = False) -> OptimizationConfig:
    return OptimizationConfig(objective=get_branin_objective(minimize=minimize))


def _validate_num_objectives(num_objectives: int) -> None:
    if num_objectives not in (2, 3):
        raise NotImplementedError("Only 2 and 3 objectives are supported.")


def get_branin_multi_objective_optimization_config(
    has_objective_thresholds: bool = False,
    num_objectives: int = 2,
) -> MultiObjectiveOptimizationConfig:
    _validate_num_objectives(num_objectives=num_objectives)
    if has_objective_thresholds:
        objective_thresholds = [
            ObjectiveThreshold(
                metric=get_branin_metric(name="branin_a"),
                bound=10,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
            ObjectiveThreshold(
                metric=get_branin_metric(name="branin_b"),
                bound=20,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
        ]
        if num_objectives == 3:
            objective_thresholds.append(
                ObjectiveThreshold(
                    metric=get_branin_metric(name="branin_c"),
                    bound=5.0,
                    op=ComparisonOp.GEQ,
                    relative=False,
                )
            )
    else:
        objective_thresholds = None
    return MultiObjectiveOptimizationConfig(
        objective=get_branin_multi_objective(num_objectives=num_objectives),
        objective_thresholds=objective_thresholds,
    )


def get_augmented_branin_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=get_augmented_branin_objective())


def get_hartmann_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=get_hartmann_objective())


def get_augmented_hartmann_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=get_augmented_hartmann_objective())


##############################
# Arms
##############################


def get_arm() -> Arm:
    # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
    # parameter `parameters` to call `ax.core.arm.Arm.__init__` but got
    # `Dict[str, typing.Union[float, str]]`.
    return Arm(parameters={"w": 0.75, "x": 1, "y": "foo", "z": True})


def get_status_quo() -> Arm:
    return Arm(
        # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
        # parameter `parameters` to call `ax.core.arm.Arm.__init__`
        # but got `Dict[str, typing.Union[float, str]]`.
        parameters={"w": 0.2, "x": 1, "y": "bar", "z": False},
        name="status_quo",
    )


def get_arm_weights1() -> MutableMapping[Arm, float]:
    parameters_dicts: List[TParameterization] = [
        {"w": 0.85, "x": 1, "y": "baz", "z": False},
        {"w": 0.75, "x": 1, "y": "foo", "z": True},
        {"w": 1.4, "x": 2, "y": "bar", "z": True},
    ]
    arms = [Arm(param_dict) for param_dict in parameters_dicts]
    weights = [0.25, 0.5, 0.25]
    return OrderedDict(zip(arms, weights))


def get_arm_weights2() -> MutableMapping[Arm, float]:  # update
    parameters_dicts: List[TParameterization] = [
        {"w": 0.96, "x": 3, "y": "hello", "z": True},
        {"w": 0.16, "x": 4, "y": "dear", "z": True},
        {"w": 3.1, "x": 5, "y": "world", "z": False},
    ]
    arms = [Arm(param_dict) for param_dict in parameters_dicts]
    weights = [0.25, 0.5, 0.25]
    return OrderedDict(zip(arms, weights))


def get_arms_from_dict(arm_weights_dict: MutableMapping[Arm, float]) -> List[Arm]:
    return list(arm_weights_dict.keys())


def get_weights_from_dict(arm_weights_dict: MutableMapping[Arm, float]) -> List[float]:
    return list(arm_weights_dict.values())


def get_arms() -> List[Arm]:
    return list(get_arm_weights1().keys())


def get_weights() -> List[float]:
    return list(get_arm_weights1().values())


def get_branin_arms(n: int, seed: int) -> List[Arm]:
    # TODO replace with sobol
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
        model_key="Sobol",
        model_kwargs={"scramble": False, "torch_device": torch.device("cpu")},
        bridge_kwargs={"transforms": Cont_X_trans, "torch_dtype": torch.double},
        generation_step_index=0,
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
    assert num_non_sq_arms < 5, "Only up to 4 arms currently handled."
    arm_names = ["status_quo"] if include_sq else []
    arm_names += [f"{trial_index}_{i}" for i in range(num_non_sq_arms)]
    num_arms = num_non_sq_arms + 1 if include_sq else num_non_sq_arms
    df_dict = {
        "trial_index": trial_index,
        "metric_name": metric_name,
        "arm_name": arm_names,
        "mean": [1, 3, 2, 2.25, 1.75][:num_arms],
        "sem": [0, 0.5, 0.25, 0.40, 0.15][:num_arms],
        "n": [100, 100, 100, 100, 100][:num_arms],
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
                },
                {
                    "arm_name": "0_0",
                    "trial_index": "0",
                    "metric_name": "branin_a",  # Obj. metric for experiment.
                    "mean": 1.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=3),
                    "end_time": now,
                },
                {
                    "arm_name": "status_quo",
                    "trial_index": "0",
                    "metric_name": "branin_b",  # Obj. metric for experiment.
                    "mean": 2.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=2),
                    "end_time": now - timedelta(days=1),
                },
                {
                    "arm_name": "0_0",
                    "trial_index": "0",
                    "metric_name": "branin_b",  # Obj. metric for experiment.
                    "mean": 1.0,
                    "sem": 0.01,
                    "start_time": now - timedelta(days=2),
                    "end_time": now - timedelta(days=1),
                },
            ]
        )
    )


def get_map_data(trial_index: int = 0) -> MapData:
    evaluations = {
        "status_quo": [
            ({"epoch": 1}, {"ax_test_metric": (1.0, 0.5)}),
            ({"epoch": 2}, {"ax_test_metric": (2.0, 0.5)}),
            ({"epoch": 3}, {"ax_test_metric": (3.0, 0.5)}),
            ({"epoch": 4}, {"ax_test_metric": (4.0, 0.5)}),
        ],
        "0_0": [
            ({"epoch": 1}, {"ax_test_metric": (3.7, 0.5)}),
            ({"epoch": 2}, {"ax_test_metric": (3.8, 0.5)}),
            ({"epoch": 3}, {"ax_test_metric": (3.9, 0.5)}),
            ({"epoch": 4}, {"ax_test_metric": (4.0, 0.5)}),
        ],
        "0_1": [
            ({"epoch": 1}, {"ax_test_metric": (3.0, 0.5)}),
            ({"epoch": 2}, {"ax_test_metric": (5.0, 0.5)}),
            ({"epoch": 3}, {"ax_test_metric": (6.0, 0.5)}),
            ({"epoch": 4}, {"ax_test_metric": (1.0, 0.5)}),
        ],
    }
    return MapData.from_map_evaluations(
        evaluations=evaluations,  # pyre-ignore [6]: Spurious param type mismatch.
        trial_index=trial_index,
        map_key_infos=[get_map_key_info()],
    )


# pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
def get_map_key_info() -> MapKeyInfo:
    return MapKeyInfo(key="epoch", default_value=0.0)


def get_branin_data(
    trial_indices: Optional[Iterable[int]] = None,
    trials: Optional[Iterable[Trial]] = None,
) -> Data:
    if trial_indices and trials:
        raise ValueError("Expected `trial_indices` or `trials`, not both.")
    if trials:
        df_dicts = [
            {
                "trial_index": trial.index,
                "metric_name": "branin",
                "arm_name": not_none(checked_cast(Trial, trial).arm).name,
                "mean": branin(
                    float(not_none(trial.arm).parameters["x1"]),  # pyre-ignore[6]
                    float(not_none(trial.arm).parameters["x2"]),  # pyre-ignore[6]
                ),
                "sem": 0.0,
            }
            for trial in trials
        ]
    else:
        df_dicts = [
            {
                "trial_index": trial_index,
                "metric_name": "branin",
                "arm_name": f"{trial_index}_0",
                "mean": 5.0,
                "sem": 0.0,
            }
            for trial_index in (trial_indices or [0])
        ]
    return Data(df=pd.DataFrame.from_records(df_dicts))


def get_branin_data_batch(batch: BatchTrial) -> Data:
    return Data(
        pd.DataFrame(
            {
                "arm_name": [arm.name for arm in batch.arms],
                "metric_name": "branin",
                "mean": [
                    # pyre-ignore[6]: This function can fail if a parameter value
                    # does not support conversion to float.
                    branin(float(arm.parameters["x1"]), float(arm.parameters["x2"]))
                    for arm in batch.arms
                ],
                "sem": 0.1,
            }
        )
    )


def get_branin_data_multi_objective(
    trial_indices: Optional[Iterable[int]] = None, num_objectives: int = 2
) -> Data:
    _validate_num_objectives(num_objectives=num_objectives)
    suffixes = ["a", "b"]
    if num_objectives == 3:
        suffixes.append("c")
    df_dicts = [
        {
            "trial_index": trial_index,
            "metric_name": f"branin_{suffix}",
            "arm_name": f"{trial_index}_0",
            "mean": 5.0,
            "sem": 0.0,
        }
        for trial_index in (trial_indices or [0])
        for suffix in suffixes
    ]
    return Data(df=pd.DataFrame.from_records(df_dicts))


def get_percentile_early_stopping_strategy() -> PercentileEarlyStoppingStrategy:
    return PercentileEarlyStoppingStrategy(
        percentile_threshold=0.25,
        min_progression=0.2,
        min_curves=10,
        trial_indices_to_ignore=[0, 1, 2],
        normalize_progressions=True,
    )


def get_percentile_early_stopping_strategy_with_true_objective_metric_name() -> PercentileEarlyStoppingStrategy:  # noqa
    strategy = get_percentile_early_stopping_strategy()
    strategy.true_objective_metric_name = "true_objective"
    return strategy


def get_percentile_early_stopping_strategy_with_non_objective_metric_name() -> PercentileEarlyStoppingStrategy:  # noqa
    return PercentileEarlyStoppingStrategy(
        metric_names=["foo"],
        percentile_threshold=0.25,
        min_progression=0.2,
        min_curves=10,
        trial_indices_to_ignore=[0, 1, 2],
        normalize_progressions=True,
    )


def get_threshold_early_stopping_strategy() -> ThresholdEarlyStoppingStrategy:
    return ThresholdEarlyStoppingStrategy(
        true_objective_metric_name="true_objective",
        metric_threshold=0.1,
        min_progression=0.2,
        trial_indices_to_ignore=[0, 1, 2],
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
    def __init__(
        self, early_stop_trials: Optional[Dict[int, Optional[str]]] = None
    ) -> None:
        self.early_stop_trials: Dict[int, Optional[str]] = early_stop_trials or {}

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
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

    def should_stop_optimization(
        self, experiment: Experiment, **kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
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


def get_model_predictions_per_arm() -> Dict[str, TModelPredictArm]:
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


def get_botorch_model() -> BoTorchModel:
    return BoTorchModel(
        surrogate=get_surrogate(), acquisition_class=get_acquisition_type()
    )


def get_botorch_model_with_default_acquisition_class() -> BoTorchModel:
    return BoTorchModel(
        surrogate=get_surrogate(),
        acquisition_class=Acquisition,
        botorch_acqf_class=get_acquisition_function_type(),
    )


def get_surrogate() -> Surrogate:
    return Surrogate(
        botorch_model_class=get_model_type(),
        mll_class=get_mll_type(),
        model_options={"some_option": "some_value"},
    )


def get_acquisition_type() -> Type[Acquisition]:
    return Acquisition


def get_model_type() -> Type[Model]:
    return SingleTaskGP


def get_mll_type() -> Type[MarginalLogLikelihood]:
    return ExactMarginalLogLikelihood


def get_acquisition_function_type() -> Type[AcquisitionFunction]:
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


##############################
# Scheduler
##############################


def get_default_scheduler_options() -> SchedulerOptions:
    return SchedulerOptions()


def get_scheduler_options_batch_trial() -> SchedulerOptions:
    return SchedulerOptions(trial_type=TrialType.BATCH_TRIAL)


##############################
# Other
##############################


def get_risk_measure() -> RiskMeasure:
    return RiskMeasure(risk_measure="Expectation", options={"n_w": 8})


def get_parameter_distribution() -> ParameterDistribution:
    return ParameterDistribution(
        parameters=["x"],
        distribution_class="norm",
        distribution_parameters={"loc": 1.0, "scale": 0.5},
    )


##############################
# Custom runner and metric
##############################


class CustomTestRunner(Runner):
    def __init__(self, test_attribute: str) -> None:
        self.test_attribute = test_attribute

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        return {"foo": "bar"}


class CustomTestMetric(Metric):
    def __init__(self, name: str, test_attribute: str) -> None:
        self.test_attribute = test_attribute
        super().__init__(name=name)
