#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, MutableMapping

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types.types import (
    ComparisonOp,
    TModelCov,
    TModelMean,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
)
from ax.metrics.branin import BraninMetric
from ax.metrics.factorial import FactorialMetric
from ax.modelbridge.factory import get_factorial, get_sobol
from ax.runners.synthetic import SyntheticRunner


# Experiments


def get_experiment() -> Experiment:
    return Experiment(
        name="test",
        search_space=get_search_space(),
        optimization_config=get_optimization_config(),
        status_quo=get_status_quo(),
        description="test description",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )


def get_branin_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=get_branin_objective())


def get_branin_experiment(
    has_optimization_config: bool = True,
    with_batch: bool = False,
    with_status_quo: bool = False,
) -> Experiment:
    exp = Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(),
        optimization_config=get_branin_optimization_config()
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
        is_test=True,
    )

    if with_status_quo:
        exp.status_quo = Arm(params={"x1": 0, "x2": 0})

    if with_batch:
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=15)
        exp.new_batch_trial().add_generator_run(sobol_run)

    return exp


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
    )

    if with_status_quo:
        exp.status_quo = Arm(
            params={"factor1": "level11", "factor2": "level21", "factor3": "level31"}
        )

    if with_batch:
        factorial_generator = get_factorial(search_space=exp.search_space)
        factorial_run = factorial_generator.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run)

    return exp


# Search Spaces


def get_search_space() -> SearchSpace:
    parameters = [
        get_range_parameter(),
        get_range_parameter2(),
        get_choice_parameter(),
        get_fixed_parameter(),
    ]
    return SearchSpace(
        # pyre: Expected `List[ax.core.parameter.Parameter]` for 1st
        # pyre: parameter `parameters` to call `ax.core.search_space.
        # pyre: SearchSpace.__init__` but got `List[typing.
        # pyre-fixme[6]: Union[ChoiceParameter, FixedParameter, RangeParameter]]`.
        parameters=parameters,
        parameter_constraints=[get_order_constraint()],
    )


def get_branin_search_space() -> SearchSpace:
    parameters = [
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
    # Expected `List[ax.core.parameter.Parameter]` for 2nd parameter
    # `parameters` to call `ax.core.search_space.SearchSpace.__init__` but got
    # `List[RangeParameter]`.
    # pyre-fixme[6]:
    return SearchSpace(parameters=parameters)


def get_factorial_search_space() -> SearchSpace:
    return SearchSpace(
        # Expected `List[ax.core.parameter.Parameter]` for 2nd parameter
        # `parameters` to call `ax.core.search_space.SearchSpace.__init__` but
        # got `List[ChoiceParameter]`.
        parameters=[
            ChoiceParameter(
                name="factor1",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ax.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level11", "level12", "level13"],
            ),
            ChoiceParameter(
                name="factor2",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ax.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level21", "level22"],
            ),
            ChoiceParameter(
                name="factor3",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ax.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level31", "level32", "level33", "level34"],
            ),
        ]
    )


# trials


def get_batch_trial() -> BatchTrial:
    experiment = get_experiment()
    batch = experiment.new_batch_trial()
    arms = get_arms()
    weights = get_weights()
    batch.add_arms_and_weights(arms=arms, weights=weights, multiplier=0.75)
    batch.mark_arm_abandoned(batch.arms[0].name, "abandoned reason")
    batch.runner = SyntheticRunner()
    batch.set_status_quo(status_quo=arms[0], weight=0.5)
    return batch


def get_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial()
    arm = get_arms()[0]
    trial.add_arm(arm)
    trial.runner = SyntheticRunner()
    return trial


def get_experiment_with_batch_trial() -> Experiment:
    batch_trial = get_batch_trial()
    return batch_trial.experiment


def get_experiment_with_batch_and_single_trial() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.new_trial(generator_run=GeneratorRun(arms=[get_arm()]))
    return batch_trial.experiment


def get_experiment_with_data() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.attach_data(data=get_data())
    return batch_trial.experiment


# Parameters


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


def get_fixed_parameter() -> FixedParameter:
    return FixedParameter(name="z", parameter_type=ParameterType.BOOL, value=True)


# Parameter Constraints


def get_order_constraint() -> OrderConstraint:
    return OrderConstraint(lower_name="x", upper_name="w")


def get_parameter_constraint() -> ParameterConstraint:
    return ParameterConstraint(constraint_dict={"x": 1.0, "w": -1.0}, bound=1.0)


def get_sum_constraint1() -> SumConstraint:
    return SumConstraint(parameter_names=["x", "w"], is_upper_bound=False, bound=10.0)


def get_sum_constraint2() -> SumConstraint:
    return SumConstraint(parameter_names=["x", "w"], is_upper_bound=True, bound=10.0)


# Metrics


def get_metric() -> Metric:
    return Metric(name="m1")


def get_branin_metric() -> BraninMetric:
    return BraninMetric(name="branin", param_names=["x1", "x2"], noise_sd=0.01)


def get_factorial_metric() -> FactorialMetric:
    coefficients = {
        "factor1": {"level11": 0.1, "level12": 0.2, "level13": 0.3},
        "factor2": {"level21": 0.1, "level22": 0.2},
        "factor3": {"level31": 0.1, "level32": 0.2, "level33": 0.3, "level34": 0.4},
    }
    return FactorialMetric(
        name="success_metric",
        # Expected `Dict[str, Dict[typing.Optional[typing.Union[bool, float, str]],
        # float]]` for 3rd parameter `coefficients` to call
        # `ax.metrics.factorial.FactorialMetric.__init__` but got `Dict[str,
        # Dict[str, float]]`.
        # pyre-fixme[6]:
        coefficients=coefficients,
        batch_size=int(1e4),
    )


# Optimization Configs


def get_objective() -> Objective:
    return Objective(metric=Metric(name="m1"), minimize=False)


def get_outcome_constraint() -> OutcomeConstraint:
    return OutcomeConstraint(metric=Metric(name="m2"), op=ComparisonOp.GEQ, bound=-0.25)


def get_optimization_config() -> OptimizationConfig:
    objective = get_objective()
    outcome_constraints = [get_outcome_constraint()]
    return OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )


def get_branin_objective() -> Objective:
    return Objective(metric=get_branin_metric(), minimize=False)


def get_branin_outcome_constraint() -> OutcomeConstraint:
    return OutcomeConstraint(metric=get_branin_metric(), op=ComparisonOp.LEQ, bound=0)


# Arms


def get_arm() -> Arm:
    # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
    # parameter `params` to call `ax.core.arm.Arm.__init__` but got
    # `Dict[str, typing.Union[float, str]]`.
    return Arm(params={"w": 0.75, "x": 1, "y": "foo", "z": True})


def get_status_quo() -> Arm:
    return Arm(
        # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
        # parameter `params` to call `ax.core.arm.Arm.__init__`
        # but got `Dict[str, typing.Union[float, str]]`.
        params={"w": 0.2, "x": 1, "y": "bar", "z": False},
        name="status_quo",
    )


def get_arm_weights() -> MutableMapping[Arm, float]:
    # pyre: params_dicts is declared to have type `List[Dict[str, typing.
    # pyre: Optional[typing.Union[bool, float, str]]]]` but is used as type
    # pyre-fixme[9]: `List[Dict[str, typing.Union[float, str]]]`.
    params_dicts: List[TParameterization] = [
        {"w": 0.85, "x": 1, "y": "baz", "z": False},
        {"w": 0.75, "x": 1, "y": "foo", "z": True},
        {"w": 1.4, "x": 2, "y": "bar", "z": True},
    ]
    arms = [Arm(param_dict) for param_dict in params_dicts]
    weights = [0.25, 0.5, 0.25]
    return OrderedDict(zip(arms, weights))


def get_arms() -> List[Arm]:
    return list(get_arm_weights().keys())


def get_weights() -> List[float]:
    return list(get_arm_weights().values())


def get_branin_arms(n: int, seed: int) -> List[Arm]:
    # TODO replace with sobol
    np.random.seed(seed)
    x1_raw = np.random.rand(n)
    x2_raw = np.random.rand(n)
    return [
        Arm(params={"x1": -5 + x1_raw[i] * 15, "x2": x2_raw[i] * 15}) for i in range(n)
    ]


def get_abandoned_arm() -> AbandonedArm:
    return AbandonedArm(name="0_0", reason="foobar", time=datetime.now())


# Generator Runs


def get_generator_run() -> GeneratorRun:
    arms = get_arms()
    weights = get_weights()
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
    )


def get_generator_run2() -> GeneratorRun:
    arms = get_arms()
    weights = get_weights()
    return GeneratorRun(arms=arms, weights=weights)


# Runners


def get_synthetic_runner() -> SyntheticRunner:
    return SyntheticRunner(dummy_metadata="foobar")


# Model Utilities


def get_model_mean() -> TModelMean:
    # pyre: mean is declared to have type `Dict[str, List[float]]` but is used
    # pyre-fixme[9]: as type `Dict[str, List[int]]`.
    mean: TModelMean = {"test_metric_1": [1, 2, 3], "test_metric_2": [3, 4, 5]}
    return mean


def get_model_covariance() -> TModelCov:
    # pyre: covariance is declared to have type `Dict[str, Dict[str,
    # pyre: List[float]]]` but is used as type `Dict[str, Dict[str,
    # pyre-fixme[9]: List[int]]]`.
    covariance: TModelCov = {
        "test_metric_1": {"test_metric_1": [5, 6, 7], "test_metric_2": [7, 8, 9]},
        "test_metric_2": {"test_metric_1": [9, 10, 11], "test_metric_2": [11, 12, 13]},
    }
    return covariance


def get_model_predictions() -> TModelPredict:
    model_predictions: TModelPredict = (get_model_mean(), get_model_covariance())
    return model_predictions


def get_model_predictions_per_arm() -> Dict[str, TModelPredictArm]:
    arms = list(get_arm_weights().keys())
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


# Data


def get_data() -> Data:
    df_dict = {
        "trial_index": 0,
        "metric_name": "ax_test_metric",
        "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3"],
        "mean": [1, 3, 2, 2.25, 1.75],
        "sem": [0, 0.5, 0.25, 0.40, 0.15],
        "n": [100, 100, 100, 100, 100],
    }
    return Data(df=pd.DataFrame.from_records(df_dict))
