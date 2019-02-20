#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, MutableMapping

import numpy as np
from ae.lazarus.ae.core.batch_trial import AbandonedCondition, BatchTrial
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ae.lazarus.ae.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.core.types.types import (
    ComparisonOp,
    TModelCov,
    TModelMean,
    TModelPredict,
    TModelPredictCondition,
    TParameterization,
)
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.metrics.factorial import FactorialMetric
from ae.lazarus.ae.runners.synthetic import SyntheticRunner


# Experiments


def get_experiment() -> Experiment:
    return Experiment(
        name="test",
        search_space=get_search_space(),
        optimization_config=get_optimization_config(),
        status_quo=get_status_quo(),
        description="test description",
        tracking_metrics=[Metric(name="tracking")],
    )


def get_branin_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(
        objective=Objective(metric=get_branin_metric(), minimize=True)
    )


def get_branin_experiment(has_optimization_config: bool = True) -> Experiment:
    return Experiment(
        name="branin_test_experiment",
        search_space=get_branin_search_space(),
        optimization_config=get_branin_optimization_config()
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
    )


def get_factorial_experiment(has_optimization_config: bool = True) -> Experiment:
    return Experiment(
        name="factorial_test_experiment",
        search_space=get_factorial_search_space(),
        optimization_config=OptimizationConfig(
            objective=Objective(metric=get_factorial_metric(), minimize=True)
        )
        if has_optimization_config
        else None,
        runner=SyntheticRunner(),
    )


# Search Spaces


def get_search_space() -> SearchSpace:
    parameters = [
        get_range_parameter(),
        get_range_parameter2(),
        get_choice_parameter(),
        get_fixed_parameter(),
    ]
    return SearchSpace(
        # pyre: Expected `List[ae.lazarus.ae.core.parameter.Parameter]` for 1st
        # pyre: parameter `parameters` to call `ae.lazarus.ae.core.search_space.
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
    # Expected `List[ae.lazarus.ae.core.parameter.Parameter]` for 2nd parameter
    # `parameters` to call `ae.lazarus.ae.core.search_space.SearchSpace.__init__` but got
    # `List[RangeParameter]`.
    # pyre-fixme[6]:
    return SearchSpace(parameters=parameters)


def get_factorial_search_space() -> SearchSpace:
    return SearchSpace(
        # Expected `List[ae.lazarus.ae.core.parameter.Parameter]` for 2nd parameter
        # `parameters` to call `ae.lazarus.ae.core.search_space.SearchSpace.__init__` but
        # got `List[ChoiceParameter]`.
        parameters=[
            ChoiceParameter(
                name="factor1",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ae.lazarus.ae.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level11", "level12", "level13"],
            ),
            ChoiceParameter(
                name="factor2",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ae.lazarus.ae.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level21", "level22"],
            ),
            ChoiceParameter(
                name="factor3",
                parameter_type=ParameterType.STRING,
                # Expected `List[typing.Optional[typing.Union[bool, float, str]]]` for
                # 4th parameter `values` to call
                # `ae.lazarus.ae.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                values=["level31", "level32", "level33", "level34"],
            ),
        ]
    )


# trials


def get_batch_trial() -> BatchTrial:
    experiment = get_experiment()
    batch = experiment.new_batch_trial()
    conditions = get_conditions()
    weights = get_weights()
    batch.add_conditions_and_weights(
        conditions=conditions, weights=weights, multiplier=0.75
    )
    batch.mark_condition_abandoned(batch.conditions[0], "abandoned reason")
    batch.runner = SyntheticRunner()
    batch.set_status_quo(status_quo=conditions[0], weight=0.5)
    return batch


def get_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial()
    condition = get_conditions()[0]
    trial.add_condition(condition)
    trial.runner = SyntheticRunner()
    return trial


def get_experiment_with_batch_trial() -> Experiment:
    batch_trial = get_batch_trial()
    return batch_trial.experiment


def get_experiment_with_batch_and_single_trial() -> Experiment:
    batch_trial = get_batch_trial()
    batch_trial.experiment.new_trial(
        generator_run=GeneratorRun(conditions=[get_condition()])
    )
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
        # `ae.lazarus.ae.core.parameter.ChoiceParameter.__init__` but got `List[str]`.
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
    return BraninMetric(name="branin", param_names=["x1", "x2"], noise_sd=0.0)


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
        # `ae.lazarus.ae.metrics.factorial.FactorialMetric.__init__` but got `Dict[str,
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
    return Objective(metric=get_branin_metric(), minimize=True)


def get_branin_outcome_constraint() -> OutcomeConstraint:
    return OutcomeConstraint(metric=get_branin_metric(), op=ComparisonOp.LEQ, bound=0)


# Conditions


def get_condition() -> Condition:
    # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
    # parameter `params` to call `ae.lazarus.ae.core.condition.Condition.__init__` but got
    # `Dict[str, typing.Union[float, str]]`.
    return Condition(params={"w": 0.75, "x": 1, "y": "foo", "z": True})


def get_status_quo() -> Condition:
    return Condition(
        # Expected `Dict[str, typing.Optional[typing.Union[bool, float, str]]]` for 2nd
        # parameter `params` to call `ae.lazarus.ae.core.condition.Condition.__init__`
        # but got `Dict[str, typing.Union[float, str]]`.
        params={"w": 0.2, "x": 1, "y": "bar", "z": False},
        name="status_quo",
    )


def get_condition_weights() -> MutableMapping[Condition, float]:
    # pyre: params_dicts is declared to have type `List[Dict[str, typing.
    # pyre: Optional[typing.Union[bool, float, str]]]]` but is used as type
    # pyre-fixme[9]: `List[Dict[str, typing.Union[float, str]]]`.
    params_dicts: List[TParameterization] = [
        {"w": 0.85, "x": 1, "y": "baz", "z": False},
        {"w": 0.75, "x": 1, "y": "foo", "z": True},
        {"w": 1.4, "x": 2, "y": "bar", "z": True},
    ]
    conditions = [Condition(param_dict) for param_dict in params_dicts]
    weights = [0.25, 0.5, 0.25]
    return OrderedDict(zip(conditions, weights))


def get_conditions() -> List[Condition]:
    return list(get_condition_weights().keys())


def get_weights() -> List[float]:
    return list(get_condition_weights().values())


def get_branin_conditions(n: int, seed: int) -> List[Condition]:
    # TODO replace with sobol
    np.random.seed(seed)
    x1_raw = np.random.rand(n)
    x2_raw = np.random.rand(n)
    return [
        Condition(params={"x1": -5 + x1_raw[i] * 15, "x2": x2_raw[i] * 15})
        for i in range(n)
    ]


def get_abandoned_condition() -> AbandonedCondition:
    return AbandonedCondition(name="0_0", reason="foobar", time=datetime.now())


# Generator Runs


def get_generator_run() -> GeneratorRun:
    conditions = get_conditions()
    weights = get_weights()
    optimization_config = get_optimization_config()
    search_space = get_search_space()
    condition_predictions = get_model_predictions_per_condition()
    return GeneratorRun(
        conditions=conditions,
        weights=weights,
        optimization_config=optimization_config,
        search_space=search_space,
        model_predictions=get_model_predictions(),
        best_condition_predictions=(
            conditions[0],
            condition_predictions[conditions[0].signature],
        ),
        fit_time=10.0,
        gen_time=5.0,
    )


def get_generator_run2() -> GeneratorRun:
    conditions = get_conditions()
    weights = get_weights()
    return GeneratorRun(conditions=conditions, weights=weights)


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


def get_model_predictions_per_condition() -> Dict[str, TModelPredictCondition]:
    conditions = list(get_condition_weights().keys())
    means = get_model_mean()
    covariances = get_model_covariance()
    metric_names = list(means.keys())
    m_1, m_2 = metric_names[0], metric_names[1]
    return {
        conditions[i].signature: (
            {m_1: means[m_1][i], m_2: means[m_2][i]},
            {
                m_1: {m_1: covariances[m_1][m_1][i], m_2: covariances[m_1][m_2][i]},
                m_2: {m_1: covariances[m_2][m_1][i], m_2: covariances[m_2][m_2][i]},
            },
        )
        for i in range(len(conditions))
    }
