#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from datetime import datetime
from typing import Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
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
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.core.types import (
    ComparisonOp,
    TModelCov,
    TModelMean,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
)
from ax.metrics.branin import BraninMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.factory import Cont_X_trans, get_factorial, get_sobol
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.logger import get_logger


logger = get_logger("ae_experiment")


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
        exp.status_quo = Arm(parameters={"x1": 0.0, "x2": 0.0})

    if with_batch:
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=15)
        exp.new_batch_trial(optimize_for_power=with_status_quo).add_generator_run(
            sobol_run
        )

    return exp


def get_multi_type_experiment(
    add_trial_type: bool = True, add_trials: bool = False
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


def get_simple_experiment() -> SimpleExperiment:
    experiment = SimpleExperiment(
        name="test_branin",
        search_space=get_branin_search_space(),
        status_quo=Arm(parameters={"x1": 0.0, "x2": 0.0}),
        objective_name="sum",
    )

    experiment.description = "foobar"

    return experiment


def get_simple_experiment_with_batch_trial() -> SimpleExperiment:
    experiment = get_simple_experiment()
    generator = get_sobol(experiment.search_space)
    generator_run = generator.gen(10)
    experiment.new_batch_trial(generator_run=generator_run)
    return experiment


def get_experiment_with_repeated_arms(num_repeated_arms: int) -> Experiment:
    batch_trial = get_batch_trial_with_repeated_arms(num_repeated_arms)
    return batch_trial.experiment


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
    batch_trial.experiment.attach_data(data=get_data())
    batch_trial.experiment.attach_data(data=get_data())
    return batch_trial.experiment


def get_experiment_with_multi_objective() -> Experiment:
    objective = get_multi_objective()
    outcome_constraints = [get_outcome_constraint()]
    optimization_config = OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )
    return Experiment(
        name="test_experiment_multi_objective",
        search_space=get_search_space(),
        optimization_config=optimization_config,
        status_quo=get_status_quo(),
        description="test experiment with scalarized objective",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
    )


def get_experiment_with_scalarized_objective() -> Experiment:
    objective = get_scalarized_objective()
    outcome_constraints = [get_outcome_constraint()]
    optimization_config = OptimizationConfig(
        objective=objective, outcome_constraints=outcome_constraints
    )
    return Experiment(
        name="test_experiment_scalarized_objective",
        search_space=get_search_space(),
        optimization_config=optimization_config,
        status_quo=get_status_quo(),
        description="test experiment with scalarized objective",
        tracking_metrics=[Metric(name="tracking")],
        is_test=True,
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
        # pyre: Expected `List[ax.core.parameter.Parameter]` for 1st
        # pyre: parameter `parameters` to call `ax.core.search_space.
        # pyre: SearchSpace.__init__` but got `List[typing.
        # pyre-fixme[6]: Union[ChoiceParameter, FixedParameter, RangeParameter]]`.
        parameters=parameters,
        parameter_constraints=[
            get_order_constraint(),
            get_parameter_constraint(),
            get_sum_constraint1(),
        ],
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
            ChoiceParameter("z", ParameterType.STRING, ["red", "panda"]),
        ]
    )


# Trials


def get_batch_trial(abandon_arm: bool = True) -> BatchTrial:
    experiment = get_experiment()
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
    """ Create a batch that contains both new arms and N arms from the last
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
    # pyre-fixme[6]: Expected `List[int]` for 1st param but got `List[float]`.
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
    return trial


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
    w = get_range_parameter()
    x = get_range_parameter2()
    return OrderConstraint(lower_parameter=x, upper_parameter=w)


def get_parameter_constraint() -> ParameterConstraint:
    return ParameterConstraint(constraint_dict={"x": 1.0, "w": -1.0}, bound=1.0)


def get_sum_constraint1() -> SumConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return SumConstraint(parameters=[x, w], is_upper_bound=False, bound=10.0)


def get_sum_constraint2() -> SumConstraint:
    w = get_range_parameter()
    x = get_range_parameter2()
    return SumConstraint(parameters=[x, w], is_upper_bound=True, bound=10.0)


# Metrics


def get_metric() -> Metric:
    return Metric(name="m1")


def get_branin_metric(name="branin") -> BraninMetric:
    return BraninMetric(name=name, param_names=["x1", "x2"], noise_sd=0.01)


def get_hartmann_metric() -> Hartmann6Metric:
    return Hartmann6Metric(name="hartmann", param_names=["x1", "x2"], noise_sd=0.01)


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


# Optimization Configs


def get_objective() -> Objective:
    return Objective(metric=Metric(name="m1"), minimize=False)


def get_multi_objective() -> Objective:
    return MultiObjective(
        metrics=[Metric(name="m1"), Metric(name="m3", lower_is_better=True)],
        minimize=False,
    )


def get_scalarized_objective() -> Objective:
    return ScalarizedObjective(
        metrics=[Metric(name="m1"), Metric(name="m3")],
        weights=[1.0, 2.0],
        minimize=False,
    )


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


def get_optimization_config_no_constraints() -> OptimizationConfig:
    return OptimizationConfig(objective=Objective(metric=Metric("test_metric")))


def get_branin_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=get_branin_objective())


# Arms


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


# Generator Runs


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


# Runners


def get_synthetic_runner() -> SyntheticRunner:
    return SyntheticRunner(dummy_metadata="foobar")


# Data


def get_data(trial_index: int = 0) -> Data:
    df_dict = {
        "trial_index": trial_index,
        "metric_name": "ax_test_metric",
        "arm_name": ["status_quo"] + [f"{trial_index}_i" for i in range(4)],
        "mean": [1, 3, 2, 2.25, 1.75],
        "sem": [0, 0.5, 0.25, 0.40, 0.15],
        "n": [100, 100, 100, 100, 100],
    }
    return Data(df=pd.DataFrame.from_records(df_dict))


def get_branin_data(trial_indices: Optional[Iterable[int]] = None) -> Data:
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


# Instances of types from core/types.py


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
