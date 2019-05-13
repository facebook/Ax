#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import enum
from collections import OrderedDict
from typing import Any, Dict, List

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data  # noqa F401
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import (
    SimpleExperiment,
    unimplemented_evaluation_function,
)
from ax.exceptions.storage import JSONDecodeError
from ax.storage.json_store.decoders import batch_trial_from_json, trial_from_json
from ax.storage.json_store.registry import DECODER_REGISTRY


def object_from_json(object_json: Any) -> Any:
    """Recursively load objects from a JSON-serializable dictionary."""
    if type(object_json) in (str, int, float, bool, type(None)):
        return object_json
    elif isinstance(object_json, list):
        return [object_from_json(i) for i in object_json]
    elif isinstance(object_json, tuple):
        return tuple(object_from_json(i) for i in object_json)
    elif isinstance(object_json, dict):
        if "__type" not in object_json:
            # this is just a regular dictionary, e.g. the one in Parameter
            # containing parameterizations
            return {k: object_from_json(v) for k, v in object_json.items()}

        _type = object_json.pop("__type")

        if _type == "datetime":
            return datetime.datetime.strptime(
                object_json["value"], "%Y-%m-%d %H:%M:%S.%f"
            )
        elif _type == "OrderedDict":
            return OrderedDict(
                [(k, object_from_json(v)) for k, v in object_json["value"]]
            )
        elif _type == "DataFrame":
            # Need dtype=False, otherwise infers arm_names like "4_1"
            # should be int 41
            return pd.read_json(object_json["value"], dtype=False)
        elif _type not in DECODER_REGISTRY:
            err = (
                f"The JSON dictionary passed to `object_from_json` has a type "
                f"{_type} that is not registered with a corresponding class in "
                f"DECODER_REGISTRY."
            )
            raise JSONDecodeError(err)

        _class = DECODER_REGISTRY[_type]

        if issubclass(_class, enum.Enum):
            # to access enum members by name, use item access
            return _class[object_json["name"]]
        elif _class == GeneratorRun:
            return generator_run_from_json(object_json=object_json)
        elif _class == SimpleExperiment:
            return simple_experiment_from_json(object_json=object_json)
        elif _class == Experiment:
            return experiment_from_json(object_json=object_json)
        elif _class == SearchSpace:
            return search_space_from_json(search_space_json=object_json)

        return _class(**{k: object_from_json(v) for k, v in object_json.items()})
    else:
        err = (
            f"The object passed to `object_from_json` has an unsupported type: "
            f"{type(object_json)}."
        )
        raise JSONDecodeError(err)


def generator_run_from_json(object_json: Dict[str, Any]) -> GeneratorRun:
    """Load Ax GeneratorRun from JSON."""
    time_created_json = object_json.pop("time_created")
    type_json = object_json.pop("generator_run_type")
    index_json = object_json.pop("index")
    generator_run = GeneratorRun(
        **{k: object_from_json(v) for k, v in object_json.items()}
    )
    generator_run._time_created = object_from_json(time_created_json)
    generator_run._generator_run_type = object_from_json(type_json)
    generator_run._index = object_from_json(index_json)
    return generator_run


def search_space_from_json(search_space_json: Dict[str, Any]) -> SearchSpace:
    """Load a SearchSpace from JSON.

    This function is necessary due to the coupled loading of SearchSpace
    and parameter constraints.
    """
    parameters = object_from_json(search_space_json.pop("parameters"))
    json_param_constraints = search_space_json.pop("parameter_constraints")
    return SearchSpace(
        parameters=parameters,
        parameter_constraints=parameter_constraints_from_json(
            parameter_constraint_json=json_param_constraints, parameters=parameters
        ),
    )


def parameter_constraints_from_json(
    parameter_constraint_json: List[Dict[str, Any]], parameters: List[Parameter]
) -> List[ParameterConstraint]:
    """Load ParameterConstraints from JSON.

    Order and SumConstraint are tied to a search space,
    and require that SearchSpace's parameters to be passed in for decoding.

    Args:
        parameter_constraint_json: JSON representation of parameter constraints.
        parameters: Parameter definitions for decoding via parameter names.

    Returns:
        parameter_constraints: Python classes for parameter constraints.
    """
    parameter_constraints = []
    parameter_map = {p.name: p for p in parameters}
    for constraint in parameter_constraint_json:
        if constraint["__type"] == "OrderConstraint":
            lower_parameter = parameter_map[constraint["lower_name"]]
            upper_parameter = parameter_map[constraint["upper_name"]]
            parameter_constraints.append(
                OrderConstraint(
                    lower_parameter=lower_parameter, upper_parameter=upper_parameter
                )
            )
        elif constraint["__type"] == "SumConstraint":
            parameters = [parameter_map[name] for name in constraint["parameter_names"]]
            parameter_constraints.append(
                SumConstraint(
                    parameters=parameters,
                    is_upper_bound=constraint["is_upper_bound"],
                    bound=constraint["bound"],
                )
            )
        else:
            parameter_constraints.append(object_from_json(constraint))
    return parameter_constraints


def trials_from_json(
    experiment: Experiment, trials_json: Dict[str, Any]
) -> Dict[int, BaseTrial]:
    """Load Ax Trials from JSON."""
    loaded_trials = {}
    for index, batch_json in trials_json.items():
        is_trial = batch_json["__type"] == "Trial"
        batch_json = {
            k: object_from_json(v) for k, v in batch_json.items() if k != "__type"
        }
        loaded_trials[int(index)] = (
            trial_from_json(experiment=experiment, **batch_json)
            if is_trial
            else batch_trial_from_json(experiment=experiment, **batch_json)
        )
    return loaded_trials


def data_from_json(
    data_by_trial_json: Dict[str, Any]
) -> Dict[int, "OrderedDict[int, Data]"]:
    """Load Ax Data from JSON."""
    data_by_trial = object_from_json(data_by_trial_json)
    # hack necessary because Python's json module converts dictionary
    # keys to strings: https://stackoverflow.com/q/1450957
    return {
        int(k): OrderedDict({int(k2): v2 for k2, v2 in v.items()})
        for k, v in data_by_trial.items()
    }


def simple_experiment_from_json(object_json: Dict[str, Any]) -> SimpleExperiment:
    """Load AE SimpleExperiment from JSON."""
    time_created_json = object_json.pop("time_created")
    trials_json = object_json.pop("trials")
    experiment_type_json = object_json.pop("experiment_type")
    data_by_trial_json = object_json.pop("data_by_trial")
    description_json = object_json.pop("description")
    is_test_json = object_json.pop("is_test")
    optimization_config = object_from_json(object_json.pop("optimization_config"))

    # not relevant to simple experiment
    del object_json["tracking_metrics"]
    del object_json["runner"]

    kwargs = {k: object_from_json(v) for k, v in object_json.items()}
    kwargs["evaluation_function"] = unimplemented_evaluation_function
    kwargs["objective_name"] = optimization_config.objective.metric.name
    kwargs["minimize"] = optimization_config.objective.minimize
    kwargs["outcome_constraints"] = optimization_config.outcome_constraints
    experiment = SimpleExperiment(**kwargs)

    experiment.description = object_from_json(description_json)
    experiment.is_test = object_from_json(is_test_json)
    experiment._time_created = object_from_json(time_created_json)
    experiment._trials = trials_from_json(experiment, trials_json)
    for trial in experiment._trials.values():
        for arm in trial.arms:
            experiment._arms_by_signature[arm.signature] = arm
    if experiment.status_quo is not None:
        sq_sig = experiment.status_quo.signature
        experiment._arms_by_signature[sq_sig] = experiment.status_quo
    experiment._experiment_type = object_from_json(experiment_type_json)
    experiment._data_by_trial = data_from_json(data_by_trial_json)
    return experiment


def experiment_from_json(object_json: Dict[str, Any]) -> Experiment:
    """Load Ax Experiment from JSON."""
    time_created_json = object_json.pop("time_created")
    trials_json = object_json.pop("trials")
    experiment_type_json = object_json.pop("experiment_type")
    data_by_trial_json = object_json.pop("data_by_trial")
    experiment = Experiment(**{k: object_from_json(v) for k, v in object_json.items()})
    experiment._time_created = object_from_json(time_created_json)
    experiment._trials = trials_from_json(experiment, trials_json)
    for trial in experiment._trials.values():
        for arm in trial.arms:
            experiment._arms_by_signature[arm.signature] = arm
    if experiment.status_quo is not None:
        sq_sig = experiment.status_quo.signature
        experiment._arms_by_signature[sq_sig] = experiment.status_quo
    experiment._experiment_type = object_from_json(experiment_type_json)
    experiment._data_by_trial = data_from_json(data_by_trial_json)
    return experiment
