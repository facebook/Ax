#!/usr/bin/env python3

import datetime
import enum
from collections import OrderedDict
from typing import Any

import pandas as pd
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.exceptions.storage import JSONDecodeError
from ae.lazarus.ae.storage.json_store.decoders import (
    batch_trial_from_json,
    trial_from_json,
)
from ae.lazarus.ae.storage.json_store.registry import DECODER_REGISTRY


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
            return pd.read_json(object_json["value"])

        if _type not in DECODER_REGISTRY:
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

        if _class == GeneratorRun:
            time_created_json = object_json.pop("time_created")
            type_json = object_json.pop("generator_run_type")
            index_json = object_json.pop("index")
            # pyre: `typing.Type[typing.Union[Experiment, GeneratorRun, ae.
            # pyre: lazarus.core.base_trial.TrialStatus, ae.lazarus.ae.core.
            # pyre: batch_trial.AbandonedArm, ae.lazarus.ae.core.
            # pyre: batch_trial.BatchTrial, ae.lazarus.ae.core.batch_trial.
            # pyre: GeneratorRunStruct, ae.lazarus.ae.core.arm.Arm,
            # pyre: ae.lazarus.ae.core.metric.Metric, ae.lazarus.ae.core.objective.
            # pyre: Objective, ae.lazarus.ae.core.optimization_config.
            # pyre: OptimizationConfig, ae.lazarus.ae.core.outcome_constraint.
            # pyre: OutcomeConstraint, ae.lazarus.ae.core.parameter.
            # pyre: ChoiceParameter, ae.lazarus.ae.core.parameter.
            # pyre: FixedParameter, ae.lazarus.ae.core.parameter.ParameterType,
            # pyre: ae.lazarus.ae.core.parameter.RangeParameter, ae.lazarus.ae.core.
            # pyre: parameter_constraint.ParameterConstraint, ae.lazarus.ae.core.
            # pyre: search_space.SearchSpace, ae.lazarus.ae.core.trial.Trial, ae.
            # pyre: lazarus.core.types.types.ComparisonOp, ae.lazarus.ae.runners.
            # pyre: synthetic.SyntheticRunner, ae.lazarus.ae.storage.utils.
            # pyre: DomainType, ae.lazarus.ae.storage.utils.
            # pyre-fixme[29]: ParameterConstraintType]]` is not a function.
            generator_run = _class(
                **{k: object_from_json(v) for k, v in object_json.items()}
            )
            generator_run._time_created = object_from_json(time_created_json)
            generator_run._generator_run_type = object_from_json(type_json)
            generator_run._index = object_from_json(index_json)
            return generator_run
        if _class == Experiment:
            time_created_json = object_json.pop("time_created")
            trials_json = object_json.pop("trials")
            experiment_type_json = object_json.pop("experiment_type")
            data_by_trial_json = object_json.pop("data_by_trial")
            # pyre: `typing.Type[typing.Union[Experiment, GeneratorRun, ae.
            # pyre: lazarus.core.base_trial.TrialStatus, ae.lazarus.ae.core.
            # pyre: batch_trial.AbandonedArm, ae.lazarus.ae.core.
            # pyre: batch_trial.BatchTrial, ae.lazarus.ae.core.batch_trial.
            # pyre: GeneratorRunStruct, ae.lazarus.ae.core.arm.Arm,
            # pyre: ae.lazarus.ae.core.metric.Metric, ae.lazarus.ae.core.objective.
            # pyre: Objective, ae.lazarus.ae.core.optimization_config.
            # pyre: OptimizationConfig, ae.lazarus.ae.core.outcome_constraint.
            # pyre: OutcomeConstraint, ae.lazarus.ae.core.parameter.
            # pyre: ChoiceParameter, ae.lazarus.ae.core.parameter.
            # pyre: FixedParameter, ae.lazarus.ae.core.parameter.ParameterType,
            # pyre: ae.lazarus.ae.core.parameter.RangeParameter, ae.lazarus.ae.core.
            # pyre: parameter_constraint.ParameterConstraint, ae.lazarus.ae.core.
            # pyre: search_space.SearchSpace, ae.lazarus.ae.core.trial.Trial, ae.
            # pyre: lazarus.core.types.types.ComparisonOp, ae.lazarus.ae.runners.
            # pyre: synthetic.SyntheticRunner, ae.lazarus.ae.storage.utils.
            # pyre: DomainType, ae.lazarus.ae.storage.utils.
            # pyre-fixme[29]: ParameterConstraintType]]` is not a function.
            experiment = _class(
                **{k: object_from_json(v) for k, v in object_json.items()}
            )
            loaded_trials = {}
            for index, batch_json in trials_json.items():
                is_trial = batch_json["__type"] == "Trial"
                batch_json = {
                    k: object_from_json(v)
                    for k, v in batch_json.items()
                    if k != "__type"
                }
                # TODO[drfreund]: if trial, decode as trial here
                loaded_trials[int(index)] = (
                    trial_from_json(experiment=experiment, **batch_json)
                    if is_trial
                    else batch_trial_from_json(experiment=experiment, **batch_json)
                )

            data_by_trial = object_from_json(data_by_trial_json)
            # hack necessary because Python's json module converts dictionary
            # keys to strings: https://stackoverflow.com/q/1450957
            data_by_trial = {
                int(k): OrderedDict({int(k2): v2 for k2, v2 in v.items()})
                for k, v in data_by_trial.items()
            }

            experiment._time_created = object_from_json(time_created_json)
            experiment._trials = loaded_trials
            experiment._experiment_type = object_from_json(experiment_type_json)
            experiment._data_by_trial = data_by_trial
            return experiment

        # pyre: `typing.Type[typing.Union[Experiment, GeneratorRun, ae.
        # pyre: lazarus.core.base_trial.TrialStatus, ae.lazarus.ae.core.
        # pyre: batch_trial.AbandonedArm, ae.lazarus.ae.core.batch_trial.
        # pyre: BatchTrial, ae.lazarus.ae.core.batch_trial.GeneratorRunStruct,
        # pyre: ae.lazarus.ae.core.arm.Arm, ae.lazarus.ae.core.metric.
        # pyre: Metric, ae.lazarus.ae.core.objective.Objective, ae.lazarus.ae.core.
        # pyre: optimization_config.OptimizationConfig, ae.lazarus.ae.core.
        # pyre: outcome_constraint.OutcomeConstraint, ae.lazarus.ae.core.
        # pyre: parameter.ChoiceParameter, ae.lazarus.ae.core.parameter.
        # pyre: FixedParameter, ae.lazarus.ae.core.parameter.ParameterType, ae.
        # pyre: lazarus.core.parameter.RangeParameter, ae.lazarus.ae.core.
        # pyre: parameter_constraint.ParameterConstraint, ae.lazarus.ae.core.
        # pyre: search_space.SearchSpace, ae.lazarus.ae.core.trial.Trial, ae.
        # pyre: lazarus.core.types.types.ComparisonOp, ae.lazarus.ae.runners.
        # pyre: synthetic.SyntheticRunner, ae.lazarus.ae.storage.utils.
        # pyre: DomainType, ae.lazarus.ae.storage.utils.
        # pyre-fixme[29]: ParameterConstraintType]]` is not a function.
        return _class(**{k: object_from_json(v) for k, v in object_json.items()})
    else:
        err = (
            f"The object passed to `object_from_json` has an unsupported type: "
            f"{type(object_json)}."
        )
        raise JSONDecodeError(err)
