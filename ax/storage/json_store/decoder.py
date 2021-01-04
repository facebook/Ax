#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import pickle
from collections import OrderedDict
from enum import Enum
from inspect import isclass
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import pandas as pd
import torch
from ax.benchmark.benchmark_problem import SimpleBenchmarkProblem
from ax.core.base_trial import BaseTrial
from ax.core.data import Data  # noqa F401
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
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
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import (
    Models,
    ModelRegistryBase,
    _decode_callables_from_references,
)
from ax.storage.json_store.decoders import batch_trial_from_json, trial_from_json
from ax.storage.json_store.registry import CLASS_DECODER_REGISTRY, DECODER_REGISTRY
from ax.utils.common.typeutils import not_none, torch_type_from_str
from ax.utils.measurement import synthetic_functions
from ax.utils.measurement.synthetic_functions import from_botorch
from botorch.test_functions import synthetic as botorch_synthetic


def object_from_json(object_json: Any) -> Any:
    """Recursively load objects from a JSON-serializable dictionary."""
    if type(object_json) in (str, int, float, bool, type(None)) or isinstance(
        object_json, Enum
    ):
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
        elif _type == "ndarray":
            return np.array(object_json["value"])
        elif _type == "Tensor":
            device = (
                object_from_json(object_json["device"])
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            return torch.tensor(
                object_json["value"],
                dtype=object_from_json(object_json["dtype"]),
                device=device,
            )
        elif _type.startswith("torch"):
            # Torch types will be encoded as "torch_<type_name>", so we drop prefix
            return torch_type_from_str(
                identifier=object_json["value"], type_name=_type[6:]
            )

        # Used for decoding classes (not objects).
        elif _type in CLASS_DECODER_REGISTRY:
            return CLASS_DECODER_REGISTRY[_type](object_json)

        elif _type not in DECODER_REGISTRY:
            err = (
                f"The JSON dictionary passed to `object_from_json` has a type "
                f"{_type} that is not registered with a corresponding class in "
                f"DECODER_REGISTRY."
            )
            raise JSONDecodeError(err)

        _class = DECODER_REGISTRY[_type]

        if isclass(_class) and issubclass(_class, Enum):
            # to access enum members by name, use item access
            return _class[object_json["name"]]
        elif _class == GeneratorRun:
            return generator_run_from_json(object_json=object_json)
        elif _class == GenerationStep:
            return generation_step_from_json(generation_step_json=object_json)
        elif _class == GenerationStrategy:
            return generation_strategy_from_json(generation_strategy_json=object_json)
        elif _class == MultiTypeExperiment:
            return multi_type_experiment_from_json(object_json=object_json)
        elif _class == SimpleExperiment:
            return simple_experiment_from_json(object_json=object_json)
        elif _class == Experiment:
            return experiment_from_json(object_json=object_json)
        elif _class == SearchSpace:
            return search_space_from_json(search_space_json=object_json)
        elif _class == SimpleBenchmarkProblem:
            return simple_benchmark_problem_from_json(object_json=object_json)

        return ax_class_from_json_dict(_class=_class, object_json=object_json)
    else:
        err = (
            f"The object {object_json} passed to `object_from_json` has an "
            f"unsupported type: {type(object_json)}."
        )
        raise JSONDecodeError(err)


def ax_class_from_json_dict(_class: Type, object_json: Dict[str, Any]) -> Any:
    """Reinstantiates an Ax class registered in `DECODER_REGISTRY` from a JSON
    dict.
    """
    return _class(**{k: object_from_json(v) for k, v in object_json.items()})


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


def multi_type_experiment_from_json(object_json: Dict[str, Any]) -> MultiTypeExperiment:
    """Load AE MultiTypeExperiment from JSON."""
    experiment_info = _get_experiment_info(object_json)

    _metric_to_canonical_name = object_json.pop("_metric_to_canonical_name")
    _metric_to_trial_type = object_json.pop("_metric_to_trial_type")
    _trial_type_to_runner = object_from_json(object_json.pop("_trial_type_to_runner"))
    tracking_metrics = object_from_json(object_json.pop("tracking_metrics"))
    # not relevant to multi type experiment
    del object_json["runner"]

    kwargs = {k: object_from_json(v) for k, v in object_json.items()}
    kwargs["default_runner"] = _trial_type_to_runner[object_json["default_trial_type"]]

    experiment = MultiTypeExperiment(**kwargs)
    for metric in tracking_metrics:
        experiment._tracking_metrics[metric.name] = metric
    experiment._metric_to_canonical_name = _metric_to_canonical_name
    experiment._metric_to_trial_type = _metric_to_trial_type
    experiment._trial_type_to_runner = _trial_type_to_runner

    _load_experiment_info(exp=experiment, exp_info=experiment_info)
    return experiment


def simple_experiment_from_json(object_json: Dict[str, Any]) -> SimpleExperiment:
    """Load AE SimpleExperiment from JSON."""
    experiment_info = _get_experiment_info(object_json)

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

    _load_experiment_info(exp=experiment, exp_info=experiment_info)
    return experiment


def experiment_from_json(object_json: Dict[str, Any]) -> Experiment:
    """Load Ax Experiment from JSON."""
    experiment_info = _get_experiment_info(object_json)

    experiment = Experiment(**{k: object_from_json(v) for k, v in object_json.items()})
    experiment._arms_by_name = {}

    _load_experiment_info(exp=experiment, exp_info=experiment_info)
    return experiment


def _get_experiment_info(object_json: Dict[str, Any]) -> Dict[str, Any]:
    """Returns basic information from `Experiment` object_json."""
    return {
        "time_created_json": object_json.pop("time_created"),
        "trials_json": object_json.pop("trials"),
        "experiment_type_json": object_json.pop("experiment_type"),
        "data_by_trial_json": object_json.pop("data_by_trial"),
    }


def _load_experiment_info(exp: Experiment, exp_info: Dict[str, Any]) -> None:
    """Loads `Experiment` object with basic information."""
    exp._time_created = object_from_json(exp_info.get("time_created_json"))
    exp._trials = trials_from_json(exp, exp_info.get("trials_json"))
    exp._experiment_type = object_from_json(exp_info.get("experiment_type_json"))
    exp._data_by_trial = data_from_json(exp_info.get("data_by_trial_json"))
    for trial in exp._trials.values():
        for arm in trial.arms:
            exp._register_arm(arm)
        if trial.ttl_seconds is not None:
            exp._trials_have_ttl = True
    if exp.status_quo is not None:
        sq = not_none(exp.status_quo)
        exp._register_arm(sq)


def _convert_generation_step_keys_for_backwards_compatibility(
    object_json: Dict[str, Any]
) -> Dict[str, Any]:
    """If necessary, converts keys in a JSON dict representing a `GenerationStep`
    for backwards compatibility.
    """
    # NOTE: this is a hack to make generation steps able to load after the
    # renaming of generation step fields to be in terms of 'trials' rather than
    # 'arms'.
    keys = list(object_json.keys())
    for k in keys:
        if "arms" in k:  # pragma: no cover
            object_json[k.replace("arms", "trials")] = object_json.pop(k)
        if k == "recommended_max_parallelism":  # pragma: no cover
            object_json["max_parallelism"] = object_json.pop(k)
    return object_json


def generation_step_from_json(generation_step_json: Dict[str, Any]) -> GenerationStep:
    """Load generation step from JSON."""
    generation_step_json = _convert_generation_step_keys_for_backwards_compatibility(
        generation_step_json
    )
    kwargs = generation_step_json.pop("model_kwargs", None)
    gen_kwargs = generation_step_json.pop("model_gen_kwargs", None)
    return GenerationStep(
        model=object_from_json(generation_step_json.pop("model")),
        num_trials=generation_step_json.pop("num_trials"),
        min_trials_observed=generation_step_json.pop("min_trials_observed", 0),
        max_parallelism=(generation_step_json.pop("max_parallelism", None)),
        use_update=generation_step_json.pop("use_update", False),
        enforce_num_trials=generation_step_json.pop("enforce_num_trials", True),
        model_kwargs=_decode_callables_from_references(object_from_json(kwargs))
        if kwargs
        else None,
        model_gen_kwargs=_decode_callables_from_references(object_from_json(gen_kwargs))
        if gen_kwargs
        else None,
        index=generation_step_json.pop("index", -1),
    )


def generation_strategy_from_json(
    generation_strategy_json: Dict[str, Any]
) -> GenerationStrategy:
    """Load generation strategy from JSON."""
    steps = object_from_json(generation_strategy_json.pop("steps"))
    gs = GenerationStrategy(steps=steps, name=generation_strategy_json.pop("name"))
    gs._db_id = object_from_json(generation_strategy_json.pop("db_id"))
    gs._experiment = object_from_json(generation_strategy_json.pop("experiment"))
    gs._curr = gs._steps[generation_strategy_json.pop("curr_index")]
    gs._generator_runs = object_from_json(
        generation_strategy_json.pop("generator_runs")
    )
    if generation_strategy_json.pop("had_initialized_model"):  # pragma: no cover
        # If model in the current step was not directly from the `Models` enum,
        # pass its type to `restore_model_from_generator_run`, which will then
        # attempt to use this type to recreate the model.
        if type(gs._curr.model) != Models:
            models_enum = type(gs._curr.model)
            assert issubclass(models_enum, ModelRegistryBase)
            # pyre-ignore[6]: `models_enum` typing hackiness
            gs._restore_model_from_generator_run(models_enum=models_enum)
            return gs

        gs._restore_model_from_generator_run()
    return gs


def simple_benchmark_problem_from_json(
    object_json: Dict[str, Any]
) -> SimpleBenchmarkProblem:
    """Load a benchmark problem from JSON."""
    uses_synthetic_function = object_json.pop("uses_synthetic_function")
    if uses_synthetic_function:
        function_name = object_json.pop("function_name")
        from_botorch_prefix = synthetic_functions.FromBotorch.__name__
        if function_name.startswith(from_botorch_prefix):
            botorch_function_name = function_name.replace(f"{from_botorch_prefix}_", "")
            botorch_function = getattr(botorch_synthetic, botorch_function_name)()
            f = from_botorch(botorch_function)
        else:
            f = getattr(synthetic_functions, function_name)()
    else:
        f = pickle.loads(object_json.pop("f").encode())
    domain = object_from_json(object_json.pop("domain"))
    assert isinstance(domain, list) and all(
        isinstance(x, (tuple, list)) for x in domain
    )
    return SimpleBenchmarkProblem(
        f=f,
        name=object_json.pop("name"),
        domain=cast(List[Tuple[float, float]], domain),
        minimize=object_json.pop("minimize"),
    )
