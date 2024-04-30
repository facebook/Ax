#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import datetime
from collections import OrderedDict
from enum import Enum
from inspect import isclass
from io import StringIO
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from ax.benchmark.problems.hpo.torchvision import (
    PyTorchCNNTorchvisionBenchmarkProblem as TorchvisionBenchmarkProblem,
)
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import Objective
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.storage import JSONDecodeError
from ax.modelbridge.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import _decode_callables_from_references
from ax.modelbridge.transition_criterion import TransitionCriterion, TrialBasedCriterion
from ax.models.torch.botorch_modular.model import SurrogateSpec
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.storage.json_store.decoders import (
    batch_trial_from_json,
    botorch_component_from_json,
    tensor_from_json,
    trial_from_json,
)
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_DECODER_REGISTRY,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import (
    SerializationMixin,
    TClassDecoderRegistry,
    TDecoderRegistry,
)
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.common.typeutils_torch import torch_type_from_str


logger: Logger = get_logger(__name__)


# pyre-fixme[3]: Return annotation cannot be `Any`.
def object_from_json(
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    object_json: Any,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Any:
    """Recursively load objects from a JSON-serializable dictionary."""
    if type(object_json) in (str, int, float, bool, type(None)) or isinstance(
        object_json, Enum
    ):
        return object_json
    elif isinstance(object_json, list):
        return [
            object_from_json(
                i,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for i in object_json
        ]
    elif isinstance(object_json, tuple):
        return tuple(
            object_from_json(
                i,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for i in object_json
        )
    elif isinstance(object_json, dict):
        if "__type" not in object_json:
            # this is just a regular dictionary, e.g. the one in Parameter
            # containing parameterizations
            return {
                k: object_from_json(
                    v,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
                for k, v in object_json.items()
            }

        _type = object_json.pop("__type")

        if _type == "datetime":
            return datetime.datetime.strptime(
                object_json["value"], "%Y-%m-%d %H:%M:%S.%f"
            )
        elif _type == "OrderedDict":
            return OrderedDict(
                [
                    (
                        k,
                        object_from_json(
                            v,
                            decoder_registry=decoder_registry,
                            class_decoder_registry=class_decoder_registry,
                        ),
                    )
                    for k, v in object_json["value"]
                ]
            )
        elif _type == "DataFrame":
            # Need dtype=False, otherwise infers arm_names like "4_1"
            # should be int 41
            return pd.read_json(StringIO(object_json["value"]), dtype=False)
        elif _type == "ndarray":
            return np.array(object_json["value"])
        elif _type == "Tensor":
            return tensor_from_json(json=object_json)
        elif _type.startswith("torch"):
            # Torch types will be encoded as "torch_<type_name>", so we drop prefix
            return torch_type_from_str(
                identifier=object_json["value"], type_name=_type[6:]
            )

        elif _type == "ListSurrogate":
            return surrogate_from_list_surrogate_json(
                list_surrogate_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _type == "set":
            return set(object_json["value"])
        # Used for decoding classes (not objects).
        elif _type in class_decoder_registry:
            return class_decoder_registry[_type](object_json)

        elif _type not in decoder_registry:
            err = (
                f"The JSON dictionary passed to `object_from_json` has a type "
                f"{_type} that is not registered with a corresponding class in "
                f"DECODER_REGISTRY."
            )
            raise JSONDecodeError(err)

        _class = decoder_registry[_type]

        if isclass(_class) and issubclass(_class, Enum):
            # to access enum members by name, use item access
            return _class[object_json["name"]]
        elif isclass(_class) and issubclass(_class, torch.nn.Module):
            return botorch_component_from_json(botorch_class=_class, json=object_json)
        elif _class == GeneratorRun:
            return generator_run_from_json(
                object_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == GenerationStep:
            return generation_step_from_json(
                generation_step_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == GenerationNode:
            return generation_node_from_json(
                generation_node_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == ModelSpec:
            return model_spec_from_json(
                model_spec_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == GenerationStrategy:
            return generation_strategy_from_json(
                generation_strategy_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == MultiTypeExperiment:
            return multi_type_experiment_from_json(
                object_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == Experiment:
            return experiment_from_json(
                object_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == SearchSpace:
            return search_space_from_json(
                search_space_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == Objective:
            return objective_from_json(
                object_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif _class == TorchvisionBenchmarkProblem:
            return TorchvisionBenchmarkProblem.from_dataset_name(
                name=object_json["name"],
                num_trials=object_json["num_trials"],
            )
        elif _class in (SurrogateSpec, Surrogate):
            if "input_transform" in object_json:
                (
                    input_transform_classes_json,
                    input_transform_options_json,
                ) = get_input_transform_json_components(
                    input_transforms_json=object_json.pop("input_transform"),
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
                object_json["input_transform_classes"] = input_transform_classes_json
                object_json["input_transform_options"] = input_transform_options_json
            if "outcome_transform" in object_json:
                (
                    outcome_transform_classes_json,
                    outcome_transform_options_json,
                ) = get_outcome_transform_json_components(
                    outcome_transforms_json=object_json.pop("outcome_transform"),
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
                object_json["outcome_transform_classes"] = (
                    outcome_transform_classes_json
                )
                object_json["outcome_transform_options"] = (
                    outcome_transform_options_json
                )
        elif isclass(_class) and issubclass(_class, TrialBasedCriterion):
            # TrialBasedCriterion contain a list of `TrialStatus` for args.
            # This list needs to be unpacked by hand to properly retain the types.
            return trial_transition_criteria_from_json(
                class_=_class,
                transition_criteria_json=object_json,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
        elif isclass(_class) and issubclass(_class, SerializationMixin):
            return _class(
                **_class.deserialize_init_args(
                    args=object_json,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
            )

        return ax_class_from_json_dict(
            _class=_class,
            object_json=object_json,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
    else:
        err = (
            f"The object {object_json} passed to `object_from_json` has an "
            f"unsupported type: {type(object_json)}."
        )
        raise JSONDecodeError(err)


# pyre-fixme[3]: Return annotation cannot be `Any`.
def ax_class_from_json_dict(
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    _class: Type,
    object_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Any:
    """Reinstantiates an Ax class registered in `DECODER_REGISTRY` from a JSON
    dict.
    """
    return _class(
        **{
            k: object_from_json(
                v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in object_json.items()
        }
    )


def generator_run_from_json(
    object_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GeneratorRun:
    """Load Ax GeneratorRun from JSON."""
    time_created_json = object_json.pop("time_created")
    type_json = object_json.pop("generator_run_type")
    index_json = object_json.pop("index")
    generator_run = GeneratorRun(
        **{
            k: object_from_json(
                v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in object_json.items()
        }
    )
    generator_run._time_created = object_from_json(
        time_created_json,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    generator_run._generator_run_type = object_from_json(
        type_json,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    generator_run._index = object_from_json(
        index_json,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    return generator_run


def trial_transition_criteria_from_json(
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
    #  avoid runtime subscripting errors.
    class_: Type,
    transition_criteria_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Optional[TransitionCriterion]:
    """Load Ax transition criteria that depend on Trials from JSON.

    Since ```TrialBasedCriterion``` contain lists of ```TrialStatus``,
    the json for these criterion needs to be carefully unpacked and
    re-processed via ```object_from_json``` in order to maintain correct
    typing. We pass in ```class_``` in order to correctly handle all classes
    which inherit from ```TrialBasedCriterion``` (ex: ```MaxTrials```).
    """
    new_dict = {}
    for key, value in transition_criteria_json.items():
        new_val = object_from_json(
            object_json=value,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        new_dict[key] = new_val

    return class_(**new_dict)


def search_space_from_json(
    search_space_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> SearchSpace:
    """Load a SearchSpace from JSON.

    This function is necessary due to the coupled loading of SearchSpace
    and parameter constraints.
    """
    parameters = object_from_json(
        search_space_json.pop("parameters"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    json_param_constraints = search_space_json.pop("parameter_constraints")
    return SearchSpace(
        parameters=parameters,
        parameter_constraints=parameter_constraints_from_json(
            parameter_constraint_json=json_param_constraints,
            parameters=parameters,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
    )


def parameter_constraints_from_json(
    parameter_constraint_json: List[Dict[str, Any]],
    parameters: List[Parameter],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
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
            parameter_constraints.append(
                object_from_json(
                    constraint,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
            )
    return parameter_constraints


def trials_from_json(
    experiment: Experiment,
    trials_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Dict[int, BaseTrial]:
    """Load Ax Trials from JSON."""
    loaded_trials = {}
    for index, batch_json in trials_json.items():
        is_trial = batch_json["__type"] == "Trial"
        batch_json = {
            k: object_from_json(
                v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in batch_json.items()
            if k != "__type"
        }
        loaded_trials[int(index)] = (
            trial_from_json(experiment=experiment, **batch_json)
            if is_trial
            else batch_trial_from_json(experiment=experiment, **batch_json)
        )
    return loaded_trials


def data_from_json(
    data_by_trial_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Dict[int, "OrderedDict[int, Data]"]:
    """Load Ax Data from JSON."""
    data_by_trial = object_from_json(
        data_by_trial_json,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    # hack necessary because Python's json module converts dictionary
    # keys to strings: https://stackoverflow.com/q/1450957
    return {
        int(k): OrderedDict({int(k2): v2 for k2, v2 in v.items()})
        for k, v in data_by_trial.items()
    }


def multi_type_experiment_from_json(
    object_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> MultiTypeExperiment:
    """Load AE MultiTypeExperiment from JSON."""
    experiment_info = _get_experiment_info(object_json)

    _metric_to_canonical_name = object_json.pop("_metric_to_canonical_name")
    _metric_to_trial_type = object_json.pop("_metric_to_trial_type")
    _trial_type_to_runner = object_from_json(
        object_json.pop("_trial_type_to_runner"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    tracking_metrics = object_from_json(
        object_json.pop("tracking_metrics"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    # not relevant to multi type experiment
    del object_json["runner"]

    kwargs = {
        k: object_from_json(
            v,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        for k, v in object_json.items()
    }
    kwargs["default_runner"] = _trial_type_to_runner[object_json["default_trial_type"]]

    experiment = MultiTypeExperiment(**kwargs)
    for metric in tracking_metrics:
        experiment._tracking_metrics[metric.name] = metric
    experiment._metric_to_canonical_name = _metric_to_canonical_name
    experiment._metric_to_trial_type = _metric_to_trial_type
    experiment._trial_type_to_runner = _trial_type_to_runner

    _load_experiment_info(
        exp=experiment,
        exp_info=experiment_info,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    return experiment


def experiment_from_json(
    object_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Experiment:
    """Load Ax Experiment from JSON."""
    experiment_info = _get_experiment_info(object_json)

    experiment = Experiment(
        **{
            k: object_from_json(
                v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in object_json.items()
        }
    )
    experiment._arms_by_name = {}

    _load_experiment_info(
        exp=experiment,
        exp_info=experiment_info,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    return experiment


def _get_experiment_info(object_json: Dict[str, Any]) -> Dict[str, Any]:
    """Returns basic information from `Experiment` object_json."""
    return {
        "time_created_json": object_json.pop("time_created"),
        "trials_json": object_json.pop("trials"),
        "experiment_type_json": object_json.pop("experiment_type"),
        "data_by_trial_json": object_json.pop("data_by_trial"),
    }


def _load_experiment_info(
    exp: Experiment,
    exp_info: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> None:
    """Loads `Experiment` object with basic information."""
    exp._time_created = object_from_json(
        exp_info.get("time_created_json"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    exp._trials = trials_from_json(
        exp,
        exp_info.get("trials_json"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    exp._experiment_type = object_from_json(
        exp_info.get("experiment_type_json"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    exp._data_by_trial = data_from_json(
        exp_info.get("data_by_trial_json"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
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
        if "arms" in k:
            object_json[k.replace("arms", "trials")] = object_json.pop(k)
        if k == "recommended_max_parallelism":
            object_json["max_parallelism"] = object_json.pop(k)
    return object_json


def generation_node_from_json(
    generation_node_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GenerationNode:
    """Load GenerationNode object from JSON."""
    return GenerationNode(
        node_name=generation_node_json.pop("node_name"),
        model_specs=object_from_json(
            generation_node_json.pop("model_specs"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        # TODO @mgarrad this should probably be a object_from_json but bestmodelselector
        # isn't implemented
        best_model_selector=generation_node_json.pop("best_model_selector", None),
        should_deduplicate=generation_node_json.pop("should_deduplicate", False),
        transition_criteria=(
            object_from_json(
                generation_node_json.pop("transition_criteria"),
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            if "transition_criteria" in generation_node_json.keys()
            else None
        ),
    )


def generation_step_from_json(
    generation_step_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GenerationStep:
    """Load generation step from JSON."""
    generation_step_json = _convert_generation_step_keys_for_backwards_compatibility(
        generation_step_json
    )
    kwargs = generation_step_json.pop("model_kwargs", None)
    kwargs.pop("fit_on_update", None)  # Remove deprecated fit_on_update.
    gen_kwargs = generation_step_json.pop("model_gen_kwargs", None)
    completion_criteria = (
        object_from_json(
            generation_step_json.pop("completion_criteria"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        if "completion_criteria" in generation_step_json.keys()
        else []
    )
    generation_step = GenerationStep(
        model=object_from_json(
            generation_step_json.pop("model"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        num_trials=generation_step_json.pop("num_trials"),
        min_trials_observed=generation_step_json.pop("min_trials_observed", 0),
        completion_criteria=(
            completion_criteria if completion_criteria is not None else []
        ),
        max_parallelism=(generation_step_json.pop("max_parallelism", None)),
        enforce_num_trials=generation_step_json.pop("enforce_num_trials", True),
        model_kwargs=(
            _decode_callables_from_references(
                object_from_json(
                    kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if kwargs
            else None
        ),
        model_gen_kwargs=(
            _decode_callables_from_references(
                object_from_json(
                    gen_kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if gen_kwargs
            else None
        ),
        index=generation_step_json.pop("index", -1),
        should_deduplicate=generation_step_json.pop("should_deduplicate", False),
    )
    return generation_step


def model_spec_from_json(
    model_spec_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> ModelSpec:
    """Load ModelSpec from JSON."""
    kwargs = model_spec_json.pop("model_kwargs", None)
    kwargs.pop("fit_on_update", None)  # Remove deprecated fit_on_update.
    gen_kwargs = model_spec_json.pop("model_gen_kwargs", None)
    return ModelSpec(
        model_enum=object_from_json(
            model_spec_json.pop("model_enum"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        model_kwargs=(
            _decode_callables_from_references(
                object_from_json(
                    kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if kwargs
            else None
        ),
        model_gen_kwargs=(
            _decode_callables_from_references(
                object_from_json(
                    gen_kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if gen_kwargs
            else None
        ),
    )


def generation_strategy_from_json(
    generation_strategy_json: Dict[str, Any],
    experiment: Optional[Experiment] = None,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GenerationStrategy:
    """Load generation strategy from JSON."""
    nodes = (
        object_from_json(
            generation_strategy_json.pop("nodes"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        if "nodes" in generation_strategy_json
        else []
    )

    steps = object_from_json(
        generation_strategy_json.pop("steps"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    if len(steps) > 0:
        gs = GenerationStrategy(steps=steps, name=generation_strategy_json.pop("name"))
        gs._curr = gs._steps[generation_strategy_json.pop("curr_index")]
    else:
        gs = GenerationStrategy(nodes=nodes, name=generation_strategy_json.pop("name"))
        curr_node_name = generation_strategy_json.pop("curr_node_name")
        for node in gs._nodes:
            if node.node_name == curr_node_name:
                gs._curr = node
                break

    gs._db_id = object_from_json(
        generation_strategy_json.pop("db_id"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    gs._experiment = experiment or object_from_json(
        generation_strategy_json.pop("experiment"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    gs._generator_runs = object_from_json(
        generation_strategy_json.pop("generator_runs"),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    return gs


def surrogate_from_list_surrogate_json(
    list_surrogate_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Surrogate:
    logger.warning(
        "`ListSurrogate` has been deprecated. Reconstructing a `Surrogate` "
        "with as similar properties as possible."
    )
    if "submodel_input_transforms" in list_surrogate_json:
        (
            list_surrogate_json["submodel_input_transform_classes"],
            list_surrogate_json["submodel_input_transform_options"],
        ) = get_input_transform_json_components(
            list_surrogate_json.pop("submodel_input_transforms"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
    if "submodel_outcome_transforms" in list_surrogate_json:
        (
            list_surrogate_json["submodel_outcome_transform_classes"],
            list_surrogate_json["submodel_outcome_transform_options"],
        ) = get_outcome_transform_json_components(
            list_surrogate_json.pop("submodel_outcome_transforms"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
    return Surrogate(
        botorch_model_class=object_from_json(
            object_json=list_surrogate_json.get("botorch_submodel_class"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        model_options=list_surrogate_json.get("submodel_options"),
        mll_class=object_from_json(
            object_json=list_surrogate_json.get("mll_class"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        mll_options=list_surrogate_json.get("mll_options"),
        input_transform_classes=object_from_json(
            object_json=list_surrogate_json.get("submodel_input_transform_classes"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        input_transform_options=object_from_json(
            object_json=list_surrogate_json.get("submodel_input_transform_options"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        outcome_transform_classes=object_from_json(
            object_json=list_surrogate_json.get("submodel_outcome_transform_classes"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        outcome_transform_options=object_from_json(
            object_json=list_surrogate_json.get("submodel_outcome_transform_options"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        covar_module_class=object_from_json(
            object_json=list_surrogate_json.get("submodel_covar_module_class"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        covar_module_options=list_surrogate_json.get("submodel_covar_module_options"),
        likelihood_class=object_from_json(
            object_json=list_surrogate_json.get("submodel_likelihood_class"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        likelihood_options=list_surrogate_json.get("submodel_likelihood_options"),
    )


def get_input_transform_json_components(
    input_transforms_json: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    if input_transforms_json is None:
        return None, None
    if isinstance(input_transforms_json, dict):
        # This is a single input transform.
        input_transforms_json = [input_transforms_json]
    else:
        input_transforms_json = [
            input_transform_json
            for input_transform_json in input_transforms_json
            if input_transform_json is not None
        ]
    input_transform_classes_json = [
        input_transform_json["index"] for input_transform_json in input_transforms_json
    ]
    input_transform_options_json = {
        checked_cast(str, input_transform_json["__type"]): input_transform_json[
            "state_dict"
        ]
        for input_transform_json in input_transforms_json
    }
    return input_transform_classes_json, input_transform_options_json


def get_outcome_transform_json_components(
    outcome_transforms_json: Optional[List[Dict[str, Any]]],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    if outcome_transforms_json is None:
        return None, None

    outcome_transforms_json = [
        outcome_transform_json
        for outcome_transform_json in outcome_transforms_json
        if outcome_transform_json is not None
    ]
    outcome_transform_classes_json = [
        outcome_transform_json["index"]
        for outcome_transform_json in outcome_transforms_json
    ]
    outcome_transform_options_json = {
        checked_cast(str, outcome_transform_json["__type"]): outcome_transform_json[
            "state_dict"
        ]
        for outcome_transform_json in outcome_transforms_json
    }
    return outcome_transform_classes_json, outcome_transform_options_json


def objective_from_json(
    object_json: Dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Objective:
    """Load an ``Objective`` from JSON in a backwards compatible way.

    If both ``minimize`` and ``lower_is_better`` are specified but have conflicting
    values, this will overwrite ``lower_is_better=minimize`` to resolve the conflict.

    # TODO: Do we need to do this for scalarized objective as well?
    """
    input_args = {
        k: object_from_json(
            v,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        for k, v in object_json.items()
    }
    metric = input_args.pop("metric")
    minimize = input_args.pop("minimize")
    if metric.lower_is_better is not None and metric.lower_is_better != minimize:
        logger.warning(
            f"Metric {metric.name} has {metric.lower_is_better=} but objective "
            f"specifies {minimize=}. Overwriting ``lower_is_better`` to match "
            f"the optimization direction {minimize=}."
        )
        metric.lower_is_better = minimize
    return Objective(
        metric=metric,
        minimize=minimize,
        **input_args,  # For future compatibility.
    )
