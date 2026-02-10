#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import datetime
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import isclass
from io import StringIO
from logging import Logger
from typing import Any

import numpy as np
import pandas as pd
import torch
from ax.adapter.registry import GeneratorRegistryBase
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.storage import JSON_STORAGE_DOCS_SUFFIX, JSONDecodeError
from ax.generation_strategy.generation_node_input_constructors import (
    InputConstructorPurpose,
)
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials, TransitionCriterion
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.storage.json_store.decoders import (
    _cast_parameter_value,
    batch_trial_from_json,
    botorch_component_from_json,
    tensor_from_json,
    trial_from_json,
)
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_DECODER_REGISTRY,
)
from ax.storage.utils import data_by_trial_to_data
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import (
    extract_init_args,
    SerializationMixin,
    TClassDecoderRegistry,
    TDecoderRegistry,
)
from ax.utils.common.typeutils_torch import torch_type_from_str
from botorch.utils.types import DEFAULT
from pyre_extensions import assert_is_instance, none_throws


logger: Logger = get_logger(__name__)


def _cast_arm_parameters(arm: Arm, search_space: SearchSpace) -> None:
    """Cast arm parameter values to the appropriate Python type.

    This is necessary because JSON may deserialize values as different types
    (e.g., ints as floats). This function modifies the arm in place.

    Args:
        arm: The arm whose parameter values should be cast.
        search_space: The search space containing parameter type information.
    """
    for param_name, param_value in arm._parameters.items():
        if param_name in search_space.parameters:
            parameter = search_space.parameters[param_name]
            arm._parameters[param_name] = _cast_parameter_value(
                param_value, parameter.parameter_type
            )


def _raise_on_legacy_callable_refs(kwarg_dict: dict[str, Any]) -> dict[str, Any]:
    """Returns kwarg_dict unchanged if no legacy callable refs are present.

    Raises:
        JSONDecodeError: If any value is a legacy encoded callable reference.
    """
    for k, v in kwarg_dict.items():
        if isinstance(v, dict) and v.get("is_callable_as_path", False):
            raise JSONDecodeError(
                f"Legacy callable reference '{k}' cannot be decoded. "
                "Callable serialization is not supported."
            )
    return kwarg_dict


# Deprecated generators registry entries and their replacements.
# Used below in `_update_deprecated_model_registry`.
_DEPRECATED_GENERATOR_TO_REPLACEMENT: dict[str, str] = {
    "GPEI": "BOTORCH_MODULAR",
    "MOO": "BOTORCH_MODULAR",
    "FULLYBAYESIAN": "SAASBO",
    "FULLYBAYESIANMOO": "SAASBO",
    "FULLYBAYESIAN_MTGP": "SAAS_MTGP",
    "FULLYBAYESIANMOO_MTGP": "SAAS_MTGP",
    "ST_MTGP_LEGACY": "ST_MTGP",
    "ST_MTGP_NEHVI": "ST_MTGP",
    "CONTEXT_SACBO": "BOTORCH_MODULAR",
    "LEGACY_BOTORCH": "BOTORCH_MODULAR",
}

# Deprecated generator kwargs, to be removed from GStep / GNodes.
_DEPRECATED_GENERATOR_KWARGS: tuple[str, ...] = (
    "fit_on_update",
    "fit_out_of_design",
    "fit_abandoned",
    "fit_only_completed_map_metrics",
    "torch_dtype",
    "status_quo_name",
    "status_quo_features",
)

# Deprecated node input constructors, removed from GNodes.
# NOTE: These are the enum keys, which are typically upper-case.
_DEPRECATED_NODE_INPUT_CONSTRUCTORS: tuple[str, ...] = ("STATUS_QUO_FEATURES",)


@dataclass
class RegistryKwargs:
    decoder_registry: TDecoderRegistry
    class_decoder_registry: TClassDecoderRegistry


def object_from_json(
    object_json: Any,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Any:
    """Recursively load objects from a JSON-serializable dictionary."""

    registry_kwargs = RegistryKwargs(
        decoder_registry=decoder_registry, class_decoder_registry=class_decoder_registry
    )

    _object_from_json = partial(object_from_json, **vars(registry_kwargs))

    if type(object_json) in (str, int, float, bool, type(None)) or isinstance(
        object_json, Enum
    ):
        return object_json
    elif isinstance(object_json, list):
        return [_object_from_json(i) for i in object_json]
    elif isinstance(object_json, tuple):
        return tuple(_object_from_json(i) for i in object_json)
    elif isinstance(object_json, dict):
        if "__type" not in object_json:
            # this is just a regular dictionary, e.g. the one in Parameter
            # containing parameterizations
            result = {}
            for k, v in object_json.items():
                # Convert "null" string back to None for dictionary keys
                # This handles the case where _trial_type_to_runner has None keys
                # that get serialized to "null" strings in JSON
                key = None if k == "null" else k
                result[key] = _object_from_json(v)
            return result

        _type = object_json.pop("__type")

        if _type == "datetime":
            return datetime.datetime.strptime(
                object_json["value"], "%Y-%m-%d %H:%M:%S.%f"
            )
        elif _type == "OrderedDict":
            return OrderedDict(
                [(k, _object_from_json(v)) for k, v in object_json["value"]]
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
                list_surrogate_json=object_json, **vars(registry_kwargs)
            )
        elif _type == "set":
            return set(object_json["value"])
        # Used for decoding classes (not objects).
        elif _type in class_decoder_registry:
            return class_decoder_registry[_type](object_json)

        elif _type == "GeneratorRunStruct":
            object_json.pop("weight", None)  # Deprecated.
            gr_json = object_json["generator_run"]
            assert gr_json.pop("__type") == "GeneratorRun"
            return generator_run_from_json(object_json=gr_json, **vars(registry_kwargs))

        elif _type not in decoder_registry:
            err = (
                f"The JSON dictionary passed to `object_from_json` has a type "
                f"{_type} that is not registered with a corresponding class in "
                f"DECODER_REGISTRY. {JSON_STORAGE_DOCS_SUFFIX}"
            )
            raise JSONDecodeError(err)

        # pyre-fixme[9, 24]: Generic type `type` expects 1 type parameter, use
        # `typing.Type[<base type>]` to avoid runtime subscripting errors.
        _class: type = decoder_registry[_type]
        if isclass(_class) and issubclass(_class, Enum):
            name = object_json["name"]
            if issubclass(_class, GeneratorRegistryBase):
                name = _update_deprecated_model_registry(name=name)
            # to access enum members by name, use item access
            return _class[name]
        elif isclass(_class) and issubclass(_class, torch.nn.Module):
            return botorch_component_from_json(botorch_class=_class, json=object_json)
        elif _class == GeneratorRun:
            return generator_run_from_json(
                object_json=object_json, **vars(registry_kwargs)
            )
        # Backward compatibility. `GenerationStep`-s are now just encoded as
        # `GenerationNode`-s, but we still need to support loading old GSteps.
        elif _class == GenerationStep:
            return generation_step_from_json(
                generation_step_json=object_json, **vars(registry_kwargs)
            )
        elif _class == GenerationNode:
            return generation_node_from_json(
                generation_node_json=object_json, **vars(registry_kwargs)
            )
        elif _class == GeneratorSpec:
            return generator_spec_from_json(
                generator_spec_json=object_json, **vars(registry_kwargs)
            )
        elif _class == GenerationStrategy:
            return generation_strategy_from_json(
                generation_strategy_json=object_json, **vars(registry_kwargs)
            )
        elif _class == MultiTypeExperiment:
            return multi_type_experiment_from_json(
                object_json=object_json, **vars(registry_kwargs)
            )
        elif _class == Experiment:
            return experiment_from_json(
                object_json=object_json, **vars(registry_kwargs)
            )
        elif _class == SearchSpace:
            return search_space_from_json(
                search_space_json=object_json, **vars(registry_kwargs)
            )
        elif _class == Objective:
            return objective_from_json(object_json=object_json, **vars(registry_kwargs))
        elif _class in (SurrogateSpec, Surrogate, ModelConfig):
            if "input_transform" in object_json:
                (
                    input_transform_classes_json,
                    input_transform_options_json,
                ) = get_input_transform_json_components(
                    input_transforms_json=object_json.pop("input_transform"),
                    **vars(registry_kwargs),
                )
                object_json["input_transform_classes"] = input_transform_classes_json
                object_json["input_transform_options"] = input_transform_options_json
            if "outcome_transform" in object_json:
                (
                    outcome_transform_classes_json,
                    outcome_transform_options_json,
                ) = get_outcome_transform_json_components(
                    outcome_transforms_json=object_json.pop("outcome_transform"),
                    **vars(registry_kwargs),
                )
                object_json["outcome_transform_classes"] = (
                    outcome_transform_classes_json
                )
                object_json["outcome_transform_options"] = (
                    outcome_transform_options_json
                )
        elif (
            isclass(_class)
            and issubclass(_class, TransitionCriterion)
            and _class is not TransitionCriterion  # TransitionCriterion is abstract
        ):
            # TransitionCriterion may contain nested Ax objects (TrialStatus, etc.)
            # that need recursive deserialization via object_from_json.
            return transition_criterion_from_json(
                transition_criterion_class=_class,
                object_json=object_json,
                **vars(registry_kwargs),
            )
        elif isclass(_class) and issubclass(_class, SerializationMixin):
            # Special handling for Data backward compatibility
            if _class is Data:
                data_json_str = object_json.get("df", {}).get("value", "")
                data_json = json.loads(data_json_str)
                if data_json and "metric_signature" not in data_json:
                    object_json["df"]["value"] = (
                        _update_data_json_with_metric_signature(
                            object_json["df"]["value"]
                        )
                    )

            return _class(
                # Note: we do not recursively call object_from_json here again as
                # that would invalidate design principles behind deserialize_init_args.
                # Any Ax class that needs serialization and who's init args include
                # another Ax class that needs serialization should implement its own
                # _to_json and _from_json methods and register them appropriately.
                **_class.deserialize_init_args(
                    args=object_json, **vars(registry_kwargs)
                )
            )
        if _class in (BoTorchGenerator, Surrogate):
            # Updates deprecated surrogate spec related inputs.
            object_json = _sanitize_surrogate_spec_input(object_json=object_json)
        if _class is Surrogate:
            object_json = _sanitize_legacy_surrogate_inputs(object_json=object_json)
        if _class is SurrogateSpec:
            object_json = _sanitize_inputs_to_surrogate_spec(object_json=object_json)
        if isclass(_class) and issubclass(_class, OptimizationConfig):
            object_json.pop("risk_measure", None)  # Deprecated.
        return ax_class_from_json_dict(
            _class=_class, object_json=object_json, **vars(registry_kwargs)
        )
    else:
        err = (
            f"The object {object_json} passed to `object_from_json` has an "
            f"unsupported type: {type(object_json)}."
        )
        raise JSONDecodeError(err)


def ax_class_from_json_dict(
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    _class: type,
    object_json: dict[str, Any],
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
    object_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GeneratorRun:
    """Load Ax GeneratorRun from JSON."""
    time_created_json = object_json.pop("time_created")
    type_json = object_json.pop("generator_run_type")
    object_json.pop("index", None)  # Deprecated.
    object_json.pop("generation_step_index", None)  # Deprecated.
    # Remove `objective_thresholds` to avoid issues with registries, since
    # `ObjectiveThreshold` depend on `Metric` objects.
    object_json.pop("objective_thresholds", None)

    # Backwards compatibility: handle old field names
    if "model_key" in object_json:
        object_json["generator_key"] = object_json.pop("model_key")
    if "model_kwargs" in object_json:
        object_json["generator_kwargs"] = object_json.pop("model_kwargs")
    if "bridge_kwargs" in object_json:
        object_json["adapter_kwargs"] = object_json.pop("bridge_kwargs")
    if "model_state_after_gen" in object_json:
        object_json["generator_state_after_gen"] = object_json.pop(
            "model_state_after_gen"
        )

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
    # NOTE: JSON converts all tuples to lists, and we need to convert them back. This is
    # an ad hoc fix, though.
    if isinstance(generator_run._best_arm_predictions, list):
        arm, arm_prediction = generator_run._best_arm_predictions
        if arm_prediction is not None and isinstance(arm_prediction, list):
            arm_prediction = tuple(arm_prediction)
        generator_run._best_arm_predictions = (arm, arm_prediction)

    if isinstance(generator_run._model_predictions, list):
        generator_run._model_predictions = tuple(generator_run._model_predictions)

    # Remove deprecated kwargs from generator kwargs & adapter kwargs.
    if generator_run._generator_kwargs is not None:
        generator_run._generator_kwargs = {
            k: v
            for k, v in generator_run._generator_kwargs.items()
            if k not in _DEPRECATED_GENERATOR_KWARGS
        }
    if generator_run._adapter_kwargs is not None:
        generator_run._adapter_kwargs = {
            k: v
            for k, v in generator_run._adapter_kwargs.items()
            if k not in _DEPRECATED_GENERATOR_KWARGS
        }
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
    return generator_run


def transition_criterion_from_json(
    transition_criterion_class: type[TransitionCriterion],
    object_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> TransitionCriterion:
    """Load TransitionCriterion from JSON.

    TransitionCriterion subclasses may contain nested Ax objects (like TrialStatus
    enums and AuxiliaryExperimentPurpose) that need recursive deserialization via
    object_from_json. We also use extract_init_args for backwards compatibility,
    filtering to only valid constructor arguments.
    """
    # Handle deprecated MinimumTrialsInStatus -> MinTrials conversion
    if transition_criterion_class is MinTrials and "status" in object_json:
        logger.warning(
            "`MinimumTrialsInStatus` has been deprecated and removed. "
            "Converting to `MinTrials` with equivalent functionality."
        )
        status = object_from_json(
            object_json=object_json.get("status"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        return MinTrials(
            threshold=object_json.get("threshold"),
            only_in_statuses=[status],
            transition_to=object_json.get("transition_to"),
            use_all_trials_in_exp=True,
        )

    decoded = {
        key: object_from_json(
            object_json=value,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        for key, value in object_json.items()
    }

    # filter to only valid constructor args (backwards compatibility)
    init_args = extract_init_args(args=decoded, class_=transition_criterion_class)

    # pyre-ignore[45]: Class passed is always a concrete subclass.
    return transition_criterion_class(**init_args)


def search_space_from_json(
    search_space_json: dict[str, Any],
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
    parameter_constraint_json: list[dict[str, Any]],
    parameters: list[Parameter],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> list[ParameterConstraint]:
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
    for constraint in parameter_constraint_json:
        # For backwards compatibility
        if constraint["__type"] == "OrderConstraint":
            parameter_constraints.append(
                ParameterConstraint(
                    inequality=(
                        f"{constraint['lower_name']} <= {constraint['upper_name']}"
                    )
                )
            )
        elif constraint["__type"] == "SumConstraint":
            parameter_constraints.append(
                ParameterConstraint(
                    inequality=" + ".join(constraint["parameter_names"])
                    + ("<=" if constraint["is_upper_bound"] else ">=")
                    + str(constraint["bound"])
                )
            )
        else:
            # Respect legacy json representation of parameter constraints
            if "constraint_dict" in constraint and "bound" in constraint:
                expr = " + ".join(
                    f"{coeff} * {param}"
                    for param, coeff in constraint["constraint_dict"].items()
                )

                parameter_constraints.append(
                    ParameterConstraint(
                        inequality=f"{expr} <= {constraint['bound']}",
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
    trials_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> dict[int, BaseTrial]:
    """Load Ax Trials from JSON."""
    loaded_trials = {}
    for index, trial_json in trials_json.items():
        is_trial = trial_json["__type"] == "Trial"
        trial_json = {
            k: object_from_json(
                v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in trial_json.items()
            if k != "__type"
        }
        if "generator_run_structs" in trial_json:
            # `GeneratorRunStruct` (deprecated) will be decoded into a `GeneratorRun`,
            # so all we have to do here is change the key it's stored under.
            trial_json["generator_runs"] = trial_json.pop("generator_run_structs")
        trial_json.pop("status_quo_weight_override", None)  # Deprecated.
        loaded_trials[int(index)] = (
            trial_from_json(experiment=experiment, **trial_json)
            if is_trial
            else batch_trial_from_json(experiment=experiment, **trial_json)
        )
    return loaded_trials


def data_from_json(
    data_by_trial_json: Mapping[str, Any] | Mapping[int, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Data:
    """
    Load Ax Data from JSON.

    Experiments used to have `_data_by_trial` is in the format
    `{trial_index: {timestamp: Data}}`; they now have a single `Data`. Data is
    still serialized via the old format (we intend to overhaul storage shortly).
    This function
    - combines multiple Datas for the same trial index into one, if there are
        multiple, keeping only the trial_index-arm_name-metric_name[-step]
        observation if it appears with multiple timestamps.
    - concatenates the data for each trial into one

    Produce `None` if `_data_by_trial` is empty. We do this rather than creating
    an empty data since we don't know the type of the data in that case.
    """
    data_by_trial = object_from_json(
        data_by_trial_json,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    # hack necessary because Python's json module converts dictionary
    # keys to strings: https://stackoverflow.com/q/1450957
    deserialized = {
        int(k): OrderedDict({int(k2): v2 for k2, v2 in v.items()})
        for k, v in data_by_trial.items()
    }
    return data_by_trial_to_data(data_by_trial=deserialized)


def multi_type_experiment_from_json(
    object_json: dict[str, Any],
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
    object_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> Experiment:
    """Load Ax Experiment from JSON."""
    experiment_info = _get_experiment_info(object_json)
    _trial_type_to_runner_json = object_json.pop("_trial_type_to_runner", None)
    _trial_type_to_runner = (
        object_from_json(
            _trial_type_to_runner_json,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        if _trial_type_to_runner_json is not None
        else None
    )

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

    # Handle backwards compatibility issue where some Experiments support None
    # trial types.
    if (
        _trial_type_to_runner is not None
        and len(_trial_type_to_runner) > 0
        and ({*_trial_type_to_runner.keys()} != {None})
    ):
        experiment._trial_type_to_runner = _trial_type_to_runner
    else:
        experiment._trial_type_to_runner = {
            Keys.DEFAULT_TRIAL_TYPE.value: experiment.runner
        }

    _load_experiment_info(
        exp=experiment,
        exp_info=experiment_info,
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    return experiment


def _get_experiment_info(object_json: dict[str, Any]) -> dict[str, Any]:
    """Returns basic information from `Experiment` object_json."""
    return {
        "time_created_json": object_json.pop("time_created"),
        "trials_json": object_json.pop("trials"),
        "experiment_type_json": object_json.pop("experiment_type"),
        "data_by_trial_json": object_json.pop("data_by_trial"),
    }


def _load_experiment_info(
    exp: Experiment,
    exp_info: dict[str, Any],
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
    exp.data = data_from_json(
        exp_info.get("data_by_trial_json", {}),
        decoder_registry=decoder_registry,
        class_decoder_registry=class_decoder_registry,
    )
    for trial in exp._trials.values():
        for arm in trial.arms:
            # Cast arm parameter values to the appropriate type based on the
            # search space parameter types. This is necessary because JSON may
            # deserialize values as different types (e.g., ints as floats).
            _cast_arm_parameters(arm, exp.search_space)
            exp._register_arm(arm)
        if trial.ttl_seconds is not None:
            exp._trials_have_ttl = True
    if exp.status_quo is not None:
        sq = none_throws(exp.status_quo)
        # Cast status_quo arm parameter values as well.
        _cast_arm_parameters(sq, exp.search_space)
        exp._register_arm(sq)


def _convert_generation_step_keys_for_backwards_compatibility(
    object_json: dict[str, Any],
) -> dict[str, Any]:
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
    generation_node_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GenerationNode:
    """Load GenerationNode object from JSON."""
    # Due to input_constructors being a dictionary with both keys and values being of
    # type enum, we must manually decode them here because object_from_json doesn't
    # recursively decode dictionary key values.
    decoded_input_constructors = None
    if "input_constructors" in generation_node_json.keys():
        decoded_input_constructors = {}
        for key, value in generation_node_json.pop("input_constructors").items():
            if key in _DEPRECATED_NODE_INPUT_CONSTRUCTORS:
                # Skip deprecated input constructors.
                continue
            decoded_input_constructors[InputConstructorPurpose[key]] = object_from_json(
                value,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
    if "model_specs" in generation_node_json:
        # Check for all kwarg to support backwards compatibility.
        generator_specs = generation_node_json.pop("model_specs")
    else:
        generator_specs = generation_node_json.pop("generator_specs")

    if "node_name" in generation_node_json.keys():
        name = generation_node_json.pop("node_name")
    else:
        name = generation_node_json.pop("name")

    # Pop step_index if present (for backward compatibility), but don't use it.
    # _step_index is non-persistent state that will be set by GenerationStrategy
    # if needed during _validate_and_set_step_sequence.
    generation_node_json.pop("step_index", None)

    # Backwards compatibility: For transition criteria with transition_to=None
    # set transition_to to point to itself.
    transition_criteria_json = generation_node_json.pop("transition_criteria")
    if transition_criteria_json is not None:
        for tc_json in transition_criteria_json:
            if tc_json.get("transition_to") is None:
                tc_json["transition_to"] = name

    return GenerationNode(
        name=name,
        generator_specs=object_from_json(
            object_json=generator_specs,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        best_model_selector=object_from_json(
            object_json=generation_node_json.pop("best_model_selector", None),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        should_deduplicate=generation_node_json.pop("should_deduplicate", False),
        transition_criteria=object_from_json(
            object_json=transition_criteria_json,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        input_constructors=decoded_input_constructors,
        previous_node_name=(
            generation_node_json.pop("previous_node_name")
            if "previous_node_name" in generation_node_json.keys()
            else None
        ),
        trial_type=(
            object_from_json(
                object_json=generation_node_json.pop("trial_type"),
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            if "trial_type" in generation_node_json.keys()
            else None
        ),
    )


def _sanitize_inputs_to_surrogate_spec(
    object_json: dict[str, Any],
) -> dict[str, Any]:
    """This is a backwards compatibility helper for inputs to ``SurrogateSpec``.
    It replaces the legacy inputs in the json with a  ``ModelConfig`` and discards
    the legacy inputs.
    """
    new_json = object_json.copy()
    # If no model configs are available, this spec was constructed using legacy inputs.
    # We will replace it with a model config constructed from the legacy inputs.
    # It is possible that both inputs are available, in which case we will discard the
    # legacy inputs and only keep the existing model config.
    model_configs = new_json.get("model_configs", [])
    new_config = [
        {
            "__type": "ModelConfig",
            "botorch_model_class": new_json.pop("botorch_model_class", None),
            "model_options": new_json.pop("botorch_model_kwargs", {}),
            "mll_class": new_json.pop("mll_class", None),
            "mll_options": new_json.pop("mll_kwargs", {}),
            "input_transform_classes": new_json.pop("input_transform_classes", DEFAULT),
            "input_transform_options": new_json.pop("input_transform_options", {})
            or {},  # Old default was None.
            "outcome_transform_classes": new_json.pop(
                "outcome_transform_classes", None
            ),
            "outcome_transform_options": new_json.pop("outcome_transform_options", {})
            or {},  # Old default was None.
            "covar_module_class": new_json.pop("covar_module_class", None),
            "covar_module_options": new_json.pop("covar_module_kwargs", {}) or {},
            "likelihood_class": new_json.pop("likelihood_class", None),
            "likelihood_options": new_json.pop("likelihood_kwargs", {}) or {},
            "name": "from deprecated args",
        },
    ]
    if len(model_configs) == 0:
        new_json["model_configs"] = new_config
    return new_json


def _sanitize_surrogate_spec_input(
    object_json: dict[str, Any],
) -> dict[str, Any]:
    """This is a backwards compatibility helper for ``SurrogateSpec`` related
    inputs to ``BoTorchGenerator``.

    If ``object_json`` includes a ``surrogate_specs`` key that is a dict
    with a single element, this method replaces it with `surrogate_spec`
    key with the value of that element.

    If the legacy inputs were used to initialize the ``SurrogateSpec``,
    a ``ModelConfig`` is constructed with the legacy inputs and the legacy
    inputs are discarded.

    Args:
        object_json: A dictionary of json encoded inputs to update.

    Returns:
        The json with the surrogate spec related inputs updated.
        If there are multiple elements in ``surrogate_specs``, the input is discarded
        after logging an exception. The default ``Surrogate`` will be used. Otherwise,
        returns a new dictionary with the ``surrogate_specs`` element replaced with
        ``surrogate_spec`` and legacy inputs replaced with ``ModelConfig``.
    """
    new_json = object_json.copy()
    specs = new_json.pop("surrogate_specs", None)
    if specs is None:
        return new_json
    if len(specs) > 1:
        logger.exception(
            "The input includes `surrogate_specs` with multiple elements. "
            "Support for multiple surrogates has been deprecated. "
            "Discarding the `surrogate_specs` input to facilitate loading "
            "of the experiment. The loaded object will utilize the default "
            "`Surrogate` and may not behave as expected."
        )
        return new_json

    spec = next(iter(specs.values()))
    new_json["surrogate_spec"] = _sanitize_inputs_to_surrogate_spec(object_json=spec)
    return new_json


def _sanitize_legacy_surrogate_inputs(
    object_json: dict[str, Any],
) -> dict[str, Any]:
    """This is a backwards compatibility helper for ``Surrogate`` that replaces
    the legacy top level inputs with a ``SurrogateSpec`` with a single ``ModelConfig``.
    """
    new_json = object_json.copy()
    if new_json.get("surrogate_spec", None) is None:
        config_json = {
            "__type": "ModelConfig",
            "botorch_model_class": new_json.pop("botorch_model_class", None),
            "model_options": new_json.pop("model_options", {}),
            "mll_class": new_json.pop("mll_class", None),
            "mll_options": new_json.pop("mll_options", {}),
            "input_transform_classes": new_json.pop("input_transform_classes", DEFAULT),
            "input_transform_options": new_json.pop("input_transform_options", {})
            or {},  # Old default was None.
            "outcome_transform_classes": new_json.pop(
                "outcome_transform_classes", None
            ),
            "outcome_transform_options": new_json.pop("outcome_transform_options", {})
            or {},  # Old default was None.
            "covar_module_class": new_json.pop("covar_module_class", None),
            "covar_module_options": new_json.pop("covar_module_options", {}) or {},
            "likelihood_class": new_json.pop("likelihood_class", None),
            "likelihood_options": new_json.pop("likelihood_options", {}) or {},
            "name": "from deprecated args",
        }
        new_json["surrogate_spec"] = {
            "__type": "SurrogateSpec",
            "model_configs": [config_json],
            "allow_batched_models": new_json.pop("allow_batched_models", True),
        }
    return new_json


def generation_step_from_json(
    generation_step_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GenerationStep:
    """Load generation step from JSON."""
    generation_step_json = _convert_generation_step_keys_for_backwards_compatibility(
        generation_step_json
    )
    if "model_kwargs" in generation_step_json:
        kwargs = generation_step_json.pop("model_kwargs", None)
    else:
        kwargs = generation_step_json.pop("generator_kwargs", None)
    if kwargs is not None:
        for k in _DEPRECATED_GENERATOR_KWARGS:
            # Remove deprecated kwargs.
            kwargs.pop(k, None)
        kwargs = _sanitize_surrogate_spec_input(object_json=kwargs)
    if "model_gen_kwargs" in generation_step_json:
        gen_kwargs = generation_step_json.pop("model_gen_kwargs", None)
    else:
        gen_kwargs = generation_step_json.pop("generator_gen_kwargs", None)
    if "model" in generation_step_json:
        # Old arg name for backwards compatibility.
        generator_json = generation_step_json.pop("model")
    else:
        generator_json = generation_step_json.pop("generator")
    generation_step = GenerationStep(
        generator=object_from_json(
            object_json=generator_json,
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        ),
        num_trials=generation_step_json.pop("num_trials"),
        min_trials_observed=generation_step_json.pop("min_trials_observed", 0),
        max_parallelism=(generation_step_json.pop("max_parallelism", None)),
        enforce_num_trials=generation_step_json.pop("enforce_num_trials", True),
        generator_kwargs=(
            _raise_on_legacy_callable_refs(
                object_from_json(
                    kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if kwargs
            else {}
        ),
        generator_gen_kwargs=(
            _raise_on_legacy_callable_refs(
                object_from_json(
                    gen_kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if gen_kwargs
            else {}
        ),
        index=generation_step_json.pop("index", -1),
        should_deduplicate=generation_step_json.pop("should_deduplicate", False),
        generator_name=generation_step_json.pop("generator_name", None),
        use_all_trials_in_exp=generation_step_json.pop("use_all_trials_in_exp", False),
    )
    return generation_step


def generator_spec_from_json(
    generator_spec_json: dict[str, Any],
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> GeneratorSpec:
    """Load GeneratorSpec from JSON."""
    generator_spec_json = generator_spec_json.copy()  # prevent in-place modification.
    if "model_kwargs" in generator_spec_json:
        kwargs = generator_spec_json.pop("model_kwargs", None)
    else:
        kwargs = generator_spec_json.pop("generator_kwargs", None)
    for k in _DEPRECATED_GENERATOR_KWARGS:
        # Remove deprecated model kwargs.
        kwargs.pop(k, None)
    if kwargs is not None:
        kwargs = _sanitize_surrogate_spec_input(object_json=kwargs)
    if "model_gen_kwargs" in generator_spec_json:
        gen_kwargs = generator_spec_json.pop("model_gen_kwargs", None)
    else:
        gen_kwargs = generator_spec_json.pop("generator_gen_kwargs", None)
    if "model_cv_kwargs" in generator_spec_json:
        cv_kwargs = generator_spec_json.pop("model_cv_kwargs", None)
    else:
        cv_kwargs = generator_spec_json.pop("cv_kwargs", None)
    if "model_enum" in generator_spec_json:
        # Old arg name for backwards compatibility.
        generator_spec_json["generator_enum"] = generator_spec_json.pop("model_enum")
    return GeneratorSpec(
        **{
            k: object_from_json(
                object_json=v,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            for k, v in generator_spec_json.items()
        },
        generator_kwargs=(
            _raise_on_legacy_callable_refs(
                object_from_json(
                    object_json=kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if kwargs
            else {}
        ),
        generator_gen_kwargs=(
            _raise_on_legacy_callable_refs(
                object_from_json(
                    object_json=gen_kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if gen_kwargs
            else {}
        ),
        cv_kwargs=(
            _raise_on_legacy_callable_refs(
                object_from_json(
                    object_json=cv_kwargs,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                ),
            )
            if cv_kwargs
            else {}
        ),
    )


def generation_strategy_from_json(
    generation_strategy_json: dict[str, Any],
    experiment: Experiment | None = None,
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
        gs._curr = gs._nodes[generation_strategy_json.pop("curr_index")]
    else:
        gs = GenerationStrategy(nodes=nodes, name=generation_strategy_json.pop("name"))
        curr_node_name = generation_strategy_json.pop("curr_node_name")
        for node in gs._nodes:
            if node.name == curr_node_name:
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
    list_surrogate_json: dict[str, Any],
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
        surrogate_spec=SurrogateSpec(
            model_configs=[
                ModelConfig(
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
                        object_json=list_surrogate_json.get(
                            "submodel_input_transform_classes"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    input_transform_options=object_from_json(
                        object_json=list_surrogate_json.get(
                            "submodel_input_transform_options"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    outcome_transform_classes=object_from_json(
                        object_json=list_surrogate_json.get(
                            "submodel_outcome_transform_classes"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    outcome_transform_options=object_from_json(
                        object_json=list_surrogate_json.get(
                            "submodel_outcome_transform_options"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    covar_module_class=object_from_json(
                        object_json=list_surrogate_json.get(
                            "submodel_covar_module_class"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    covar_module_options=list_surrogate_json.get(
                        "submodel_covar_module_options"
                    ),
                    likelihood_class=object_from_json(
                        object_json=list_surrogate_json.get(
                            "submodel_likelihood_class"
                        ),
                        decoder_registry=decoder_registry,
                        class_decoder_registry=class_decoder_registry,
                    ),
                    likelihood_options=list_surrogate_json.get(
                        "submodel_likelihood_options"
                    ),
                    name="from deprecated args",
                )
            ]
        )
    )


def get_input_transform_json_components(
    input_transforms_json: list[dict[str, Any]] | dict[str, Any] | None,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
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
        assert_is_instance(input_transform_json["__type"], str): input_transform_json[
            "state_dict"
        ]
        for input_transform_json in input_transforms_json
    }
    return input_transform_classes_json, input_transform_options_json


def get_outcome_transform_json_components(
    outcome_transforms_json: list[dict[str, Any]] | None,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    class_decoder_registry: TClassDecoderRegistry = CORE_CLASS_DECODER_REGISTRY,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
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
        assert_is_instance(
            outcome_transform_json["__type"], str
        ): outcome_transform_json["state_dict"]
        for outcome_transform_json in outcome_transforms_json
    }
    return outcome_transform_classes_json, outcome_transform_options_json


def objective_from_json(
    object_json: dict[str, Any],
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


def _update_deprecated_model_registry(name: str) -> str:
    """Update the enum name for deprecated model registry entries to point to
    a replacement model. This will log an exception to alert the user to the change.

    The replacement models are listed in `_DEPRECATED_GENERATOR_TO_REPLACEMENT` above.
    If a deprecated model does not list a replacement, nothing will be done and it
    will error out while looking it up in the corresponding enum.

    Args:
        name: The name of the ``Generators`` enum.

    Returns:
        Either the given name or the name of a replacement ``Generators`` enum.
    """
    if name in _DEPRECATED_GENERATOR_TO_REPLACEMENT:
        new_name = _DEPRECATED_GENERATOR_TO_REPLACEMENT[name]
        logger.exception(
            f"{name} model is deprecated and replaced by Generators.{new_name}. "
            f"Please use {new_name} in the future. Note that this warning only "
            "enables deserialization of experiments with deprecated models. "
            "Model fitting with the loaded experiment may still fail. "
        )
        return new_name
    else:
        return name


def _update_data_json_with_metric_signature(data_json_str: str) -> str:
    data_json = json.loads(data_json_str)
    data_json["metric_signature"] = data_json.get("metric_name", {})
    return json.dumps(data_json)
