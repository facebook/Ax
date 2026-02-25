#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeVar

import pandas as pd
import torch
from ax.adapter.transforms.base import Transform
from ax.core.arm import Arm
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    PARAMETER_PYTHON_TYPE_MAP,
    ParameterType,
    TParamValue,
)
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import TCandidateMetadata
from ax.early_stopping.strategies.base import REMOVED_EARLY_STOPPING_STRATEGY_KWARGS
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.early_stopping.strategies.threshold import ThresholdEarlyStoppingStrategy
from ax.exceptions.storage import JSONDecodeError
from ax.storage.botorch_modular_registry import (
    CLASS_TO_REVERSE_REGISTRY,
    REVERSE_INPUT_TRANSFORM_REGISTRY,
    REVERSE_OUTCOME_TRANSFORM_REGISTRY,
)
from ax.storage.transform_registry import (
    DEPRECATED_TRANSFORMS,
    REMOVED_TRANSFORMS,
    REVERSE_TRANSFORM_REGISTRY,
)
from ax.utils.common.constants import Keys
from ax.utils.common.kwargs import warn_on_kwargs
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils_torch import torch_type_from_str
from botorch.models.transforms.input import ChainedInputTransform, InputTransform
from botorch.models.transforms.outcome import ChainedOutcomeTransform, OutcomeTransform
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.priors.utils import BUFFERED_PREFIX
from pyre_extensions import assert_is_instance
from torch.distributions.transformed_distribution import TransformedDistribution

logger: logging.Logger = get_logger(__name__)


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401

T = TypeVar("T")


def string_to_parameter_value(s: str, parameter_type: ParameterType) -> TParamValue:
    if parameter_type == ParameterType.BOOL:
        if s == "false":
            return False
        elif s == "true":
            return True
        else:
            raise ValueError(
                f"Expected 'true' or 'false' for boolean parameter, got {s}."
            )

    elif parameter_type == ParameterType.INT:
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"Expected integer for int parameter, got {s}.")

    elif parameter_type == ParameterType.FLOAT:
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Expected float for float parameter, got {s}.")

    elif parameter_type == ParameterType.STRING:
        return s


def _cast_parameter_value(
    value: TParamValue, parameter_type: ParameterType
) -> TParamValue:
    """Cast a parameter value to the appropriate Python type based on parameter_type.

    This is necessary because JSON may deserialize values as different types
    (e.g., ints as floats).

    Args:
        value: The value to cast.
        parameter_type: The ParameterType to cast to.

    Returns:
        The value cast to the appropriate Python type.
    """
    if value is None:
        return None
    python_type = PARAMETER_PYTHON_TYPE_MAP[parameter_type]
    return python_type(value)


def batch_trial_from_json(
    experiment: core.experiment.Experiment,
    index: int,
    trial_type: str | None,
    status: TrialStatus,
    time_created: datetime,
    time_completed: datetime | None,
    time_staged: datetime | None,
    time_run_started: datetime | None,
    run_metadata: dict[str, Any] | None,
    generator_runs: list[GeneratorRun],
    runner: Runner | None,
    abandoned_arms_metadata: dict[str, AbandonedArm],
    num_arms_created: int,
    # TODO: check status_quo logic
    status_quo: Arm | None,
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    status_reason: str | None = None,
    abandoned_reason: str | None = None,
    failed_reason: str | None = None,
    ttl_seconds: int | None = None,
    properties: dict[str, Any] | None = None,
    stop_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BatchTrial:
    """Load Ax BatchTrial from JSON.

    Other classes don't need explicit deserializers, because we can just use
    their constructors (see decoder.py). However, the constructor for Batch
    does not allow us to exactly recreate an existing object.
    """

    batch = BatchTrial(
        experiment=experiment,
        ttl_seconds=ttl_seconds,
        generator_runs=generator_runs,
        # Historically, some trials might have status quo arms that do not
        # match the SQ set on the experiment, so we cannot just pass in
        # `shoud_add_status_quo_arm=True` here. Instead, we manually re-add
        # the SQ at the end of this function.
    )
    batch._index = index
    batch._trial_type = (
        trial_type if trial_type is not None else Keys.DEFAULT_TRIAL_TYPE.value
    )
    batch._time_created = time_created
    batch._time_completed = time_completed
    batch._time_staged = time_staged
    batch._time_run_started = time_run_started
    # Backward compatibility: use status_reason if available, otherwise fall back
    # to abandoned_reason or failed_reason
    batch._status_reason = status_reason or abandoned_reason or failed_reason
    batch._run_metadata = run_metadata or {}
    batch._stop_metadata = stop_metadata or {}
    batch._generator_runs = generator_runs
    batch._abandoned_arms_metadata = abandoned_arms_metadata
    batch._num_arms_created = num_arms_created
    batch._properties = properties or {}
    batch._refresh_arms_by_name()  # Trigger cache build

    # Trial.arms_by_name only returns arms with weights
    batch.should_add_status_quo_arm = batch.status_quo is not None
    if batch.should_add_status_quo_arm:
        batch.add_status_quo_arm()

    # Set trial status last, after adding all the arms.
    batch._status = status
    warn_on_kwargs(callable_with_kwargs=BatchTrial, **kwargs)
    return batch


def trial_from_json(
    experiment: core.experiment.Experiment,
    index: int,
    trial_type: str | None,
    status: TrialStatus,
    time_created: datetime,
    time_completed: datetime | None,
    time_staged: datetime | None,
    time_run_started: datetime | None,
    run_metadata: dict[str, Any] | None,
    generator_run: GeneratorRun,
    runner: Runner | None,
    num_arms_created: int,
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    status_reason: str | None = None,
    abandoned_reason: str | None = None,
    failed_reason: str | None = None,
    ttl_seconds: int | None = None,
    properties: dict[str, Any] | None = None,
    stop_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Trial:
    """Load Ax trial from JSON.

    Other classes don't need explicit deserializers, because we can just use
    their constructors (see decoder.py). However, the constructor for Trial
    does not allow us to exactly recreate an existing object.
    """

    trial = Trial(
        experiment=experiment, generator_run=generator_run, ttl_seconds=ttl_seconds
    )
    trial._index = index
    trial._trial_type = (
        trial_type if trial_type is not None else Keys.DEFAULT_TRIAL_TYPE.value
    )
    # Swap `DISPATCHED` for `RUNNING`, since `DISPATCHED` is deprecated and nearly
    # equivalent to `RUNNING`.
    trial._status = status if status != TrialStatus.DISPATCHED else TrialStatus.RUNNING
    trial._time_created = time_created
    trial._time_completed = time_completed
    trial._time_staged = time_staged
    trial._time_run_started = time_run_started
    # Backward compatibility: use status_reason if available, otherwise fall back
    # to abandoned_reason or failed_reason
    trial._status_reason = status_reason or abandoned_reason or failed_reason
    trial._run_metadata = run_metadata or {}
    trial._stop_metadata = stop_metadata or {}
    trial._num_arms_created = num_arms_created
    trial._properties = properties or {}
    warn_on_kwargs(callable_with_kwargs=Trial, **kwargs)
    return trial


def transform_type_from_json(object_json: dict[str, Any]) -> type[Transform]:
    """Load the transform type from JSON."""
    transform_type = object_json.pop("transform_type")
    # As the encoder is implemented, this transform type will just be the
    # name of the transform. However, the previous implementation utilized
    # the str(transform_type), which produces a string including the
    # module path. If this is the case, first we need to extract the class name.
    if transform_type.startswith("<class '"):
        # The string is "<class 'ax.adapter.transforms.transform_type'>".
        transform_type = transform_type[:-2].split(".")[-1]
    # Handle deprecated & removed transforms.
    if transform_type in DEPRECATED_TRANSFORMS:
        return DEPRECATED_TRANSFORMS[transform_type]
    if transform_type in REMOVED_TRANSFORMS:
        logger.exception(
            f"Transform {transform_type} has been deprecated and removed from Ax. "
            "We are unable to load this transform and will return the base "
            "`Transform` class instead. The models on the loaded generation strategy "
            "may not work correctly!"
        )
        return Transform
    return REVERSE_TRANSFORM_REGISTRY[transform_type]


def input_transform_type_from_json(object_json: dict[str, Any]) -> type[InputTransform]:
    input_transform_type = object_json.pop("index")
    if input_transform_type not in REVERSE_INPUT_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform {input_transform_type}.")
    return REVERSE_INPUT_TRANSFORM_REGISTRY[input_transform_type]


def outcome_transform_type_from_json(
    object_json: dict[str, Any],
) -> type[OutcomeTransform]:
    outcome_transform_type = object_json.pop("index")
    if outcome_transform_type not in REVERSE_OUTCOME_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform {outcome_transform_type}.")
    return REVERSE_OUTCOME_TRANSFORM_REGISTRY[outcome_transform_type]


def class_from_json(json: dict[str, Any]) -> type[Any]:
    """Load any class registered in `CLASS_DECODER_REGISTRY` from JSON."""
    index_in_registry = json.pop("index")
    class_path = json.pop("class")
    # Replace modelbridge -> adapter, models -> generators for backwards compatibility.
    class_path = class_path.replace("ax.modelbridge", "ax.adapter").replace(
        "ax.models", "ax.generators"
    )
    for _class in CLASS_TO_REVERSE_REGISTRY:
        if class_path == f"{_class}":
            reverse_registry = CLASS_TO_REVERSE_REGISTRY[_class]
            if index_in_registry not in reverse_registry:
                raise ValueError(
                    f"Index '{index_in_registry}' is not registered in the reverse "
                    f"registry."
                )
            return reverse_registry[index_in_registry]
    raise ValueError(
        f"{class_path} does not have a corresponding entry in "
        "CLASS_TO_REVERSE_REGISTRY."
    )


def tensor_from_json(json: dict[str, Any]) -> torch.Tensor:
    try:
        device = (
            assert_is_instance(
                torch_type_from_str(
                    identifier=json["device"]["value"], type_name="device"
                ),
                torch.device,
            )
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return torch.tensor(
            json["value"],
            dtype=assert_is_instance(
                torch_type_from_str(
                    identifier=json["dtype"]["value"], type_name="dtype"
                ),
                torch.dtype,
            ),
            device=device,
        )
    except KeyError as e:
        raise JSONDecodeError(
            f"Got KeyError {e} while attempting to construct a tensor from json. "
            f"Expected value, dtype, and device fields; got {json=}."
        )


def tensor_or_size_from_json(json: dict[str, Any]) -> torch.Tensor | torch.Size:
    if json["__type"] == "Tensor":
        return tensor_from_json(json)
    elif json["__type"] == "torch_Size":
        return assert_is_instance(
            torch_type_from_str(identifier=json["value"], type_name="Size"),
            torch.Size,
        )
    else:
        raise JSONDecodeError(
            f"Expected json encoding of a torch.Tensor or torch.Size. Got {json=}"
        )


def botorch_component_from_json(botorch_class: type[T], json: dict[str, Any]) -> T:
    """Load any instance of `torch.nn.Module` or descendants registered in
    `CLASS_DECODER_REGISTRY` from state dict."""
    state_dict = json.pop("state_dict")
    if issubclass(botorch_class, ChainedInputTransform):
        return botorch_class(
            **{
                k: botorch_component_from_json(
                    botorch_class=REVERSE_INPUT_TRANSFORM_REGISTRY[v.pop("__type")],
                    json=v,
                )
                for k, v in state_dict.items()
            }
        )
    if issubclass(botorch_class, ChainedOutcomeTransform):
        return botorch_class(
            **{
                k: botorch_component_from_json(
                    botorch_class=REVERSE_OUTCOME_TRANSFORM_REGISTRY[v.pop("__type")],
                    json=v,
                )
                for k, v in state_dict.items()
            }
        )
    if issubclass(botorch_class, TransformedDistribution):
        # Extract the buffered attributes for transformed priors.
        for k in list(state_dict.keys()):
            if k.startswith(BUFFERED_PREFIX):
                state_dict[k[len(BUFFERED_PREFIX) :]] = state_dict.pop(k)
    class_path = json.pop("class")
    init_args = inspect.signature(botorch_class).parameters
    required_args = {
        p for p, v in init_args.items() if v.default is inspect._empty and p != "kwargs"
    }
    allowable_args = set(init_args)
    received_args = set(state_dict)
    missing_args = required_args - received_args
    if missing_args:
        raise ValueError(
            f"Missing required initialization args {missing_args} for class "
            f"{class_path}. For gpytorch objects, this is likely because the "
            "object's `state_dict` method does not return the args required "
            "for initialization."
        )
    extra_args = received_args - allowable_args
    if extra_args:
        raise ValueError(
            f"Received unused args {extra_args} for class {class_path}. "
            "For gpytorch objects, this is likely because the object's "
            "`state_dict` method returns these extra args, which could "
            "indicate that the object's state will not be fully recreated "
            "by this serialization/deserialization method."
        )
    return botorch_class(
        **{
            k: (
                tensor_or_size_from_json(json=v)
                if isinstance(v, dict) and "__type" in v
                else v
            )
            for k, v in state_dict.items()
        }
    )


def pathlib_from_json(pathsegments: str | Iterable[str]) -> Path:
    if isinstance(pathsegments, str):
        return Path(pathsegments)

    return Path(*pathsegments)


def default_from_json(json: dict[str, Any]) -> _DefaultType:
    if json != {}:
        raise JSONDecodeError(
            f"Expected empty json object for ``DEFAULT``, got {json=}"
        )
    return DEFAULT


def multi_objective_from_json(
    objectives: list[Objective], **kwargs: Any
) -> MultiObjective:
    """
    Load MultiObjective from JSON.

    Ignore fields which are no longer supported, such as ``weights``,
    ``metrics``, and ``minimize``.
    """
    if len(kwargs) > 0:
        warn_on_kwargs(callable_with_kwargs=MultiObjective, **kwargs)
    return MultiObjective(objectives=objectives)


def choice_parameter_from_json(
    name: str,
    parameter_type: ParameterType,
    values: list[TParamValue],
    is_ordered: bool | None = None,
    is_task: bool = False,
    is_fidelity: bool = False,
    target_value: TParamValue = None,
    sort_values: bool | None = None,
    log_scale: bool | None = None,
    dependents: dict[TParamValue, list[str]] | None = None,
) -> ChoiceParameter:
    # JSON converts dictionary keys to strings. We need to convert them back.
    if dependents is not None:
        dependents = {
            # pyre-ignore [6]: JSON keys are always strings
            string_to_parameter_value(s=key, parameter_type=parameter_type): value
            for key, value in dependents.items()
        }

    # Backward compatibility: Override sort_values=False for numeric ordered parameters
    # to prevent validation errors when loading old experiments.
    if (
        sort_values is False
        and parameter_type.is_numeric
        and (is_ordered or (is_ordered is None and len(values) == 2))
    ):
        logger.warning(
            f"Parameter '{name}' is numeric ordered with sort_values=False. "
            f"Overriding to sort_values=True for backward compatibility. "
            f"This parameter was likely stored before the validation requiring "
            f"sort_values=True for numeric ordered parameters was added."
        )
        sort_values = True

    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=is_ordered,
        is_task=is_task,
        is_fidelity=is_fidelity,
        target_value=target_value,
        sort_values=sort_values,
        log_scale=log_scale,
        dependents=dependents,
    )


def fixed_parameter_from_json(
    name: str,
    parameter_type: ParameterType,
    value: TParamValue,
    is_fidelity: bool = False,
    target_value: TParamValue = None,
    dependents: dict[TParamValue, list[str]] | None = None,
) -> FixedParameter:
    # JSON converts dictionary keys to strings. We need to convert them back.
    if dependents is not None:
        dependents = {
            # pyre-ignore [6]: JSON keys are always strings
            string_to_parameter_value(s=key, parameter_type=parameter_type): value
            for key, value in dependents.items()
        }

    return FixedParameter(
        name=name,
        parameter_type=parameter_type,
        value=value,
        is_fidelity=is_fidelity,
        target_value=target_value,
        dependents=dependents,
    )


def observation_features_from_json(
    parameters: dict[str, TParamValue],
    trial_index: int,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    metadata: TCandidateMetadata = None,
    **kwargs: Any,
) -> ObservationFeatures:
    """
    `random_split` used to be supported, so it may appear in
    `kwargs`.
    """
    warn_on_kwargs(callable_with_kwargs=ObservationFeatures, **kwargs)
    return ObservationFeatures(
        parameters=parameters,
        trial_index=trial_index,
        start_time=start_time,
        end_time=end_time,
        metadata=metadata,
    )


def percentile_early_stopping_strategy_from_json(
    **kwargs: Any,
) -> PercentileEarlyStoppingStrategy:
    """Load PercentileEarlyStoppingStrategy from JSON.

    Discards removed kwargs for backwards compatibility.
    """
    for key in REMOVED_EARLY_STOPPING_STRATEGY_KWARGS:
        kwargs.pop(key, None)
    return PercentileEarlyStoppingStrategy(**kwargs)


def threshold_early_stopping_strategy_from_json(
    **kwargs: Any,
) -> ThresholdEarlyStoppingStrategy:
    """Load ThresholdEarlyStoppingStrategy from JSON.

    Discards removed kwargs for backwards compatibility.
    """
    for key in REMOVED_EARLY_STOPPING_STRATEGY_KWARGS:
        kwargs.pop(key, None)
    return ThresholdEarlyStoppingStrategy(**kwargs)
