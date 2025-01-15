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

import torch
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import (
    AbandonedArm,
    BatchTrial,
    GeneratorRunStruct,
    LifecycleStage,
)
from ax.core.generator_run import GeneratorRun
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.exceptions.storage import JSONDecodeError
from ax.modelbridge.transforms.base import Transform
from ax.storage.botorch_modular_registry import (
    CLASS_TO_REVERSE_REGISTRY,
    REVERSE_INPUT_TRANSFORM_REGISTRY,
    REVERSE_OUTCOME_TRANSFORM_REGISTRY,
)
from ax.storage.transform_registry import REVERSE_TRANSFORM_REGISTRY
from ax.utils.common.kwargs import warn_on_kwargs
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils_torch import torch_type_from_str
from botorch.models.transforms.input import ChainedInputTransform, InputTransform
from botorch.models.transforms.outcome import ChainedOutcomeTransform, OutcomeTransform
from botorch.utils.types import _DefaultType, DEFAULT
from pyre_extensions import assert_is_instance
from torch.distributions.transformed_distribution import TransformedDistribution

logger: logging.Logger = get_logger(__name__)


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401

T = TypeVar("T")


def batch_trial_from_json(
    experiment: core.experiment.Experiment,
    index: int,
    trial_type: str | None,
    status: TrialStatus,
    time_created: datetime,
    time_completed: datetime | None,
    time_staged: datetime | None,
    time_run_started: datetime | None,
    abandoned_reason: str | None,
    run_metadata: dict[str, Any] | None,
    generator_run_structs: list[GeneratorRunStruct],
    runner: Runner | None,
    abandoned_arms_metadata: dict[str, AbandonedArm],
    num_arms_created: int,
    status_quo: Arm | None,
    status_quo_weight_override: float,
    optimize_for_power: bool | None,
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    failed_reason: str | None = None,
    ttl_seconds: int | None = None,
    generation_step_index: int | None = None,
    properties: dict[str, Any] | None = None,
    stop_metadata: dict[str, Any] | None = None,
    lifecycle_stage: LifecycleStage | None = None,
    **kwargs: Any,
) -> BatchTrial:
    """Load Ax BatchTrial from JSON.

    Other classes don't need explicit deserializers, because we can just use
    their constructors (see decoder.py). However, the constructor for Batch
    does not allow us to exactly recreate an existing object.
    """

    batch = BatchTrial(experiment=experiment, ttl_seconds=ttl_seconds)
    batch._index = index
    batch._trial_type = trial_type
    batch._status = status
    batch._time_created = time_created
    batch._time_completed = time_completed
    batch._time_staged = time_staged
    batch._time_run_started = time_run_started
    batch._abandoned_reason = abandoned_reason
    batch._failed_reason = failed_reason
    batch._run_metadata = run_metadata or {}
    batch._stop_metadata = stop_metadata or {}
    batch._generator_run_structs = generator_run_structs
    batch._runner = runner
    batch._abandoned_arms_metadata = abandoned_arms_metadata
    batch._num_arms_created = num_arms_created
    batch._status_quo = status_quo
    batch._status_quo_weight_override = status_quo_weight_override
    batch.optimize_for_power = optimize_for_power
    batch._generation_step_index = generation_step_index
    batch._lifecycle_stage = lifecycle_stage
    batch._properties = properties
    batch._refresh_arms_by_name()  # Trigger cache build
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
    abandoned_reason: str | None,
    run_metadata: dict[str, Any] | None,
    generator_run: GeneratorRun,
    runner: Runner | None,
    num_arms_created: int,
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    failed_reason: str | None = None,
    ttl_seconds: int | None = None,
    generation_step_index: int | None = None,
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
    trial._trial_type = trial_type
    # Swap `DISPATCHED` for `RUNNING`, since `DISPATCHED` is deprecated and nearly
    # equivalent to `RUNNING`.
    trial._status = status if status != TrialStatus.DISPATCHED else TrialStatus.RUNNING
    trial._time_created = time_created
    trial._time_completed = time_completed
    trial._time_staged = time_staged
    trial._time_run_started = time_run_started
    trial._abandoned_reason = abandoned_reason
    trial._failed_reason = failed_reason
    trial._run_metadata = run_metadata or {}
    trial._stop_metadata = stop_metadata or {}
    trial._runner = runner
    trial._num_arms_created = num_arms_created
    trial._generation_step_index = generation_step_index
    trial._properties = properties or {}
    warn_on_kwargs(callable_with_kwargs=Trial, **kwargs)
    return trial


def transform_type_from_json(object_json: dict[str, Any]) -> type[Transform]:
    """Load the transform type from JSON."""
    index_in_registry = object_json.pop("index_in_registry")
    if index_in_registry not in REVERSE_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform '{object_json.pop('transform_type')}'")
    return REVERSE_TRANSFORM_REGISTRY[index_in_registry]


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


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def class_from_json(json: dict[str, Any]) -> type[Any]:
    """Load any class registered in `CLASS_DECODER_REGISTRY` from JSON."""
    index_in_registry = json.pop("index")
    class_path = json.pop("class")
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
        # Extract the transformed attributes for transformed priors.
        for k in list(state_dict.keys()):
            if k.startswith("_transformed_"):
                state_dict[k[13:]] = state_dict.pop(k)
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
