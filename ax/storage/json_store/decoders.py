#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ax.core.generator_run import GeneratorRun
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.modelbridge.transforms.base import Transform
from ax.storage.botorch_modular_registry import CLASS_TO_REVERSE_REGISTRY
from ax.storage.transform_registry import REVERSE_TRANSFORM_REGISTRY
from ax.utils.common.kwargs import warn_on_kwargs


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


def batch_trial_from_json(
    experiment: core.experiment.Experiment,
    index: int,
    trial_type: Optional[str],
    status: TrialStatus,
    time_created: datetime,
    time_completed: Optional[datetime],
    time_staged: Optional[datetime],
    time_run_started: Optional[datetime],
    abandoned_reason: Optional[str],
    run_metadata: Optional[Dict[str, Any]],
    generator_run_structs: List[GeneratorRunStruct],
    runner: Optional[Runner],
    abandoned_arms_metadata: Dict[str, AbandonedArm],
    num_arms_created: int,
    status_quo: Optional[Arm],
    status_quo_weight_override: float,
    optimize_for_power: Optional[bool],
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    ttl_seconds: Optional[int] = None,
    generation_step_index: Optional[int] = None,
    properties: Optional[Dict[str, Any]] = None,
    stop_metadata: Optional[Dict[str, Any]] = None,
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
    batch._properties = properties
    batch._refresh_arms_by_name()  # Trigger cache build
    warn_on_kwargs(callable_with_kwargs=BatchTrial, **kwargs)
    return batch


def trial_from_json(
    experiment: core.experiment.Experiment,
    index: int,
    trial_type: Optional[str],
    status: TrialStatus,
    time_created: datetime,
    time_completed: Optional[datetime],
    time_staged: Optional[datetime],
    time_run_started: Optional[datetime],
    abandoned_reason: Optional[str],
    run_metadata: Optional[Dict[str, Any]],
    generator_run: GeneratorRun,
    runner: Optional[Runner],
    num_arms_created: int,
    # Allowing default values for backwards compatibility with
    # objects stored before these fields were added.
    ttl_seconds: Optional[int] = None,
    generation_step_index: Optional[int] = None,
    properties: Optional[Dict[str, Any]] = None,
    stop_metadata: Optional[Dict[str, Any]] = None,
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
    trial._run_metadata = run_metadata or {}
    trial._stop_metadata = stop_metadata or {}
    trial._runner = runner
    trial._num_arms_created = num_arms_created
    trial._generation_step_index = generation_step_index
    trial._properties = properties or {}
    warn_on_kwargs(callable_with_kwargs=Trial, **kwargs)
    return trial


def transform_type_from_json(object_json: Dict[str, Any]) -> Type[Transform]:
    """Load the transform type from JSON."""
    index_in_registry = object_json.pop("index_in_registry")
    if index_in_registry not in REVERSE_TRANSFORM_REGISTRY:  # pragma: no cover
        raise ValueError(f"Unknown transform '{object_json.pop('transform_type')}'")
    return REVERSE_TRANSFORM_REGISTRY[index_in_registry]


def class_from_json(json: Dict[str, Any]) -> Type[Any]:
    """Load any class registered in `CLASS_DECODER_REGISTRY` from JSON."""
    index_in_registry = json.pop("index")
    class_path = json.pop("class")
    for _class in CLASS_TO_REVERSE_REGISTRY:
        if class_path == f"{_class}":
            reverse_registry = CLASS_TO_REVERSE_REGISTRY[_class]
            if index_in_registry not in reverse_registry:  # pragma: no cover
                raise ValueError(
                    f"Index '{index_in_registry}'"
                    " is not registered in the reverse registry."
                )
            return reverse_registry[index_in_registry]
    raise ValueError(
        f"{class_path} does not have a corresponding entry in "
        "CLASS_TO_REVERSE_REGISTRY."
    )
