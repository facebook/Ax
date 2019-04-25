#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ax.core.generator_run import GeneratorRun
from ax.core.runner import Runner
from ax.core.trial import Trial


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


def batch_trial_from_json(
    experiment: "core.experiment.Experiment",
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
    status_quo_weight: float,
) -> BatchTrial:
    """Load Ax BatchTrial from JSON.

    Other classes don't need explicit deserializers, because we can just use
    their constructors (see decoder.py). However, the constructor for Batch
    does not allow us to exactly recreate an existing object.
    """

    batch = BatchTrial(experiment=experiment)
    batch._index = index
    batch._trial_type = trial_type
    batch._status = status
    batch._time_created = time_created
    batch._time_completed = time_completed
    batch._time_staged = time_staged
    batch._time_run_started = time_run_started
    batch._abandoned_reason = abandoned_reason
    batch._run_metadata = run_metadata or {}
    batch._generator_run_structs = generator_run_structs
    batch._runner = runner
    batch._abandoned_arms_metadata = abandoned_arms_metadata
    batch._num_arms_created = num_arms_created
    batch._status_quo = status_quo
    batch._status_quo_weight = status_quo_weight
    return batch


def trial_from_json(
    experiment: "core.experiment.Experiment",
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
) -> Trial:
    """Load Ax trial from JSON.

    Other classes don't need explicit deserializers, because we can just use
    their constructors (see decoder.py). However, the constructor for Trial
    does not allow us to exactly recreate an existing object.
    """

    trial = Trial(experiment=experiment, generator_run=generator_run)
    trial._index = index
    trial._trial_type = trial_type
    trial._status = status
    trial._time_created = time_created
    trial._time_completed = time_completed
    trial._time_staged = time_staged
    trial._time_run_started = time_run_started
    trial._abandoned_reason = abandoned_reason
    trial._run_metadata = run_metadata or {}
    trial._runner = runner
    trial._num_arms_created = num_arms_created
    return trial
