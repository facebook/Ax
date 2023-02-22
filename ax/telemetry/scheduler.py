# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord


@dataclass(frozen=True)
class SchedulerCreatedRecord(ExperimentCreatedRecord, GenerationStrategyCreatedRecord):
    """
    Record of the Scheduler creation event. This can be used for telemetry in settings
    where many Schedulers are being created either manually or programatically. In
    order to facilitate easy serialization only include simple types: numbers, strings,
    bools, and None.
    """

    # SchedulerOptions info
    total_trials: Optional[int]
    max_pending_trials: int
    arms_per_trial: int
    early_stopping_strategy_cls: Optional[str]
    global_stopping_strategy_cls: Optional[str]

    # Dimensionality of transformed SearchSpace can often be much higher due to one-hot
    # encoding of unordered ChoiceParameters
    transformed_dimensionality: int

    product_surface: str
    launch_surface: str
    # Can be used to join against deployment engine-specific tables for more metadata
    deployed_job_id: Optional[int]

    is_manual_generation_strategy: bool
    warm_started_from: Optional[str]
    num_custom_trials: int


@dataclass(frozen=True)
class SchedulerCompletedRecord(ExperimentCompletedRecord):
    """
    Record of the Scheduler completion event. This will have information only available
    after the optimization has completed.
    """

    best_point_quality: float
    model_fit_quality: float

    total_fit_time: float
    total_gen_time: float

    estimated_early_stopping_savings: float
    estimated_global_stopping_savings: float

    num_metric_fetch_errs_encountered: int

    # Can be used to join against deployment engine-specific tables for more metadata,
    # and with scheduler creation event table
    deployed_job_id: Optional[int]
