# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from ax.service.scheduler import Scheduler

from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord


@dataclass(frozen=True)
class SchedulerCreatedRecord:
    """
    Record of the Scheduler creation event. This can be used for telemetry in settings
    where many Schedulers are being created either manually or programatically. In
    order to facilitate easy serialization only include simple types: numbers, strings,
    bools, and None.
    """

    experiment_created_record: ExperimentCreatedRecord
    generation_strategy_created_record: GenerationStrategyCreatedRecord

    # SchedulerOptions info
    scheduler_total_trials: Optional[int]
    scheduler_max_pending_trials: int
    arms_per_trial: int
    early_stopping_strategy_cls: Optional[str]
    global_stopping_strategy_cls: Optional[str]

    # Dimensionality of transformed SearchSpace can often be much higher due to one-hot
    # encoding of unordered ChoiceParameters
    transformed_dimensionality: int

    @classmethod
    def from_scheduler(cls, scheduler: Scheduler) -> SchedulerCreatedRecord:
        return cls(
            experiment_created_record=ExperimentCreatedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            generation_strategy_created_record=(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=scheduler.generation_strategy
                )
            ),
            scheduler_total_trials=scheduler.options.total_trials,
            scheduler_max_pending_trials=scheduler.options.max_pending_trials,
            # If batch_size is None then we are using single-Arm trials
            arms_per_trial=scheduler.options.batch_size or 1,
            early_stopping_strategy_cls=(
                scheduler.options.early_stopping_strategy.__class__.__name__
            ),
            global_stopping_strategy_cls=(
                scheduler.options.global_stopping_strategy.__class__.__name__
            ),
            transformed_dimensionality=-1,  # TODO[mpolson64]
        )

    def flatten(self) -> Dict[str, Any]:
        """
        Flatten into an appropriate format for logging to a tabular database.
        """

        self_dict = asdict(self)
        experiment_created_record_dict = self_dict.pop("experiment_created_record")
        generation_strategy_created_record_dict = self_dict.pop(
            "generation_strategy_created_record"
        )

        return {
            **self_dict,
            **experiment_created_record_dict,
            **generation_strategy_created_record_dict,
        }


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
