# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing import Any, Dict, Optional

from ax.telemetry.scheduler import SchedulerCompletedRecord, SchedulerCreatedRecord


@dataclass(frozen=True)
class OptimizationCreatedRecord:
    """
    Record of the "Optimization" creation event. This includes the
    SchedulerCreatedRecord as well as miscellaneous metadata about the optimization not
    available from the Scheduler. In order to facilitate easy serialization only
    include simple types: numbers, strings, bools, and None.
    """

    scheduler_created_record: SchedulerCreatedRecord

    product_surface: str
    launch_surface: str

    deployed_job_id: int
    trial_evaluation_identifier: Optional[str]

    # Miscellaneous product info
    is_manual_generation_strategy: bool
    warm_started_from: Optional[str]
    num_custom_trials: int

    def flatten(self) -> Dict[str, Any]:
        """
        Flatten into an appropriate format for logging to a tabular database.
        """

        self_dict = asdict(self)
        self_dict.pop("scheduler_created_record")
        scheduler_created_record_dict = self.scheduler_created_record.flatten()

        return {
            **self_dict,
            **scheduler_created_record_dict,
        }


@dataclass(frozen=True)
class OptimizationCompletedRecord:
    """
    Record of the "Optimization" completion event. This includes the
    SchedulerCompletedRecord as well as miscellaneous metadata about the optimization
    not available from the Scheduler. In order to facilitate easy serialization only
    include simple types: numbers, strings, bools, and None.
    """

    scheduler_completed_record: SchedulerCompletedRecord

    # Can be used to join against deployment engine-specific tables for more metadata,
    # and with scheduler creation event table
    deployed_job_id: Optional[int]

    # Miscellaneous deployment specific info
    estimated_early_stopping_savings: float
    estimated_global_stopping_savings: float

    def flatten(self) -> Dict[str, Any]:
        """
        Flatten into an appropriate format for logging to a tabular database.
        """

        self_dict = asdict(self)
        self_dict.pop("scheduler_completed_record")
        scheduler_completed_record_dict = self.scheduler_completed_record.flatten()

        return {
            **self_dict,
            **scheduler_completed_record_dict,
        }
