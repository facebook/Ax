# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from warnings import warn

from ax.service.scheduler import get_fitted_model_bridge, Scheduler
from ax.telemetry.common import _get_max_transformed_dimensionality

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
                None
                if scheduler.options.early_stopping_strategy is None
                else scheduler.options.early_stopping_strategy.__class__.__name__
            ),
            global_stopping_strategy_cls=(
                None
                if scheduler.options.global_stopping_strategy is None
                else scheduler.options.global_stopping_strategy.__class__.__name__
            ),
            transformed_dimensionality=_get_max_transformed_dimensionality(
                search_space=scheduler.experiment.search_space,
                generation_strategy=scheduler.generation_strategy,
            ),
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
class SchedulerCompletedRecord:
    """
    Record of the Scheduler completion event. This will have information only available
    after the optimization has completed.
    """

    experiment_completed_record: ExperimentCompletedRecord

    best_point_quality: float
    model_fit_quality: float

    num_metric_fetch_e_encountered: int
    num_trials_bad_due_to_err: int

    @classmethod
    def from_scheduler(cls, scheduler: Scheduler) -> SchedulerCompletedRecord:
        try:
            model_bridge = get_fitted_model_bridge(scheduler)
            model_fit_dict = model_bridge.compute_model_fit_metrics(
                experiment=scheduler.experiment
            )
            # We'd ideally log the entire `model_fit_dict` as a single model fit metric
            # can't capture the nuances of multiple experimental metrics, but this might
            # lead to database performance issues. So instead, we take the worst
            # coefficient of determination as model fit quality and store the full data
            # in Manifold (TODO).
            model_fit_quality = min(
                model_fit_dict["coefficient_of_determination"].values()
            )

        except Exception as e:
            warn("Encountered exception in computing model fit quality: " + str(e))
            model_fit_quality = float("-inf")
            # model_std_quality = float("-inf")

        return cls(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=float("-inf"),  # TODO[T147907632]
            model_fit_quality=model_fit_quality,
            num_metric_fetch_e_encountered=scheduler._num_metric_fetch_e_encountered,
            num_trials_bad_due_to_err=scheduler._num_trials_bad_due_to_err,
        )

    def flatten(self) -> Dict[str, Any]:
        """
        Flatten into an appropriate format for logging to a tabular database.
        """

        self_dict = asdict(self)
        experiment_completed_record_dict = self_dict.pop("experiment_completed_record")

        return {
            **self_dict,
            **experiment_completed_record_dict,
        }
