# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

from ax.modelbridge.modelbridge_utils import (
    extract_search_space_digest,
    transform_search_space,
)
from ax.modelbridge.registry import ModelRegistryBase, SearchSpace
from ax.modelbridge.transforms.base import Transform

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
        return cls(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=-1,  # TODO[T147907632]
            model_fit_quality=-1,  # TODO[T147907632]
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


def _get_max_transformed_dimensionality(
    search_space: SearchSpace, generation_strategy: GenerationStrategy
) -> int:
    """
    Get dimensionality of transformed SearchSpace for all steps in the
    GenerationStrategy and return the maximum.
    """

    transforms_by_step = [
        _extract_transforms_and_configs(step=step)
        for step in generation_strategy._steps
    ]

    transformed_search_spaces = [
        transform_search_space(
            search_space=search_space,
            transforms=transforms,
            transform_configs=transform_configs,
        )
        for transforms, transform_configs in transforms_by_step
    ]

    # The length of the bounds of a SearchSpaceDigest is equal to the number of
    # dimensions present.
    dimensionalities = [
        len(
            extract_search_space_digest(
                search_space=tf_search_space,
                param_names=list(tf_search_space.parameters.keys()),
            ).bounds
        )
        for tf_search_space in transformed_search_spaces
    ]

    return max(dimensionalities)


def _extract_transforms_and_configs(
    step: GenerationStep,
) -> Tuple[List[Type[Transform]], Dict[str, Any]]:
    """
    Extract Transforms and their configs from the GenerationStep. Prefer kwargs
    provided over the model's defaults.
    """

    kwargs = step.model_spec.model_kwargs or {}
    transforms = kwargs.get("transforms")
    transform_configs = kwargs.get("transform_configs")

    if transforms is not None and transform_configs is not None:
        return transforms, transform_configs

    model = step.model
    if isinstance(model, ModelRegistryBase):
        _, bridge_kwargs = model.view_defaults()
        transforms = transforms or bridge_kwargs.get("transforms")
        transform_configs = transform_configs or bridge_kwargs.get("transform_configs")

    return (transforms or [], transform_configs or {})
