# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy

from ax.service.ax_client import AxClient
from ax.service.scheduler import Scheduler
from ax.telemetry.ax_client import AxClientCompletedRecord, AxClientCreatedRecord
from ax.telemetry.common import (
    _get_max_transformed_dimensionality,
    DEFAULT_PRODUCT_SURFACE,
)
from ax.telemetry.experiment import ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.telemetry.scheduler import SchedulerCompletedRecord, SchedulerCreatedRecord


@dataclass(frozen=True)
class OptimizationCreatedRecord:
    """
    Record of the "Optimization" creation event. This can come from either an
    AxClient or a Scheduler. This Record is especially useful for logging Ax-backed
    optimization results to a tabular database (i.e. one row per Record).
    """

    unique_identifier: str
    owner: str

    # ExperimentCreatedRecord fields
    experiment_name: Optional[str]
    experiment_type: Optional[str]
    num_continuous_range_parameters: int
    num_int_range_parameters_small: int
    num_int_range_parameters_medium: int
    num_int_range_parameters_large: int
    num_log_scale_range_parameters: int
    num_unordered_choice_parameters_small: int
    num_unordered_choice_parameters_medium: int
    num_unordered_choice_parameters_large: int
    num_fixed_parameters: int
    dimensionality: int
    hierarchical_tree_height: int
    num_parameter_constraints: int
    num_objectives: int
    num_tracking_metrics: int
    num_outcome_constraints: int
    num_map_metrics: int
    metric_cls_to_quantity: Dict[str, int]
    runner_cls: str

    # GenerationStrategyCreatedRecord fields
    generation_strategy_name: Optional[str]
    num_requested_initialization_trials: Optional[int]
    num_requested_bayesopt_trials: Optional[int]
    num_requested_other_trials: Optional[int]
    max_parallelism: Optional[int]

    # {AxClient, Scheduler}CreatedRecord fields
    early_stopping_strategy_cls: Optional[str]
    global_stopping_strategy_cls: Optional[str]
    transformed_dimensionality: Optional[int]
    scheduler_total_trials: Optional[int]
    scheduler_max_pending_trials: int
    arms_per_trial: int

    # Top level info
    product_surface: str
    launch_surface: str

    deployed_job_id: int
    trial_evaluation_identifier: Optional[str]

    # Miscellaneous product info
    is_manual_generation_strategy: bool
    warm_started_from: Optional[str]
    num_custom_trials: int
    support_tier: str

    @classmethod
    def from_scheduler(
        cls,
        scheduler: Scheduler,
        unique_identifier: str,
        owner: str,
        product_surface: str,
        launch_surface: str,
        deployed_job_id: int,
        trial_evaluation_identifier: Optional[str],
        is_manual_generation_strategy: bool,
        warm_started_from: Optional[str],
        num_custom_trials: int,
        support_tier: str,
    ) -> OptimizationCreatedRecord:
        scheduler_created_record = SchedulerCreatedRecord.from_scheduler(
            scheduler=scheduler
        )

        experiment_created_record = scheduler_created_record.experiment_created_record
        generation_strategy_created_record = (
            scheduler_created_record.generation_strategy_created_record
        )

        return cls(
            experiment_name=experiment_created_record.experiment_name,
            experiment_type=experiment_created_record.experiment_type,
            num_continuous_range_parameters=(
                experiment_created_record.num_continuous_range_parameters
            ),
            num_int_range_parameters_small=(
                experiment_created_record.num_int_range_parameters_small
            ),
            num_int_range_parameters_medium=(
                experiment_created_record.num_int_range_parameters_medium
            ),
            num_int_range_parameters_large=(
                experiment_created_record.num_int_range_parameters_large
            ),
            num_log_scale_range_parameters=(
                experiment_created_record.num_log_scale_range_parameters
            ),
            num_unordered_choice_parameters_small=(
                experiment_created_record.num_unordered_choice_parameters_small
            ),
            num_unordered_choice_parameters_medium=(
                experiment_created_record.num_unordered_choice_parameters_medium
            ),
            num_unordered_choice_parameters_large=(
                experiment_created_record.num_unordered_choice_parameters_large
            ),
            num_fixed_parameters=experiment_created_record.num_fixed_parameters,
            dimensionality=experiment_created_record.dimensionality,
            hierarchical_tree_height=experiment_created_record.hierarchical_tree_height,
            num_parameter_constraints=(
                experiment_created_record.num_parameter_constraints
            ),
            num_objectives=experiment_created_record.num_objectives,
            num_tracking_metrics=experiment_created_record.num_tracking_metrics,
            num_outcome_constraints=experiment_created_record.num_outcome_constraints,
            num_map_metrics=experiment_created_record.num_map_metrics,
            metric_cls_to_quantity=experiment_created_record.metric_cls_to_quantity,
            runner_cls=experiment_created_record.runner_cls,
            generation_strategy_name=(
                generation_strategy_created_record.generation_strategy_name
            ),
            num_requested_initialization_trials=(
                generation_strategy_created_record.num_requested_initialization_trials
            ),
            num_requested_bayesopt_trials=(
                generation_strategy_created_record.num_requested_bayesopt_trials
            ),
            num_requested_other_trials=(
                generation_strategy_created_record.num_requested_other_trials
            ),
            max_parallelism=generation_strategy_created_record.max_parallelism,
            early_stopping_strategy_cls=(
                scheduler_created_record.early_stopping_strategy_cls
            ),
            global_stopping_strategy_cls=(
                scheduler_created_record.global_stopping_strategy_cls
            ),
            transformed_dimensionality=(
                scheduler_created_record.transformed_dimensionality
            ),
            scheduler_total_trials=scheduler_created_record.scheduler_total_trials,
            scheduler_max_pending_trials=(
                scheduler_created_record.scheduler_max_pending_trials
            ),
            arms_per_trial=scheduler_created_record.arms_per_trial,
            unique_identifier=unique_identifier,
            owner=owner,
            product_surface=product_surface,
            launch_surface=launch_surface,
            deployed_job_id=deployed_job_id,
            trial_evaluation_identifier=trial_evaluation_identifier,
            is_manual_generation_strategy=is_manual_generation_strategy,
            warm_started_from=warm_started_from,
            num_custom_trials=num_custom_trials,
            support_tier=support_tier,
        )

    @classmethod
    def from_ax_client(
        cls,
        ax_client: AxClient,
        unique_identifier: str,
        owner: str,
        product_surface: str,
        launch_surface: str,
        deployed_job_id: int,
        trial_evaluation_identifier: Optional[str],
        is_manual_generation_strategy: bool,
        warm_started_from: Optional[str],
        num_custom_trials: int,
    ) -> OptimizationCreatedRecord:
        ax_client_created_record = AxClientCreatedRecord.from_ax_client(
            ax_client=ax_client
        )

        experiment_created_record = ax_client_created_record.experiment_created_record
        generation_strategy_created_record = (
            ax_client_created_record.generation_strategy_created_record
        )

        return cls(
            experiment_name=experiment_created_record.experiment_name,
            experiment_type=experiment_created_record.experiment_type,
            num_continuous_range_parameters=(
                experiment_created_record.num_continuous_range_parameters
            ),
            num_int_range_parameters_small=(
                experiment_created_record.num_int_range_parameters_small
            ),
            num_int_range_parameters_medium=(
                experiment_created_record.num_int_range_parameters_medium
            ),
            num_int_range_parameters_large=(
                experiment_created_record.num_int_range_parameters_large
            ),
            num_log_scale_range_parameters=(
                experiment_created_record.num_log_scale_range_parameters
            ),
            num_unordered_choice_parameters_small=(
                experiment_created_record.num_unordered_choice_parameters_small
            ),
            num_unordered_choice_parameters_medium=(
                experiment_created_record.num_unordered_choice_parameters_medium
            ),
            num_unordered_choice_parameters_large=(
                experiment_created_record.num_unordered_choice_parameters_large
            ),
            num_fixed_parameters=experiment_created_record.num_fixed_parameters,
            dimensionality=experiment_created_record.dimensionality,
            hierarchical_tree_height=(
                experiment_created_record.hierarchical_tree_height
            ),
            num_parameter_constraints=(
                experiment_created_record.num_parameter_constraints
            ),
            num_objectives=experiment_created_record.num_objectives,
            num_tracking_metrics=experiment_created_record.num_tracking_metrics,
            num_outcome_constraints=experiment_created_record.num_outcome_constraints,
            num_map_metrics=experiment_created_record.num_map_metrics,
            metric_cls_to_quantity=experiment_created_record.metric_cls_to_quantity,
            runner_cls=experiment_created_record.runner_cls,
            generation_strategy_name=(
                generation_strategy_created_record.generation_strategy_name
            ),
            num_requested_initialization_trials=(
                generation_strategy_created_record.num_requested_initialization_trials
            ),
            num_requested_bayesopt_trials=(
                generation_strategy_created_record.num_requested_bayesopt_trials
            ),
            num_requested_other_trials=(
                generation_strategy_created_record.num_requested_other_trials
            ),
            max_parallelism=generation_strategy_created_record.max_parallelism,
            early_stopping_strategy_cls=(
                ax_client_created_record.early_stopping_strategy_cls
            ),
            global_stopping_strategy_cls=(
                ax_client_created_record.global_stopping_strategy_cls
            ),
            transformed_dimensionality=(
                ax_client_created_record.transformed_dimensionality
            ),
            arms_per_trial=ax_client_created_record.arms_per_trial,
            unique_identifier=unique_identifier,
            owner=owner,
            product_surface=product_surface,
            launch_surface=launch_surface,
            deployed_job_id=deployed_job_id,
            trial_evaluation_identifier=trial_evaluation_identifier,
            is_manual_generation_strategy=is_manual_generation_strategy,
            warm_started_from=warm_started_from,
            num_custom_trials=num_custom_trials,
            # The following are not applicable for AxClient
            scheduler_total_trials=None,
            scheduler_max_pending_trials=-1,
            support_tier="",  # This support may be added in the future
        )

    @classmethod
    def from_experiment(
        cls,
        experiment: Experiment,
        generation_strategy: Optional[GenerationStrategy],
        unique_identifier: str,
        owner: str,
        product_surface: str,
        launch_surface: str,
        deployed_job_id: int,
        is_manual_generation_strategy: bool,
        num_custom_trials: int,
        warm_started_from: Optional[str] = None,
        arms_per_trial: Optional[int] = None,
        trial_evaluation_identifier: Optional[str] = None,
    ) -> OptimizationCreatedRecord:
        experiment_created_record = ExperimentCreatedRecord.from_experiment(
            experiment=experiment,
        )
        generation_strategy_created_record = (
            None
            if generation_strategy is None
            else (
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=generation_strategy,
                )
            )
        )
        arms_per_trial = -1 if arms_per_trial is None else arms_per_trial
        product_surface = (
            DEFAULT_PRODUCT_SURFACE if product_surface is None else product_surface
        )

        num_requested_initialization_trials = (
            None
            if generation_strategy_created_record is None
            else generation_strategy_created_record.num_requested_initialization_trials
        )
        return cls(
            experiment_name=experiment_created_record.experiment_name,
            experiment_type=experiment_created_record.experiment_type,
            num_continuous_range_parameters=(
                experiment_created_record.num_continuous_range_parameters
            ),
            num_int_range_parameters_small=(
                experiment_created_record.num_int_range_parameters_small
            ),
            num_int_range_parameters_medium=(
                experiment_created_record.num_int_range_parameters_medium
            ),
            num_int_range_parameters_large=(
                experiment_created_record.num_int_range_parameters_large
            ),
            num_log_scale_range_parameters=(
                experiment_created_record.num_log_scale_range_parameters
            ),
            num_unordered_choice_parameters_small=(
                experiment_created_record.num_unordered_choice_parameters_small
            ),
            num_unordered_choice_parameters_medium=(
                experiment_created_record.num_unordered_choice_parameters_medium
            ),
            num_unordered_choice_parameters_large=(
                experiment_created_record.num_unordered_choice_parameters_large
            ),
            num_fixed_parameters=experiment_created_record.num_fixed_parameters,
            dimensionality=experiment_created_record.dimensionality,
            hierarchical_tree_height=(
                experiment_created_record.hierarchical_tree_height
            ),
            num_parameter_constraints=(
                experiment_created_record.num_parameter_constraints
            ),
            num_objectives=experiment_created_record.num_objectives,
            num_tracking_metrics=experiment_created_record.num_tracking_metrics,
            num_outcome_constraints=experiment_created_record.num_outcome_constraints,
            num_map_metrics=experiment_created_record.num_map_metrics,
            metric_cls_to_quantity=experiment_created_record.metric_cls_to_quantity,
            runner_cls=experiment_created_record.runner_cls,
            generation_strategy_name=(
                None
                if generation_strategy_created_record is None
                else generation_strategy_created_record.generation_strategy_name
            ),
            num_requested_initialization_trials=num_requested_initialization_trials,
            num_requested_bayesopt_trials=(
                None
                if generation_strategy_created_record is None
                else generation_strategy_created_record.num_requested_bayesopt_trials
            ),
            num_requested_other_trials=(
                None
                if generation_strategy_created_record is None
                else generation_strategy_created_record.num_requested_other_trials
            ),
            max_parallelism=(
                None
                if generation_strategy_created_record is None
                else generation_strategy_created_record.max_parallelism
            ),
            early_stopping_strategy_cls=None,
            global_stopping_strategy_cls=None,
            transformed_dimensionality=(
                None
                if generation_strategy is None
                else _get_max_transformed_dimensionality(
                    search_space=experiment.search_space,
                    generation_strategy=generation_strategy,
                )
            ),
            arms_per_trial=arms_per_trial,
            unique_identifier=unique_identifier,
            owner=owner,
            product_surface=product_surface,
            launch_surface=launch_surface,
            deployed_job_id=deployed_job_id,
            trial_evaluation_identifier=trial_evaluation_identifier,
            is_manual_generation_strategy=is_manual_generation_strategy,
            warm_started_from=warm_started_from,
            num_custom_trials=num_custom_trials,
            # The following are not applicable for AxClient
            scheduler_total_trials=None,
            scheduler_max_pending_trials=-1,
            support_tier="",  # This support may be added in the future
        )


@dataclass(frozen=True)
class OptimizationCompletedRecord:
    """
    Record of the "Optimization" completion event. This can come from either an
    AxClient or a Scheduler. This Record is especially useful for logging Ax-backed
    optimization results to a tabular database (i.e. one row per Record)
    """

    unique_identifier: str

    # ExperimentCompletedRecord fields
    num_initialization_trials: int
    num_bayesopt_trials: int
    num_other_trials: int

    num_completed_trials: int
    num_failed_trials: int
    num_abandoned_trials: int
    num_early_stopped_trials: int

    total_fit_time: int
    total_gen_time: int

    # SchedulerCompletedRecord fields
    best_point_quality: float
    model_fit_quality: float
    model_std_quality: float
    model_fit_generalization: float
    model_std_generalization: float

    improvement_over_baseline: float

    num_metric_fetch_e_encountered: int
    num_trials_bad_due_to_err: int

    # TODO[mpolson64] Deprecate this field as it is redundant with unique_identifier
    deployed_job_id: Optional[int]

    # Miscellaneous deployment specific info
    estimated_early_stopping_savings: float
    estimated_global_stopping_savings: float

    @classmethod
    def from_scheduler(
        cls,
        scheduler: Scheduler,
        unique_identifier: str,
        deployed_job_id: Optional[int],
        estimated_early_stopping_savings: float,
        estimated_global_stopping_savings: float,
    ) -> OptimizationCompletedRecord:
        scheduler_completed_record = SchedulerCompletedRecord.from_scheduler(
            scheduler=scheduler
        )
        experiment_completed_record = (
            scheduler_completed_record.experiment_completed_record
        )

        return cls(
            num_initialization_trials=(
                experiment_completed_record.num_initialization_trials
            ),
            num_bayesopt_trials=experiment_completed_record.num_bayesopt_trials,
            num_other_trials=experiment_completed_record.num_other_trials,
            num_completed_trials=experiment_completed_record.num_completed_trials,
            num_failed_trials=experiment_completed_record.num_failed_trials,
            num_abandoned_trials=experiment_completed_record.num_abandoned_trials,
            num_early_stopped_trials=(
                experiment_completed_record.num_early_stopped_trials
            ),
            total_fit_time=experiment_completed_record.total_fit_time,
            total_gen_time=experiment_completed_record.total_gen_time,
            best_point_quality=scheduler_completed_record.best_point_quality,
            **_extract_model_fit_dict(scheduler_completed_record),
            improvement_over_baseline=(
                scheduler_completed_record.improvement_over_baseline
            ),
            num_metric_fetch_e_encountered=(
                scheduler_completed_record.num_metric_fetch_e_encountered
            ),
            num_trials_bad_due_to_err=(
                scheduler_completed_record.num_trials_bad_due_to_err
            ),
            unique_identifier=unique_identifier,
            deployed_job_id=deployed_job_id,
            estimated_early_stopping_savings=estimated_early_stopping_savings,
            estimated_global_stopping_savings=estimated_global_stopping_savings,
        )

    @classmethod
    def from_ax_client(
        cls,
        ax_client: AxClient,
        unique_identifier: str,
        deployed_job_id: Optional[int],
        estimated_early_stopping_savings: float,
        estimated_global_stopping_savings: float,
    ) -> OptimizationCompletedRecord:
        ax_client_completed_record = AxClientCompletedRecord.from_ax_client(
            ax_client=ax_client
        )
        experiment_completed_record = (
            ax_client_completed_record.experiment_completed_record
        )

        return cls(
            num_initialization_trials=(
                experiment_completed_record.num_initialization_trials
            ),
            num_bayesopt_trials=experiment_completed_record.num_bayesopt_trials,
            num_other_trials=experiment_completed_record.num_other_trials,
            num_completed_trials=experiment_completed_record.num_completed_trials,
            num_failed_trials=experiment_completed_record.num_failed_trials,
            num_abandoned_trials=experiment_completed_record.num_abandoned_trials,
            num_early_stopped_trials=(
                experiment_completed_record.num_early_stopped_trials
            ),
            total_fit_time=experiment_completed_record.total_fit_time,
            total_gen_time=experiment_completed_record.total_gen_time,
            best_point_quality=ax_client_completed_record.best_point_quality,
            **_extract_model_fit_dict(ax_client_completed_record),
            unique_identifier=unique_identifier,
            deployed_job_id=deployed_job_id,
            estimated_early_stopping_savings=estimated_early_stopping_savings,
            estimated_global_stopping_savings=estimated_global_stopping_savings,
            # The following are not applicable for AxClient
            improvement_over_baseline=float("nan"),
            num_metric_fetch_e_encountered=-1,
            num_trials_bad_due_to_err=-1,
        )


def _extract_model_fit_dict(
    completed_record: Union[SchedulerCompletedRecord, AxClientCompletedRecord],
) -> Dict[str, float]:
    model_fit_names = [
        "model_fit_quality",
        "model_std_quality",
        "model_fit_generalization",
        "model_std_generalization",
    ]
    return {n: getattr(completed_record, n) for n in model_fit_names}
