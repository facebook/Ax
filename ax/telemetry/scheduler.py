# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from warnings import warn

import numpy as np
from ax.modelbridge.cross_validation import compute_model_fit_metrics_from_modelbridge

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
                    generation_strategy=scheduler.standard_generation_strategy
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
                generation_strategy=scheduler.standard_generation_strategy,
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
    model_std_quality: float
    model_fit_generalization: float
    model_std_generalization: float

    improvement_over_baseline: float

    num_metric_fetch_e_encountered: int
    num_trials_bad_due_to_err: int

    @classmethod
    def from_scheduler(cls, scheduler: Scheduler) -> SchedulerCompletedRecord:
        try:
            model_bridge = get_fitted_model_bridge(scheduler)
            model_fit_dict = compute_model_fit_metrics_from_modelbridge(
                model_bridge=model_bridge,
                experiment=scheduler.experiment,
                generalization=False,
                untransform=False,
            )
            model_fit_quality = _model_fit_metric(model_fit_dict)
            # similar for uncertainty quantification, but distance from 1 matters
            std = list(model_fit_dict["std_of_the_standardized_error"].values())
            model_std_quality = _model_std_quality(np.array(std))

            # generalization metrics
            model_gen_dict = compute_model_fit_metrics_from_modelbridge(
                model_bridge=model_bridge,
                experiment=scheduler.experiment,
                generalization=True,
                untransform=False,
            )
            model_fit_generalization = _model_fit_metric(model_gen_dict)
            gen_std = list(model_gen_dict["std_of_the_standardized_error"].values())
            model_std_generalization = _model_std_quality(np.array(gen_std))

        except Exception as e:
            warn("Encountered exception in computing model fit quality: " + str(e))
            model_fit_quality = float("nan")
            model_std_quality = float("nan")
            model_fit_generalization = float("nan")
            model_std_generalization = float("nan")

        try:
            improvement_over_baseline = scheduler.get_improvement_over_baseline()
        except Exception as e:
            warn(
                "Encountered exception in computing improvement over baseline: "
                + str(e)
            )
            improvement_over_baseline = float("nan")

        return cls(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=scheduler.experiment
            ),
            best_point_quality=float("nan"),  # TODO[T147907632]
            model_fit_quality=model_fit_quality,
            model_std_quality=model_std_quality,
            model_fit_generalization=model_fit_generalization,
            model_std_generalization=model_std_generalization,
            improvement_over_baseline=improvement_over_baseline,
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


def _model_fit_metric(metric_dict: Dict[str, Dict[str, float]]) -> float:
    # We'd ideally log the entire `model_fit_dict` as a single model fit metric
    # can't capture the nuances of multiple experimental metrics, but this might
    # lead to database performance issues. So instead, we take the worst
    # coefficient of determination as model fit quality and store the full data
    # in Manifold (TODO).
    return min(metric_dict["coefficient_of_determination"].values())


def _model_std_quality(std: np.ndarray) -> float:
    """Quantifies quality of the model uncertainty. A value of one means the
    uncertainty is perfectly predictive of the true standard deviation of the error.
    Values larger than one indicate over-estimation and negative values indicate
    under-estimation of the true standard deviation of the error. In particular, a value
    of 2 (resp. 1 / 2) represents an over-estimation (resp. under-estimation) of the
    true standard deviation of the error by a factor of 2.

    Args:
        std: The standard deviation of the standardized error.

    Returns:
        The factor corresponding to the worst over- or under-estimation factor of the
        standard deviation of the error among all experimentally observed metrics.
    """
    max_std, min_std = np.max(std), np.min(std)
    # comparing worst over-estimation factor with worst under-estimation factor
    inv_model_std_quality = max_std if max_std > 1 / min_std else min_std
    # reciprocal so that values greater than one indicate over-estimation and
    # values smaller than indicate underestimation of the uncertainty.
    return 1 / inv_model_std_quality
