# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from ax.service.ax_client import AxClient
from ax.telemetry.common import _get_max_transformed_dimensionality
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord


@dataclass(frozen=True)
class AxClientCreatedRecord:
    """
    Record of the AxClient creation event. This can be used for telemetry in settings
    where many AxClients are being created either manually or programatically. In
    order to facilitate easy serialization only include simple types: numbers, strings,
    bools, and None.
    """

    experiment_created_record: ExperimentCreatedRecord
    generation_strategy_created_record: GenerationStrategyCreatedRecord

    arms_per_trial: int
    early_stopping_strategy_cls: Optional[str]
    global_stopping_strategy_cls: Optional[str]

    # Dimensionality of transformed SearchSpace can often be much higher due to one-hot
    # encoding of unordered ChoiceParameters
    transformed_dimensionality: int

    @classmethod
    def from_ax_client(cls, ax_client: AxClient) -> AxClientCreatedRecord:
        # Some AxClients may implement `batch_size`, those that do not use
        # one trial arms.
        if getattr(ax_client, "batch_size", None) is not None:
            # pyre-fixme[16] `AxClient` has no attribute `batch_size`
            arms_per_trial = ax_client.batch_size
        else:
            arms_per_trial = 1

        return cls(
            experiment_created_record=ExperimentCreatedRecord.from_experiment(
                experiment=ax_client.experiment
            ),
            generation_strategy_created_record=(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=ax_client.generation_strategy
                )
            ),
            arms_per_trial=arms_per_trial,
            early_stopping_strategy_cls=(
                None
                if ax_client.early_stopping_strategy is None
                else ax_client.early_stopping_strategy.__class__.__name__
            ),
            global_stopping_strategy_cls=(
                None
                if ax_client.global_stopping_strategy is None
                else ax_client.global_stopping_strategy.__class__.__name__
            ),
            transformed_dimensionality=_get_max_transformed_dimensionality(
                search_space=ax_client.experiment.search_space,
                generation_strategy=ax_client.generation_strategy,
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
class AxClientCompletedRecord:
    """
    Record of the AxClient completion event. This will have information only available
    after the optimization has completed.
    """

    experiment_completed_record: ExperimentCompletedRecord

    best_point_quality: float
    model_fit_quality: float
    model_std_quality: float
    model_fit_generalization: float
    model_std_generalization: float

    @classmethod
    def from_ax_client(cls, ax_client: AxClient) -> AxClientCompletedRecord:
        return cls(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=ax_client.experiment
            ),
            best_point_quality=float("nan"),  # TODO[T147907632]
            model_fit_quality=float("nan"),  # TODO[T147907632]
            model_std_quality=float("nan"),
            model_fit_generalization=float("nan"),
            model_std_generalization=float("nan"),
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
