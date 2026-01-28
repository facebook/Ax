#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from enum import Enum

from ax.core.data import Data, DataRow
from ax.core.types import FloatLike, SingleMetricData, TEvaluationOutcome
from ax.exceptions.core import UserInputError


# -------------------- Data formatting utils. ---------------------
class DataType(Enum):
    """Deprecated. Used only for storage backwards compatibility."""

    DATA = 1
    MAP_DATA = 3


def _validate_and_extract_single_metric_data(
    dat: SingleMetricData,
) -> tuple[float, float | None]:
    error_message = (
        "Raw data does not conform to the expected structure. Expected either a"
        f" tuple of (mean, SEM) or a float, but got {dat}."
    )
    if isinstance(dat, tuple):
        if len(dat) != 2:
            raise UserInputError(error_message)
        mean, sem = dat
        if not isinstance(mean, FloatLike) or not (
            sem is None or isinstance(sem, FloatLike)
        ):
            raise UserInputError(error_message)
        return mean, sem
    if not isinstance(dat, FloatLike):
        raise UserInputError(error_message)
    return dat, None


def raw_evaluations_to_data(
    raw_data: Mapping[str, TEvaluationOutcome],
    metric_name_to_signature: Mapping[str, str],
    trial_index: int,
) -> Data:
    """Transforms evaluations into Ax Data.

    Each value in ``raw_data`` is one of the following:
    - ``TTrialEvaluation``: {metric_name -> (mean, SEM)}
    - ``TMapTrialEvaluation``: [(step, TTrialEvaluation)]
    - ``SingleMetricData``: (mean, SEM) or mean. This is a
        ``TTrialEvaluation`` that is missing a metric name and possibly an SEM.
        As long as there is only one element in ``metric_name_to_signature``,
        this will be assigned that one metric name. If the SEM is missing, it
        will be inferred to be None.

    Args:
        raw_data: Mapping from arm name to raw evaluations.
        metric_name_to_signature: Mapping of metric names to signatures used to
            transform raw data to evaluations.
        trial_index: Index of the trial, for which the evaluations are.
        data_type: An element of the ``DataType`` enum.
    """
    data_rows: list[DataRow] = []
    metric_names_seen: set[str] = set()

    for arm_name, evaluation in raw_data.items():
        # TTrialEvaluation case ({metric_name -> (mean, SEM) or metric_name -> mean})
        if isinstance(evaluation, dict):
            for metric_name, outcome in evaluation.items():
                mean, sem = _validate_and_extract_single_metric_data(dat=outcome)
                metric_names_seen.add(metric_name)
                data_rows.append(
                    DataRow(
                        trial_index=trial_index,
                        arm_name=arm_name,
                        metric_name=metric_name,
                        metric_signature=metric_name_to_signature.get(metric_name, ""),
                        mean=float(mean),
                        se=float(sem) if sem is not None else float("nan"),
                    )
                )
        elif isinstance(evaluation, list):
            # TMapTrialEvaluation case [(step, TTrialEvaluation)]
            for step, step_eval in evaluation:
                if not isinstance(step, FloatLike):
                    raise UserInputError(
                        "Raw data does not conform to the expected structure. Expected "
                        f"step to be a float, but got {step}."
                    )
                for metric_name, outcome in step_eval.items():
                    mean, sem = _validate_and_extract_single_metric_data(dat=outcome)
                    metric_names_seen.add(metric_name)
                    data_rows.append(
                        DataRow(
                            trial_index=trial_index,
                            arm_name=arm_name,
                            metric_name=metric_name,
                            metric_signature=metric_name_to_signature.get(
                                metric_name, ""
                            ),
                            mean=float(mean),
                            se=float(sem) if sem is not None else float("nan"),
                            step=float(step),
                        )
                    )
        # SingleMetricData case: (mean, SEM) or mean
        else:
            if len(metric_name_to_signature) != 1:
                raise UserInputError(
                    "Metric name must be provided in `raw_data` if there are "
                    "multiple metrics."
                )
            metric_name = next(iter(metric_name_to_signature.keys()))
            # pyre-fixme[6]: Incmopatible parameter type (Pyre doesn't know that
            # this is in fact a SingleMetricData)
            mean, sem = _validate_and_extract_single_metric_data(dat=evaluation)
            metric_names_seen.add(metric_name)
            data_rows.append(
                DataRow(
                    trial_index=trial_index,
                    arm_name=arm_name,
                    metric_name=metric_name,
                    metric_signature=metric_name_to_signature.get(metric_name, ""),
                    mean=float(mean),
                    se=float(sem) if sem is not None else float("nan"),
                )
            )

    metrics_missing_signatures = metric_names_seen - set(
        metric_name_to_signature.keys()
    )
    if len(metrics_missing_signatures) > 0:
        raise UserInputError(
            f"Metric(s) {metrics_missing_signatures} not found in "
            "metric_name_to_signature. Please provide a mapping for all metric "
            "names present in the evaluations to their respective signatures."
        )

    return Data(data_rows=data_rows)
