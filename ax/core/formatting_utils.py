#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from enum import Enum

import pandas as pd
from ax.core.data import Data, MAP_KEY
from ax.core.map_data import MapData
from ax.core.types import FloatLike, SingleMetricData, TEvaluationOutcome
from ax.exceptions.core import UserInputError


# -------------------- Data formatting utils. ---------------------


class DataType(Enum):
    DATA = 1
    MAP_DATA = 3


DATA_TYPE_LOOKUP: dict[DataType, type[Data]] = {
    DataType.DATA: Data,
    DataType.MAP_DATA: MapData,
}


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
    data_type: DataType,
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
    records = []
    for arm_name, evaluation in raw_data.items():
        # TTrialEvaluation case ({metric_name -> (mean, SEM) or metric_name -> mean})
        if isinstance(evaluation, dict):
            if data_type is DataType.MAP_DATA:
                raise UserInputError(
                    "The format of the `raw_data` is not compatible with `MapData`. "
                    f"Received: {raw_data=}"
                )
            for metric_name, outcome in evaluation.items():
                mean, sem = _validate_and_extract_single_metric_data(dat=outcome)
                records.append(
                    {
                        "arm_name": arm_name,
                        "metric_name": metric_name,
                        "mean": mean,
                        "sem": sem,
                    }
                )
        elif isinstance(evaluation, list):
            # TMapTrialEvaluation case [(step, TTrialEvaluation)]
            if data_type is DataType.DATA:
                raise UserInputError(
                    "The format of the `raw_data` is not compatible with `Data`. "
                    f"Received: {raw_data=}"
                )
            for step, step_eval in evaluation:
                if not isinstance(step, FloatLike):
                    raise UserInputError(
                        "Raw data does not conform to the expected structure. Expected "
                        f"step to be a float, but got {step}."
                    )
                for metric_name, outcome in step_eval.items():
                    mean, sem = _validate_and_extract_single_metric_data(dat=outcome)
                    records.append(
                        {
                            "arm_name": arm_name,
                            "metric_name": metric_name,
                            "mean": mean,
                            "sem": sem,
                            MAP_KEY: step,
                        }
                    )
        # SingleMetricData case: (mean, SEM) or mean
        else:
            if data_type is DataType.MAP_DATA:
                raise UserInputError(
                    "The format of the `raw_data` is not compatible with `MapData`. "
                    f"Received: {raw_data=}"
                )
            if len(metric_name_to_signature) != 1:
                raise UserInputError(
                    "Metric name must be provided in `raw_data` if there are "
                    "multiple metrics."
                )
            metric_name = next(iter(metric_name_to_signature.keys()))
            # pyre-fixme[6]: Incmopatible parameter type (Pyre doesn't know that
            # this is in fact a SingleMetricData)
            mean, sem = _validate_and_extract_single_metric_data(dat=evaluation)
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": metric_name,
                    "mean": mean,
                    "sem": sem,
                }
            )

    df = pd.DataFrame.from_records(records)
    metrics_missing_signatures = set(df["metric_name"].unique()) - set(
        metric_name_to_signature.keys()
    )
    if len(metrics_missing_signatures) > 0:
        raise UserInputError(
            f"Metric(s) {metrics_missing_signatures} not found in "
            "metric_name_to_signature. Please provide a mapping for all metric "
            "names present in the evaluations to their respective signatures."
        )
    df["metric_signature"] = df["metric_name"].map(metric_name_to_signature)
    df["trial_index"] = trial_index

    if data_type == DataType.MAP_DATA:
        return MapData(df=df)
    return Data(df=df)
