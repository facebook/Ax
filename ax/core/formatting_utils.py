#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import Enum
from typing import cast

import numpy as np
from ax.core.data import Data
from ax.core.map_data import MapData
from ax.core.types import (
    TEvaluationOutcome,
    TMapTrialEvaluation,
    TTrialEvaluation,
    validate_evaluation_outcome,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.typeutils_nonnative import numpy_type_to_python_type


# -------------------- Data formatting utils. ---------------------


class DataType(Enum):
    DATA = 1
    MAP_DATA = 3


DATA_TYPE_LOOKUP: dict[DataType, type[Data]] = {
    DataType.DATA: Data,
    DataType.MAP_DATA: MapData,
}


def raw_data_to_evaluation(
    raw_data: TEvaluationOutcome,
    metric_names: list[str],
) -> TEvaluationOutcome:
    """Format the trial evaluation data to a standard `TTrialEvaluation`
    (mapping from metric names to a tuple of mean and SEM) representation, or
    to a TMapTrialEvaluation.

    Note: this function expects raw_data to be data for a `Trial`, not a
    `BatchedTrial`.
    """
    if isinstance(raw_data, dict):
        if any(isinstance(x, dict) for x in raw_data.values()):
            raise UserInputError("Raw data is expected to be just for one arm.")
        for metric_name, dat in raw_data.items():
            if not isinstance(dat, tuple):
                if not isinstance(dat, (float, int)):
                    raise UserInputError(
                        "Raw data for an arm is expected to either be a tuple of "
                        "numerical mean and SEM or just a numerical mean. "
                        f"Got: {dat} for metric '{metric_name}'."
                    )
                raw_data[metric_name] = (float(dat), None)
        return raw_data
    try:
        validate_evaluation_outcome(outcome=raw_data)
    except Exception as e:
        raise UserInputError(
            "Raw data does not conform to the expected structure. For simple "
            "evaluations of one or more metrics, `raw_data` is expected to be "
            "a dictionary of the form `{metric_name -> mean}` or `{metric_name "
            "-> (mean, SEM)}`. For fidelity or mapping (e.g., early stopping) "
            "evaluation, the expected format is `[(fidelities, {metric_name -> "
            "(mean, SEM)})]` or `[({mapping_key, mapping_value}, {metric_name -> "
            "(mean, SEM)})]`."
            f"Received {raw_data=}. Original validation error: {e}."
        )
    if isinstance(raw_data, list):
        validate_evaluation_outcome(raw_data)
        return raw_data
    elif len(metric_names) > 1:
        raise UserInputError(
            "Raw data must be a dictionary of metric names to mean "
            "for experiments with multiple metrics attached. "
            f"Got {raw_data=} for {metric_names=}."
        )
    elif isinstance(raw_data, tuple):
        return {metric_names[0]: raw_data}
    elif isinstance(raw_data, (float, int)):
        return {metric_names[0]: (raw_data, None)}
    elif isinstance(raw_data, (np.float32, np.float64, np.int32, np.int64)):
        return {metric_names[0]: (numpy_type_to_python_type(raw_data), None)}
    else:
        raise UserInputError(
            "Raw data has an invalid type. The data must either be in the form "
            "of a dictionary of metric names to mean, sem tuples, "
            "or a single mean, sem tuple, or a single mean."
        )


def data_and_evaluations_from_raw_data(
    raw_data: dict[str, TEvaluationOutcome],
    metric_names: list[str],
    trial_index: int,
    sample_sizes: dict[str, int],
    data_type: DataType,
    start_time: int | str | None = None,
    end_time: int | str | None = None,
) -> tuple[dict[str, TEvaluationOutcome], Data]:
    """Transforms evaluations into Ax Data.

    Each evaluation is either a trial evaluation: {metric_name -> (mean, SEM)}
    or a fidelity trial evaluation for multi-fidelity optimizations:
    [(fidelities, {metric_name -> (mean, SEM)})].

    Args:
        raw_data: Mapping from arm name to raw_data.
        metric_names: Names of metrics used to transform raw data to evaluations.
        trial_index: Index of the trial, for which the evaluations are.
        sample_sizes: Number of samples collected for each arm, may be empty
            if unavailable.
        start_time: Optional start time of run of the trial that produced this
            data, in milliseconds or iso format.  Milliseconds will eventually be
            converted to iso format because iso format automatically works with the
            pandas column type `Timestamp`.
        end_time: Optional end time of run of the trial that produced this
            data, in milliseconds or iso format.  Milliseconds will eventually be
            converted to iso format because iso format automatically works with the
            pandas column type `Timestamp`.
    """
    evaluations = {
        arm_name: raw_data_to_evaluation(
            raw_data=raw_data[arm_name],
            metric_names=metric_names,
        )
        for arm_name in raw_data
    }
    if all(isinstance(evaluations[x], dict) for x in evaluations.keys()):
        if data_type is DataType.MAP_DATA:
            raise UserInputError(
                "The format of the `raw_data` is not compatible with `MapData`. "
                "Possible cause: Did you set default data type to `MapData`, e.g., "
                "for early stopping, but forgot to provide the `raw_data` "
                "in the form of `[(fidelities, {metric_name -> (mean, SEM)})]` or "
                "`[({mapping_key, mapping_value}, {metric_name -> (mean, SEM)})]`? "
                f"Received: {raw_data=}"
            )
        # All evaluations are no-fidelity evaluations.
        data = Data.from_evaluations(
            evaluations=cast(dict[str, TTrialEvaluation], evaluations),
            trial_index=trial_index,
            sample_sizes=sample_sizes,
            start_time=start_time,
            end_time=end_time,
        )
    elif all(isinstance(evaluations[x], list) for x in evaluations.keys()):
        if data_type is DataType.DATA:
            raise UserInputError(
                "The format of the `raw_data` is not compatible with `Data`. "
                "Possible cause: Did you provide data for multi-fidelity evaluations, "
                "e.g., for early stopping, but forgot to set the default data type "
                f"to `MapData`? Received: {raw_data=}"
            )
        # All evaluations are map evaluations.
        data = MapData.from_map_evaluations(
            evaluations=cast(dict[str, TMapTrialEvaluation], evaluations),
            trial_index=trial_index,
        )
    else:
        raise ValueError(
            "Evaluations included a mixture of no-fidelity and with-fidelity "
            "evaluations, which is not currently supported."
        )
    return evaluations, data
