#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping, Sequence
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
    raw_data: TEvaluationOutcome, metric_names: Sequence[str]
) -> TEvaluationOutcome:
    """Format the trial evaluation data to a standard `TTrialEvaluation`
    (mapping from metric names to a tuple of mean and SEM) representation, or
    to a TMapTrialEvaluation.

    Note: this function expects raw_data to be data for a `Trial`, not a
    `BatchedTrial`.
    """
    # TTrialEvaluation case
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
            "-> (mean, SEM)}`. For mapping (e.g., early stopping) evaluation, "
            "the expected format is `[({mapping_key, mapping_value}, "
            "{metric_name  -> (mean, SEM)})]`."
            f"Received {raw_data=}. Original validation error: {e}."
        )
    # TMapTrialEvaluation case
    if isinstance(raw_data, list):
        validate_evaluation_outcome(raw_data)
        return raw_data
    elif len(metric_names) > 1:
        raise UserInputError(
            "Raw data must be a dictionary of metric names to mean "
            "for experiments with multiple metrics attached. "
            f"Got {raw_data=} for {metric_names=}."
        )
    # SingleMetricData tuple case
    elif isinstance(raw_data, tuple):
        return {metric_names[0]: raw_data}
    # SingleMetricData Python scalar case
    elif isinstance(raw_data, (float, int)):
        return {metric_names[0]: (raw_data, None)}
    # SingleMetricData Numpy scalar case
    elif isinstance(raw_data, (np.float32, np.float64, np.int32, np.int64)):
        return {metric_names[0]: (numpy_type_to_python_type(raw_data), None)}
    else:
        raise UserInputError(
            "Raw data has an invalid type. The data must either be in the form "
            "of a dictionary of metric names to mean, sem tuples, "
            "or a single mean, sem tuple, or a single mean."
        )


def data_and_evaluations_from_raw_data(
    raw_data: Mapping[str, TEvaluationOutcome],
    metric_name_to_signature: Mapping[str, str],
    trial_index: int,
    data_type: DataType,
) -> tuple[dict[str, TEvaluationOutcome], Data]:
    """Transforms evaluations into Ax Data.

    Each evaluation is either a trial evaluation: {metric_name -> (mean, SEM)}
    or a fidelity trial evaluation for multi-fidelity optimizations:
    [(fidelities, {metric_name -> (mean, SEM)})].

    Args:
        raw_data: Mapping from arm name to raw_data.
        metric_name_to_signature: Mapping of metric names to signatures used to
            transform raw data to evaluations.
        trial_index: Index of the trial, for which the evaluations are.
        data_type: An element of the ``DataType`` enum.

    """
    evaluations = {
        arm_name: raw_data_to_evaluation(
            raw_data=raw_data[arm_name],
            metric_names=list(metric_name_to_signature.keys()),
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
            metric_name_to_signature=metric_name_to_signature,
            trial_index=trial_index,
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
            metric_name_to_signature=metric_name_to_signature,
        )
    else:
        raise ValueError(
            "Evaluations included a mixture of no-fidelity and with-fidelity "
            "evaluations, which is not currently supported."
        )
    return evaluations, data
