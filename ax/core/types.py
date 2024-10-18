#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, Optional, Union

import numpy as np


TNumeric = Union[float, int]
TParamCounter = defaultdict[int, int]
TParamValue = Union[None, str, bool, float, int]
TParameterization = dict[str, TParamValue]
TParamValueList = list[TParamValue]  # a parameterization without the keys
TContextStratum = Optional[dict[str, Union[str, float, int]]]

# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
TBounds = Optional[tuple[np.ndarray, np.ndarray]]
TModelMean = dict[str, list[float]]
TModelCov = dict[str, dict[str, list[float]]]
TModelPredict = tuple[TModelMean, TModelCov]
# Model predictions for a single arm:
# ( { metric -> mean }, { metric -> { other_metric -> covariance } } ).
TModelPredictArm = tuple[dict[str, float], Optional[dict[str, dict[str, float]]]]

# pyre-fixme[24]: Generic type `np.floating` expects 1 type parameter.
# pyre-fixme[24]: Generic type `np.integer` expects 1 type parameter.
FloatLike = Union[int, float, np.floating, np.integer]
SingleMetricDataTuple = tuple[FloatLike, Optional[FloatLike]]
SingleMetricData = Union[FloatLike, tuple[FloatLike, Optional[FloatLike]]]
# 1-arm `Trial` evaluation data: {metric_name -> (mean, standard error)}}.
TTrialEvaluation = dict[str, SingleMetricData]

# 1-arm evaluation data with trace fidelities
TFidelityTrialEvaluation = list[tuple[TParameterization, TTrialEvaluation]]

# 1-arm evaluation data with arbitrary partial results
TMapDict = dict[str, Hashable]
TMapTrialEvaluation = list[tuple[TMapDict, TTrialEvaluation]]

# Format for trasmitting evaluation data to Ax is either:
# 1) {metric_name -> (mean, standard error)} (TTrialEvaluation)
# 2) (mean, standard error) and we assume metric name == objective name
# 3) only the mean, and we assume metric name == objective name and standard error == 0
# 4) [({fidelity_param -> value}, {metric_name} -> (mean, standard error))]

TEvaluationOutcome = Union[
    TTrialEvaluation,
    SingleMetricData,
    TFidelityTrialEvaluation,
    TMapTrialEvaluation,
]
TEvaluationFunction = Union[
    Callable[[TParameterization], TEvaluationOutcome],
    Callable[[TParameterization, Optional[float]], TEvaluationOutcome],
]

TBucket = list[dict[str, list[str]]]

TGenMetadata = dict[str, Any]

# Model's metadata about a given candidate (or X).
TCandidateMetadata = Optional[dict[str, Any]]


class ComparisonOp(enum.Enum):
    """Class for enumerating comparison operations."""

    # pyre-fixme[35]: Target cannot be annotated.
    GEQ: int = 0
    # pyre-fixme[35]: Target cannot be annotated.
    LEQ: int = 1


def merge_model_predict(
    predict: TModelPredict, predict_append: TModelPredict
) -> TModelPredict:
    """Append model predictions to an existing set of model predictions.

    TModelPredict is of the form:
        {metric_name: [mean1, mean2, ...],
        {metric_name: {metric_name: [var1, var2, ...]}})

    This will append the predictions

    Args:
        predict: Initial set of predictions.
        other_predict: Predictions to be appended.

    Returns:
        TModelPredict with the new predictions appended.
    """
    mu, cov = predict
    mu_append, cov_append = predict_append
    if len(mu) != len(mu_append) or len(cov) != len(cov_append):
        raise ValueError("Both sets of model predictions must have the same metrics")

    # Iterate down to the list level and simply add.
    for metric_name, metric_values in mu.items():
        mu[metric_name] = metric_values + mu_append[metric_name]

    for metric_name, co_cov in cov.items():
        for co_metric_name, cov_values in co_cov.items():
            cov[metric_name][co_metric_name] = (
                cov_values + cov_append[metric_name][co_metric_name]
            )
    return mu, cov


def validate_floatlike(floatlike: FloatLike) -> None:
    if not (
        isinstance(floatlike, float)
        or isinstance(floatlike, int)
        or isinstance(floatlike, np.floating)
        or isinstance(floatlike, np.integer)
    ):
        raise TypeError(f"Expected FloatLike, found {floatlike}")


def validate_single_metric_data(data: SingleMetricData) -> None:
    if isinstance(data, tuple):
        if len(data) != 2:
            raise TypeError(
                f"Tuple-valued SingleMetricData must have len == 2, found {data}"
            )

        mean, sem = data
        validate_floatlike(floatlike=mean)

        if sem is not None:
            validate_floatlike(floatlike=sem)

    else:
        validate_floatlike(floatlike=data)


def validate_trial_evaluation(evaluation: TTrialEvaluation) -> None:
    for key, value in evaluation.items():
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings in TTrialEvaluation, found {key}.")

        validate_single_metric_data(data=value)


def validate_param_value(param_value: TParamValue) -> None:
    if not (
        isinstance(param_value, str)
        or isinstance(param_value, bool)
        or isinstance(param_value, float)
        or isinstance(param_value, int)
        or param_value is None
    ):
        raise TypeError(f"Expected None, bool, float, int, or str, found {param_value}")


def validate_parameterization(parameterization: TParameterization) -> None:
    for key, value in parameterization.items():
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings in TParameterization, found {key}.")

        validate_param_value(param_value=value)


def validate_map_dict(map_dict: TMapDict) -> None:
    for key, value in map_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings in TMapDict, found {key}.")

        if not isinstance(value, Hashable):
            raise TypeError(f"Values must be Hashable in TMapDict, found {value}.")


def validate_fidelity_trial_evaluation(evaluation: TFidelityTrialEvaluation) -> None:
    for parameterization, trial_evaluation in evaluation:
        validate_parameterization(parameterization=parameterization)
        validate_trial_evaluation(evaluation=trial_evaluation)


def validate_map_trial_evaluation(evaluation: TMapTrialEvaluation) -> None:
    for map_dict, trial_evaluation in evaluation:
        validate_map_dict(map_dict=map_dict)
        validate_trial_evaluation(evaluation=trial_evaluation)


def validate_evaluation_outcome(outcome: TEvaluationOutcome) -> None:
    """Runtime validate that the supplied outcome has correct structure."""

    if isinstance(outcome, dict):
        # Check if outcome is TTrialEvaluation
        validate_trial_evaluation(evaluation=outcome)

    elif isinstance(outcome, list):
        # Check if outcome is TFidelityTrialEvaluation or TMapTrialEvaluation
        try:
            validate_fidelity_trial_evaluation(evaluation=outcome)  # pyre-ignore[6]
        except Exception:
            try:
                validate_map_trial_evaluation(evaluation=outcome)  # pyre-ignore[6]
            except Exception:
                raise TypeError(
                    "Expected either TFidelityTrialEvaluation or TMapTrialEvaluation, "
                    f"found {outcome}"
                )

    else:
        # Check if outcome is SingleMetricData
        validate_single_metric_data(data=outcome)
