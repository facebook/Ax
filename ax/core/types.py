#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np


TNumeric = Union[float, int]
TParamCounter = DefaultDict[int, int]
TParamValue = Union[None, str, bool, float, int]
TParameterization = Dict[str, TParamValue]
TParamValueList = List[TParamValue]  # a parameterization without the keys
TContextStratum = Optional[Dict[str, Union[str, float, int]]]

TBounds = Optional[Tuple[np.ndarray, np.ndarray]]
TModelMean = Dict[str, List[float]]
TModelCov = Dict[str, Dict[str, List[float]]]
TModelPredict = Tuple[TModelMean, TModelCov]
# Model predictions for a single arm:
# ( { metric -> mean }, { metric -> { other_metric -> covariance } } ).
TModelPredictArm = Tuple[Dict[str, float], Optional[Dict[str, Dict[str, float]]]]

FloatLike = Union[int, float, np.floating, np.integer]
SingleMetricDataTuple = Tuple[FloatLike, Optional[FloatLike]]
SingleMetricData = Union[FloatLike, Tuple[FloatLike, Optional[FloatLike]]]
# 1-arm `Trial` evaluation data: {metric_name -> (mean, standard error)}}.
TTrialEvaluation = Dict[str, SingleMetricData]

# 1-arm evaluation data with trace fidelities
TFidelityTrialEvaluation = List[Tuple[TParameterization, TTrialEvaluation]]

# 1-arm evaluation data with arbitrary partial results
TMapDict = Dict[str, Hashable]
TMapTrialEvaluation = List[Tuple[TMapDict, TTrialEvaluation]]

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

TBucket = List[Dict[str, List[str]]]

TGenMetadata = Dict[str, Any]

# Model's metadata about a given candidate (or X).
TCandidateMetadata = Optional[Dict[str, Any]]


class ComparisonOp(enum.Enum):
    """Class for enumerating comparison operations."""

    GEQ: int = 0
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
