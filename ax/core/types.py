#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
SingleMetricData = Union[FloatLike, tuple[FloatLike, Optional[FloatLike]]]
# 1-arm `Trial` evaluation data: {metric_name -> (mean, standard error)}}.
TTrialEvaluation = Mapping[str, SingleMetricData]

#                     [      (step (float), TTrialEvaluation)]
TMapTrialEvaluation = Sequence[tuple[float, TTrialEvaluation]]

# Format for trasmitting evaluation data to Ax is either:
# 1) {metric_name -> (mean, standard error)} (TTrialEvaluation)
# 2) (mean, standard error) and we assume metric name == objective name
# 3) only the mean, and we assume metric name == objective name and standard error == 0

TEvaluationOutcome = Union[
    TTrialEvaluation,
    SingleMetricData,
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


def _is_floatlike(floatlike: Any) -> bool:
    return (
        isinstance(floatlike, float)
        or isinstance(floatlike, int)
        or isinstance(floatlike, np.floating)
        or isinstance(floatlike, np.integer)
    )


def validate_floatlike(floatlike: Any) -> None:
    if not _is_floatlike(floatlike=floatlike):
        raise TypeError(f"Expected FloatLike, found {floatlike}")


def validate_single_metric_data(data: Any) -> None:
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


def validate_trial_evaluation(evaluation: Mapping[Any, Any]) -> None:
    for key, value in evaluation.items():
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings in TTrialEvaluation, found {key}.")

        validate_single_metric_data(data=value)


def validate_param_value(param_value: Any) -> None:
    if not (
        isinstance(param_value, str)
        or isinstance(param_value, bool)
        or isinstance(param_value, float)
        or isinstance(param_value, int)
        or param_value is None
    ):
        raise TypeError(f"Expected None, bool, float, int, or str, found {param_value}")


def validate_parameterization(parameterization: Mapping[Any, Any]) -> None:
    for key, value in parameterization.items():
        if not isinstance(key, str):
            raise TypeError(f"Keys must be strings in TParameterization, found {key}.")

        validate_param_value(param_value=value)


def validate_step(step: float) -> None:
    if not _is_floatlike(floatlike=step):
        raise TypeError(
            f"Steps must be float-like in TMapTrialEvaluation; found {step}."
        )


def validate_map_trial_evaluation(evaluation: TMapTrialEvaluation) -> None:
    for step, trial_evaluation in evaluation:
        validate_step(step=step)
        validate_trial_evaluation(evaluation=trial_evaluation)


def validate_evaluation_outcome(outcome: Any) -> None:
    """Runtime validate that the supplied outcome has correct structure."""

    # TTrialEvaluation case
    if isinstance(outcome, dict):
        validate_trial_evaluation(evaluation=outcome)

    # TMapTrialEvaluation case
    elif isinstance(outcome, list):
        validate_map_trial_evaluation(evaluation=outcome)

    # SingleMetricData case
    else:
        validate_single_metric_data(data=outcome)
