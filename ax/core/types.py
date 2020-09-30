#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from botorch.acquisition import AcquisitionFunction


TNumeric = Union[float, int]
TParamCounter = DefaultDict[int, int]
TParamValue = Optional[Union[str, bool, float, int]]
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

# 1-arm `Trial` evaluation data: {metric_name -> (mean, standard error)}}.
TTrialEvaluation = Dict[str, Tuple[float, Optional[float]]]

# 1-arm evaluation data with trace fidelities
TFidelityTrialEvaluation = List[Tuple[TParameterization, TTrialEvaluation]]

# Format for trasmitting evaluation data to Ax is either:
# 1) {metric_name -> (mean, standard error)} (TTrialEvaluation)
# 2) (mean, standard error) and we assume metric name == objective name
# 3) only the mean, and we assume metric name == objective name and standard error == 0
# 4) [({fidelity_param -> value}, {metric_name} -> (mean, standard error))]
TEvaluationOutcome = Union[
    TTrialEvaluation, Tuple[float, Optional[float]], float, TFidelityTrialEvaluation
]

TConfig = Dict[str, Union[int, float, str, AcquisitionFunction, Dict[str, Any]]]
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
