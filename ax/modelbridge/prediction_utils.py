#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge import ModelBridge


def predict_at_point(
    model: ModelBridge, obsf: ObservationFeatures, metric_names: Set[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Make a prediction at a point.

    Returns mean and standard deviation in format expected by plotting.

    Args:
        model: ModelBridge
        obsf: ObservationFeatures for which to predict
        metric_names: Limit predictions to these metrics.

    Returns:
        A tuple containing

        - Map from metric name to prediction.
        - Map from metric name to standard error.
    """
    y_hat = {}
    se_hat = {}
    f_pred, cov_pred = model.predict([obsf])
    for metric_name in f_pred:
        if metric_name in metric_names:
            y_hat[metric_name] = f_pred[metric_name][0]
            se_hat[metric_name] = np.sqrt(cov_pred[metric_name][metric_name][0])
    return y_hat, se_hat


def predict_by_features(
    model: ModelBridge,
    label_to_feature_dict: Dict[int, ObservationFeatures],
    metric_names: Set[str],
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Predict for given data points and model.

    Args:
        model: Model to be used for the prediction
        metric_names: Names of the metrics, for which to retrieve predictions.
        label_to_feature_dict: Mapping from an int label to
            a Parameterization. These data points are predicted.

    Returns:
        A mapping from an int label to a mapping of metric names to tuples
        of predicted metric mean and SEM, of form:
        { trial_index -> { metric_name: ( mean, SEM ) } }.
    """
    predictions_dict = {}  # Store predictions to return
    for label in label_to_feature_dict:
        try:
            y_hat, se_hat = predict_at_point(
                model=model,
                obsf=label_to_feature_dict[label],
                metric_names=metric_names,
            )
        except NotImplementedError:
            raise NotImplementedError(
                "The model associated with the current generation strategy "
                "step is not one that can be used for predicting values. "
                "For example, this may be the Sobol generator associated with the "
                "initialization step where quasi-random points are generated. "
                "Try again by calling the `AxClient.create_experiment()` "
                "method with the `choose_generation_strategy_kwargs="
                '{"num_initialization_trials": 0}` parameter if you are looking '
                "to use a generation strategy without an initialization step that "
                "proceeds straight to the Bayesian optimization step, but note "
                "that performance of Bayesian optimization can be suboptimal if "
                "search space is not sampled well in the initialization phase."
            )

        predictions_dict[label] = {
            metric: (
                y_hat[metric],
                se_hat[metric],
            )
            for metric in metric_names
        }

    return predictions_dict
