#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge import ModelBridge


def predict_at_point(
    model: ModelBridge,
    obsf: ObservationFeatures,
    metric_names: set[str],
    scalarized_metric_config: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Make a prediction at a point.

    Returns mean and standard deviation in format expected by plotting.

    Args:
        model: ModelBridge
        obsf: ObservationFeatures for which to predict
        metric_names: Limit predictions to these metrics.
        scalarized_metric_config: An optional list of dicts specifying how to aggregate
            multiple metrics into a single scalarized metric. For each dict, the key is
            the name of the new scalarized metric, and the value is a dictionary mapping
            each metric to its weight. e.g.
            {"name": "metric1:agg", "weight": {"metric1_c1": 0.5, "metric1_c2": 0.5}}.

    Returns:
        A tuple containing

        - Map from metric name to prediction.
        - Map from metric name to standard error.
    """
    f_pred, cov_pred = model.predict([obsf])
    mean_pred_dict = {metric_name: pred[0] for metric_name, pred in f_pred.items()}
    cov_pred_dict = {}
    for metric_name1, pred_dict in cov_pred.items():
        cov_pred_dict[metric_name1] = {}
        for metric_name2, pred in pred_dict.items():
            cov_pred_dict[metric_name1][metric_name2] = pred[0]

    y_hat = {}
    se_hat = {}
    for metric_name in f_pred:
        if metric_name in metric_names:
            y_hat[metric_name] = mean_pred_dict[metric_name]
            se_hat[metric_name] = np.sqrt(cov_pred_dict[metric_name][metric_name])
    if scalarized_metric_config is not None:
        for agg_metric in scalarized_metric_config:
            agg_metric_name = agg_metric["name"]
            if agg_metric_name in metric_names:
                agg_metric_weight_dict = agg_metric["weight"]
                pred_mean, pred_var = _compute_scalarized_outcome(
                    mean_dict=mean_pred_dict,
                    cov_dict=cov_pred_dict,
                    agg_metric_weight_dict=agg_metric_weight_dict,
                )
                y_hat[agg_metric_name] = pred_mean
                se_hat[agg_metric_name] = np.sqrt(pred_var)
    return y_hat, se_hat


def predict_by_features(
    model: ModelBridge,
    label_to_feature_dict: dict[int, ObservationFeatures],
    metric_names: set[str],
) -> dict[int, dict[str, tuple[float, float]]]:
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


def _compute_scalarized_outcome(
    mean_dict: dict[str, float],
    cov_dict: dict[str, dict[str, float]],
    agg_metric_weight_dict: dict[str, float],
) -> tuple[float, float]:
    """Compute the mean and variance of a scalarized outcome.

    Args:
        mean_dict: Dictionary of means of individual metrics. e.g.
            {"metric1": 0.1, "metric2": 0.2}
        cov_dict: Dictionary of covariances of each metric pair. e.g.
            {"metric1": {"metric1": 0.25, "metric2": 0.0}, "metric2": {"metric1": 0.0,
            "metric2": 0.1}}
        agg_metric_weight_dict. Dictionary of scalarization weights of the scalarized
            outcome. e.g. {"name": "metric1:agg", "weight": {"metric1_c1": 0.5,
             "metric1_c2": 0.5}}
    """
    pred_mean = 0
    pred_var = 0
    component_metrics = list(agg_metric_weight_dict.keys())
    for i, metric_name in enumerate(component_metrics):
        weight = agg_metric_weight_dict[metric_name]
        pred_mean += weight * mean_dict[metric_name]
        pred_var += (weight**2) * cov_dict[metric_name][metric_name]
        # include cross-metric covariance
        for metric_name2 in component_metrics[(i + 1) :]:
            weight2 = agg_metric_weight_dict[metric_name2]
            pred_var += 2 * weight * weight2 * cov_dict[metric_name][metric_name2]
    return pred_mean, pred_var
