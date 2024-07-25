#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Set

import numpy as np
import pandas as pd

from ax.modelbridge import ModelBridge
from ax.modelbridge.prediction_utils import predict_at_point

from ax.modelbridge.transforms.ivw import IVW


def get_plot_data_in_sample_arms_df(
    model: ModelBridge,
    metric_names: Set[str],
) -> pd.DataFrame:
    """Get in-sample arms from a model with observed and predicted values
    for specified metrics.

    Returns a dataframe in which repeated observations are merged
    with IVW (inverse variance weighting)

    Args:
        model: An instance of the model bridge.
        metric_names: Restrict predictions to these metrics. If None, uses all
            metrics in the model.

    Returns:
        A dataframe containing
        columns:
            {
                "arm_name": name of the arm in the cross validation result
                "metric_name": name of the observed/predicted metric
                "x": value observed for the metric for this arm
                "x_se": standard error of observed metric (0 for observations)
                "y": value predicted for the metric for this arm
                "y_se": standard error of predicted metric for this arm
                "arm_parameters": Parametrization of the arm
            }
    """
    observations = model.get_training_data()
    training_in_design: List[bool] = model.training_in_design

    # Merge multiple measurements within each Observation with IVW to get
    # un-modeled prediction
    observations = IVW(None, []).transform_observations(observations)

    # Create records for dict
    records = []

    for i, obs in enumerate(observations):
        # Extract raw measurement
        features = obs.features

        if training_in_design[i]:
            pred_y_dict, pred_se_dict = predict_at_point(model, features, metric_names)
        else:
            pred_y_dict = None
            pred_se_dict = None

        for metric_name in obs.data.metric_names:
            if metric_name not in metric_names:
                continue
            obs_y = obs.data.means_dict[metric_name]
            obs_se = np.sqrt(obs.data.covariance_matrix[metric_name][metric_name])

            if pred_y_dict and pred_se_dict:
                pred_y = pred_y_dict[metric_name]
                pred_se = pred_se_dict[metric_name]
            else:
                pred_y = obs_y
                pred_se = obs_se

            records.append(
                {
                    "arm_name": obs.arm_name,
                    "metric_name": metric_name,
                    "x": obs_y,
                    "x_se": obs_se,
                    "y": pred_y,
                    "y_se": pred_se,
                    "arm_parameters": obs.features.parameters,
                }
            )

    return pd.DataFrame.from_records(records)
