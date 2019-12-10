#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger


logger = get_logger("IVW")


def ivw_metric_merge(
    obsd: ObservationData, conflicting_noiseless: str = "warn"
) -> ObservationData:
    """Merge multiple observations of a metric with inverse variance weighting.

    Correctly updates the covariance of the new merged estimates:
    ybar1 = Sum_i w_i * y_i
    ybar2 = Sum_j w_j * y_j
    cov[ybar1, ybar2] = Sum_i Sum_j w_i * w_j * cov[y_i, y_j]

    w_i will be infinity if any variance is 0. If one variance is 0., then
    the IVW estimate is the corresponding mean. If there are multiple
    measurements with 0 variance but means are all the same, then IVW estimate
    is that mean. If there are multiple measurements and means differ, behavior
    depends on argument conflicting_noiseless. "ignore" and "warn" will use
    the first of the measurements as the IVW estimate. "warn" will additionally
    log a warning. "raise" will raise an exception.

    Args:
        obsd: An ObservationData object
        conflicting_noiseless: "warn", "ignore", or "raise"
    """
    if len(obsd.metric_names) == len(set(obsd.metric_names)):
        return obsd
    if conflicting_noiseless not in {"warn", "ignore", "raise"}:
        raise ValueError(
            'conflicting_noiseless should be "warn", "ignore", or "raise".'
        )
    # Get indicies and weights for each metric.
    # weights is a map from metric name to a vector of the weights for each
    # measurement of that metric. indicies gives the corresponding index in
    # obsd.means for each measurement.
    weights: Dict[str, np.ndarray] = {}
    indicies: Dict[str, List[int]] = {}
    for metric_name in set(obsd.metric_names):
        indcs = [i for i, mn in enumerate(obsd.metric_names) if mn == metric_name]
        indicies[metric_name] = indcs
        # Extract variances for observations of this metric
        sigma2s = obsd.covariance[indcs, indcs]
        # Check for noiseless observations
        idx_noiseless = np.where(sigma2s == 0.0)[0]
        if len(idx_noiseless) == 0:
            # Weight is inverse of variance, normalized
            # Expected `np.ndarray` for 3rd anonymous parameter to call
            # `dict.__setitem__` but got `float`.
            # pyre-fixme[6]:
            weights[metric_name] = 1.0 / sigma2s
            weights[metric_name] /= np.sum(weights[metric_name])
        else:
            # Check if there are conflicting means for the noiseless observations
            means_noiseless = obsd.means[indcs][idx_noiseless]
            _check_conflicting_means(
                means_noiseless, metric_name, conflicting_noiseless
            )
            # The first observation gets all the weight.
            weights[metric_name] = np.zeros_like(sigma2s)
            weights[metric_name][idx_noiseless[0]] = 1.0
    # Compute the new values
    metric_names = sorted(set(obsd.metric_names))
    means = np.zeros(len(metric_names))
    covariance = np.zeros((len(metric_names), len(metric_names)))
    for i, metric_name in enumerate(metric_names):
        ys = obsd.means[indicies[metric_name]]
        means[i] = np.sum(weights[metric_name] * ys)
        # Calculate covariances with metric_name
        for j, metric_name2 in enumerate(metric_names[i:], start=i):
            for ii, idx_i in enumerate(indicies[metric_name]):
                for jj, idx_j in enumerate(indicies[metric_name2]):
                    covariance[i, j] += (
                        weights[metric_name][ii]
                        * weights[metric_name2][jj]
                        * obsd.covariance[idx_i, idx_j]
                    )
            covariance[j, i] = covariance[i, j]
    return ObservationData(
        metric_names=metric_names, means=means, covariance=covariance
    )


def _check_conflicting_means(
    means_noiseless: np.ndarray, metric_name: str, conflicting_noiseless: str
) -> None:
    if np.var(means_noiseless) > 0:
        message = f"Conflicting noiseless measurements for {metric_name}."
        if conflicting_noiseless == "warn":
            logger.warning(message)
        elif conflicting_noiseless == "raise":
            raise ValueError(message)


class IVW(Transform):
    """If an observation data contains multiple observations of a metric, they
    are combined using inverse variance weighting.
    """

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        # pyre: conflicting_noiseless is declared to have type `str` but is
        # pyre-fixme[9]: used as type `typing.Union[float, int, str]`.
        conflicting_noiseless: str = self.config.get("conflicting_noiseless", "warn")
        return [
            ivw_metric_merge(obsd=obsd, conflicting_noiseless=conflicting_noiseless)
            for obsd in observation_data
        ]
