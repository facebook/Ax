#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Utility functions for working with observation data.

This module is intentionally kept minimal and dependency-free (with respect to
other ax.adapter modules) to avoid circular imports.
"""

from ax.core.observation import ObservationData
from ax.core.types import TModelCov, TModelMean, TModelPredict


def unwrap_observation_data(observation_data: list[ObservationData]) -> TModelPredict:
    """Converts observation data to the format for model prediction outputs.
    That format assumes each observation data has the same set of metrics.
    """
    metrics = set(observation_data[0].metric_signatures)
    f: TModelMean = {metric: [] for metric in metrics}
    cov: TModelCov = {m1: {m2: [] for m2 in metrics} for m1 in metrics}
    for od in observation_data:
        if set(od.metric_signatures) != metrics:
            raise ValueError(
                "Each ObservationData should use same set of metrics. "
                "Expected {exp}, got {got}.".format(
                    exp=metrics, got=set(od.metric_signatures)
                )
            )
        for i, m1 in enumerate(od.metric_signatures):
            f[m1].append(od.means[i])
            for j, m2 in enumerate(od.metric_signatures):
                cov[m1][m2].append(od.covariance[i, j])
    return f, cov
