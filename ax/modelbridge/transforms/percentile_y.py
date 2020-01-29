#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger
from scipy import stats


logger = get_logger("Empirical")


class PercentileY(Transform):
    """Map Y values to percentiles based on their empirical CDF.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError(
                "Percentile transform requires non-empty observation data."
            )
        metric_names = {x for obsd in observation_data for x in obsd.metric_names}
        metric_values = {metric_name: [] for metric_name in metric_names}
        for obsd in observation_data:
            for i, metric_name in enumerate(obsd.metric_names):
                metric_values[metric_name].append(obsd.means[i])
        self.percentiles = {
            metric_name: vals for metric_name, vals in metric_values.items()
        }

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Transform observation data in place."""
        for obsd in observation_data:
            for idx, metric_name in enumerate(obsd.metric_names):
                if metric_name not in self.percentiles:  # pragma: no cover
                    raise ValueError(
                        f"Cannot map value to percentile"
                        f" for unknown metric {metric_name}"
                    )
                # apply map function
                vals = self.percentiles[metric_name]

                # pyre-fixme[16]: `scipy.stats` has no attribute `percentileofscore`.
                obsd.means[idx] = stats.percentileofscore(
                    vals, obsd.means[idx], kind="weak"
                )
        return observation_data
