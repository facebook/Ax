#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast


logger = get_logger(__name__)


class Winsorize(Transform):
    """Clip the mean values for each metric to lay within the limits provided in
    the config as a tuple of (lower bound percentile, upper bound percentile).
    These values are the percentile to trim from the lower and upper bounds,
    specified as a float in [0.0, 1). To clip to the 10th and 90th percentile,
    for instance, you'll pass (0.1, 0.1). Config should include those bounds
    under the key "winsorization_limits".
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError("Winsorize transform requires non-empty observation data.")
        # If winsorization limits are missing or either one of them is None,
        # we can just replace that limit(s) with 0.0, as in those cases the
        # percentile will just interpret them as 0-th or 100-th percentile,
        # leaving the data unclipped.
        lower = 0.0
        if config is not None and "winsorization_lower" in config:
            lower = checked_cast(float, (config.get("winsorization_lower") or 0.0))
        upper = 0.0
        if config is not None and "winsorization_upper" in config:
            upper = checked_cast(float, (config.get("winsorization_upper") or 0.0))
        metric_names = {x for obsd in observation_data for x in obsd.metric_names}
        metric_values = {metric_name: [] for metric_name in metric_names}
        for obsd in observation_data:
            for i, metric_name in enumerate(obsd.metric_names):
                metric_values[metric_name].append(obsd.means[i])
        if lower >= 1 - upper:
            raise ValueError(  # pragma: no cover
                f"Lower bound: {lower} was greater than the inverse of the upper "
                f"bound: {1 - upper}. Decrease one or both of your "
                f"winsorization_limits: {(lower, upper)}."
            )
        self.percentiles = {
            metric_name: (
                np.percentile(vals, lower * 100, interpolation="lower"),
                np.percentile(vals, (1 - upper) * 100, interpolation="higher"),
            )
            for metric_name, vals in metric_values.items()
        }

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for idx, metric_name in enumerate(obsd.metric_names):
                if metric_name not in self.percentiles:  # pragma: no cover
                    raise ValueError(f"Cannot winsorize unknown metric {metric_name}")
                # Clip on the winsorization bounds.
                obsd.means[idx] = max(obsd.means[idx], self.percentiles[metric_name][0])
                obsd.means[idx] = min(obsd.means[idx], self.percentiles[metric_name][1])
        return observation_data
