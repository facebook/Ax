#!/usr/bin/env python3

from typing import List, Optional

import numpy as np
from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.generator.transforms.base import Transform
from ae.lazarus.ae.utils.common.logger import get_logger


logger = get_logger("Winsorize")


class Winsorize(Transform):
    """Clip the mean values for each metric to lay within the limits provided in
    the config as a tuple of (lower bound percentile, upper bound percentile).
    Config should include those bounds under key "winsorization_limits".
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
        lower, upper = (
            config.get("winsorization_limits", (0.0, 0.0))
            if config is not None
            else (0.0, 0.0)
        )
        lower, upper = lower or 0.0, upper or 0.0
        metric_names = {x for obsd in observation_data for x in obsd.metric_names}
        metric_values = {metric_name: [] for metric_name in metric_names}
        for obsd in observation_data:
            for i, metric_name in enumerate(obsd.metric_names):
                metric_values[metric_name].append(obsd.means[i])
        assert (
            lower <= 1 - upper
        ), f"Upper bound: {upper} was less than the inverse of lower: {lower}"
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
