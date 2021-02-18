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
from ax.modelbridge.transforms.utils import get_data
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast


logger = get_logger(__name__)


class Winsorize(Transform):
    """Clip the mean values for each metric to lay within the limits provided in
    the config as a tuple of (lower bound percentile, upper bound percentile).
    These values are the percentile to trim from the lower and upper bounds,
    specified as a float in [0.0, 1). To clip to the 10th and 90th percentile,
    for instance, you'll pass (0.1, 0.1). Config should include those bounds
    under the key "winsorization_limits". Additionally, you can pass in
    percentile_bounds that specify the largest/smallest possible values for the
    percentiles. This is useful in the MOO setting where we want to make sure
    winsorization doesn't move values to the other side of the reference point.
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
        metric_values = get_data(observation_data=observation_data)
        if lower >= 1 - upper:
            raise ValueError(  # pragma: no cover
                f"Lower bound: {lower} was greater than the inverse of the upper "
                f"bound: {1 - upper}. Decrease one or both of your "
                f"winsorization_limits: {(lower, upper)}."
            )
        pct_bounds = {}
        if config is not None and "percentile_bounds" in config:
            pct_bounds = checked_cast(dict, config.get("percentile_bounds") or {})

        self.percentiles = {}
        for metric_name, vals in metric_values.items():
            pct_l = np.percentile(vals, lower * 100, interpolation="lower")
            pct_u = np.percentile(vals, (1 - upper) * 100, interpolation="higher")
            if metric_name in pct_bounds:
                # Update the percentiles if percentile_bounds are specified
                metric_bnds = pct_bounds.get(metric_name)
                if len(metric_bnds) != 2:
                    raise ValueError(  # pragma: no cover
                        f"Expected percentile_bounds for metric {metric_name} to be "
                        f"of the form (l, u), got {metric_bnds}."
                    )
                bnd_l, bnd_u = metric_bnds
                pct_l = min(pct_l, bnd_l if bnd_l is not None else float("inf"))
                pct_u = max(pct_u, bnd_u if bnd_u is not None else -float("inf"))
            self.percentiles[metric_name] = (pct_l, pct_u)

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
