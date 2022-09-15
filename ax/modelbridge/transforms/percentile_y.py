#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import Logger
from typing import List, Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationData
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import get_data
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast
from scipy import stats


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


# TODO(jej): Add OptimizationConfig validation - can't transform outcome constraints.
class PercentileY(Transform):
    """Map Y values to percentiles based on their empirical CDF."""

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert observations is not None, "PercentileY requires observations"
        if len(observations) == 0:
            raise DataRequiredError("Percentile transform requires non-empty data.")
        observation_data = [obs.data for obs in observations]
        metric_values = get_data(observation_data=observation_data)
        # pyre-fixme[4]: Attribute must be annotated.
        self.percentiles = {
            metric_name: vals for metric_name, vals in metric_values.items()
        }
        if config is not None and "winsorize" in config:
            # pyre-fixme[4]: Attribute must be annotated.
            self.winsorize = checked_cast(bool, (config.get("winsorize") or False))
        else:
            self.winsorize = False

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        """Map observation data to empirical CDF quantiles in place."""
        # TODO (jej): Transform covariances.
        if self.winsorize:
            winsorization_rates = {}
            for metric_name, vals in self.percentiles.items():
                n = len(vals)
                # Calculate winsorization rate based on number of observations
                # using formula from [Salinas, Shen, Perrone 2020]
                # https://arxiv.org/abs/1909.13595
                winsorization_rates[metric_name] = (
                    1.0 / (4 * math.pow(n, 0.25) * math.pow(math.pi * math.log(n), 0.5))
                    if n > 1
                    else 0.25
                )
        else:
            winsorization_rates = {
                metric_name: 0 for metric_name in self.percentiles.keys()
            }
        for obsd in observation_data:
            for idx, metric_name in enumerate(obsd.metric_names):
                if metric_name not in self.percentiles:  # pragma: no cover
                    raise ValueError(
                        f"Cannot map value to percentile"
                        f" for unknown metric {metric_name}"
                    )
                # apply map function
                percentile = self._map(obsd.means[idx], metric_name)
                # apply winsorization. If winsorization_rate is 0, has no effect.
                metric_wr = winsorization_rates[metric_name]
                percentile = max(metric_wr, percentile)
                percentile = min((1 - metric_wr), percentile)
                obsd.means[idx] = percentile
                obsd.covariance.fill(float("nan"))
        return observation_data

    def _map(self, val: float, metric_name: str) -> float:
        vals = self.percentiles[metric_name]
        mapped_val = stats.percentileofscore(vals, val, kind="weak") / 100.0
        return mapped_val
