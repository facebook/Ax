#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import isnan
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger
from scipy.stats import norm


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.modelbridge import base as base_modelbridge  # noqa F401  # pragma: no cover


logger = get_logger("LogY")


# TODO(jej): Add OptimizationConfig validation - can't transform outcome constraints.
class InverseGaussianCdfY(Transform):
    """Apply inverse CDF transform to Y.

    This means that we model uniform distributions as gaussian-distributed.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        # pyre-fixme[29]: `scipy.stats.norm_gen` is not a function.
        self.dist = norm(loc=0, scale=1)

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Map to inverse Gaussian CDF in place."""
        # TODO (jej): Transform covariances.
        for obsd in observation_data:
            for idx, _ in enumerate(obsd.metric_names):
                mean = float(obsd.means[idx])
                # Error on out-of-domain values.
                if mean <= 0.0 or mean >= 1.0:
                    raise ValueError(
                        f"Inverse CDF cannot transform value: {mean} outside (0, 1)"
                    )
                var = float(obsd.covariance[idx, idx])
                transformed_mean, transformed_var = match_ci_width_truncated(
                    mean, var, self._map
                )
                obsd.means[idx] = transformed_mean
                obsd.covariance[idx, idx] = transformed_var
        return observation_data

    def _map(self, val: float) -> float:
        mapped_val = self.dist.ppf(val)
        return mapped_val


def match_ci_width_truncated(
    mean: float,
    variance: float,
    transform: Callable[[float], float],
    level: float = 0.95,
    margin: float = 0.001,
) -> Tuple[float, float]:
    """Estimate a transformed variance using the match ci width method.

    See log_y transform for the original. Here, bounds are forced to lie
    within a [0,1] interval after transformation."""
    fac = norm.ppf(1 - (1 - level) / 2)
    d = fac * np.sqrt(variance)
    upper_bound = min(mean + d, 1.0 - margin)
    lower_bound = max(mean - d, margin)
    width_asym = transform(upper_bound) - transform(lower_bound)
    new_mean = transform(mean)
    new_variance = float("nan") if isnan(variance) else (width_asym / 2 / fac) ** 2
    return new_mean, new_variance
