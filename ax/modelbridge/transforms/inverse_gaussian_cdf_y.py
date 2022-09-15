#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import List, Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationData
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import match_ci_width_truncated
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from scipy.stats import norm


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.modelbridge import base as base_modelbridge  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


# TODO(jej): Add OptimizationConfig validation - can't transform outcome constraints.
class InverseGaussianCdfY(Transform):
    """Apply inverse CDF transform to Y.

    This means that we model uniform distributions as gaussian-distributed.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["base_modelbridge.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.dist = norm(loc=0, scale=1)

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
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
                    mean, var, self._map, lower_bound=0.0, upper_bound=1.0
                )
                obsd.means[idx] = transformed_mean
                obsd.covariance[idx, idx] = transformed_var
        return observation_data

    def _map(self, val: float) -> float:
        mapped_val = self.dist.ppf(val)
        return mapped_val
