#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class MapUnitX(UnitX):
    """A `UnitX` transform for map parameters in observation_features, identified
    as those that are not part of the search space. Since they are not part of the
    search space, the bounds are inferred from the set of observation features. Only
    observation features are transformed; all other objects undergo identity transform.
    """

    target_lb: float = 0.0
    target_range: float = 1.0

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert observations is not None, "MapUnitX requires observations"
        assert search_space is not None, "MapUnitX requires search space"
        # Loop through observation features and identify parameters that
        # are not part of the search space. Store all observed values to
        # infer bounds
        map_values = defaultdict(list)
        for obs in observations:
            for p in obs.features.parameters:
                if p not in search_space.parameters:
                    map_values[p].append(obs.features.parameters[p])

        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        def get_range(values: List) -> Tuple[float, float]:
            return (min(values), max(values))

        self.bounds: Dict[str, Tuple[float, float]] = {
            p: get_range(v) for p, v in map_values.items()
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        return search_space

    def _transform_parameter_distributions(self, search_space: SearchSpace) -> None:
        return super(UnitX, self)._transform_parameter_distributions(
            search_space=search_space
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        """Untransform if the parameter exists in the observation feature. Note the
        extra existence check from `UnitX.untransform_observation_features` because
        when map key features are used, they may not exist after generation or best
        point computations."""
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    param: float = obsf.parameters[p_name]  # pyre-ignore[9]
                    scale_fac = (u - l) / self.target_range
                    obsf.parameters[p_name] = scale_fac * (param - self.target_lb) + l
        return observation_features
