#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional, TYPE_CHECKING

from ax.core.map_metric import MapMetric
from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.metadata_to_range import MetadataToFloat
from ax.models.types import TConfig
from pyre_extensions import assert_is_instance

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class MapKeyToFloat(MetadataToFloat):
    DEFAULT_LOG_SCALE: bool = True
    DEFAULT_MAP_KEY: str = MapMetric.map_key_info.key

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        self.parameters: dict[str, dict[str, Any]] = assert_is_instance(
            config.setdefault("parameters", {}), dict
        )
        # TODO[tiao]: raise warning if `DEFAULT_MAP_KEY` is already in keys(?)
        self.parameters.setdefault(self.DEFAULT_MAP_KEY, {})
        super().__init__(
            search_space=search_space,
            observations=observations,
            modelbridge=modelbridge,
            config=config,
        )

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        if not obsf.parameters:
            for p in self._parameter_list:
                # TODO[tiao]: can we use be p.target_value?
                # (not its original intended use but could be advantageous)
                obsf.parameters[p.name] = p.upper
            return
        super()._transform_observation_feature(obsf)
