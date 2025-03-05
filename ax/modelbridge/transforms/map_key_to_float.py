#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.core.map_metric import MapMetric
from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.metadata_to_float import MetadataToFloat
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class MapKeyToFloat(MetadataToFloat):
    """
    This transform extracts the entry from the metadata field of the observation
    features corresponding to the `parameters` specified in the transform config,
    or the default map key (`MapMetric.map_key_info.key`) if not specified,
    and inserts it into the parameter field.

    Inheriting from the `MetadataToFloat` transform, this transform
    also adds a range (float) parameter to the search space.
    Similarly, users can override the default behavior by specifying
    the `config` with `parameters` as the key, where each entry maps
    a metadata key to a dictionary of keyword arguments for the
    corresponding RangeParameter constructor.

    Transform is done in-place.
    """

    DEFAULT_LOG_SCALE: bool = True
    DEFAULT_MAP_KEY: str = MapMetric.map_key_info.key

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        # Use the default map key if nothing is specified in the config.
        if "parameters" not in config:
            config["parameters"] = {self.DEFAULT_MAP_KEY: {}}
        super().__init__(
            search_space=search_space,
            observations=observations,
            modelbridge=modelbridge,
            config=config,
        )

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        if not obsf.parameters:
            for p in self._parameter_list:
                obsf.parameters[p.name] = p.upper
            return
        super()._transform_observation_feature(obsf)
