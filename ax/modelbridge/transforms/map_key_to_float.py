#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.utils import extract_map_keys_from_opt_config
from ax.exceptions.core import UserInputError
from ax.modelbridge.transforms.metadata_to_float import MetadataToFloat
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class MapKeyToFloat(MetadataToFloat):
    """
    This transform extracts the entry from the metadata field of the observation
    features corresponding to the `parameters` specified in the transform config
    and inserts it into the parameter field. If no parameters are specified in the
    config, the transform will extract all map key names from the optimization config.

    Inheriting from the `MetadataToFloat` transform, this transform
    also adds a range (float) parameter to the search space.
    Similarly, users can override the default behavior by specifying
    the `config` with `parameters` as the key, where each entry maps
    a metadata key to a dictionary of keyword arguments for the
    corresponding RangeParameter constructor.

    Transform is done in-place.
    """

    # NOTE: This will be ignored if the lower bound is <= 0.
    DEFAULT_LOG_SCALE: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        if "parameters" not in config:
            # Extract map keys from the optimization config, if no parameters are
            # specified in the config.
            if modelbridge is not None and modelbridge._optimization_config is not None:
                config["parameters"] = {
                    key: {}
                    for key in extract_map_keys_from_opt_config(
                        optimization_config=modelbridge._optimization_config
                    )
                }
            else:
                raise UserInputError(
                    f"{self.__class__.__name__} requires either `parameters` to be "
                    "specified in the transform config or a modelbridge with an "
                    "optimization config, from which the map keys can be extracted."
                )
        super().__init__(
            search_space=search_space,
            observations=observations,
            modelbridge=modelbridge,
            config=config,
        )

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        if len(obsf.parameters) == 0:
            obsf.parameters = {p.name: p.upper for p in self._parameter_list}
            return
        if obsf.metadata is None or len(obsf.metadata) == 0:
            obsf.metadata = {p.name: p.upper for p in self._parameter_list}
        super()._transform_observation_feature(obsf)
