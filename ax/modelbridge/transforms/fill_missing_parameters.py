#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class FillMissingParameters(Transform):
    """If a parameter is missing from an arm, fill it with the value from
    the dict given in the config.

    Config supports two options.
        fill_values: a dict of {parameter_name: value} to fill in for missing
            parameters. Required.
        fill_None: a boolean indicating whether to fill in None values. Default
            is True. If False, parameters specified as None will remain None,
            and only parameters absent altogether will be filled.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        self.fill_values: TParameterization | None = config.get(  # pyre-ignore[8]
            "fill_values", None
        )
        self.fill_None: bool = assert_is_instance(config.get("fill_None", True), bool)

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        if self.fill_values is None:
            return observation_features
        for obsf in observation_features:
            fill_params = {
                k: v
                for k, v in none_throws(self.fill_values).items()
                if k not in obsf.parameters
                or (obsf.parameters[k] is None and self.fill_None)
            }
            obsf.parameters.update(fill_params)
        return observation_features
