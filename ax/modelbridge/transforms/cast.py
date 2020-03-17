#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class Cast(Transform):
    """Cast each param value to the respective parameter's type/format

    This is a default transform that should run across all models

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: Optional[List[ObservationFeatures]] = None,
        observation_data: Optional[List[ObservationData]] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        self.search_space = search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, p_inst in self.search_space.parameters.items():
                param = obsf.parameters.get(p_name)
                obsf.parameters[p_name] = p_inst.cast(param)
        return observation_features
