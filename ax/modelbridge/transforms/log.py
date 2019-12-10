#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Set

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class Log(Transform):
    """Apply log base 10 to a float RangeParameter domain.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        # Identify parameters that should be transformed
        self.transform_parameters: Set[str] = {
            p_name
            for p_name, p in search_space.parameters.items()
            if isinstance(p, RangeParameter)
            and p.parameter_type == ParameterType.FLOAT
            and p.log_scale is True
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_parameters:
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = math.log10(param)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters and isinstance(p, RangeParameter):
                p.set_log_scale(False).update_range(
                    lower=math.log10(p.lower), upper=math.log10(p.upper)
                )
                if p.target_value is not None:
                    p._target_value = math.log10(p.target_value)  # pyre-ignore [6]
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_parameters:
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = math.pow(10, param)
        return observation_features
