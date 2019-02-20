#!/usr/bin/env python3

import math
from typing import List, Optional, Set

from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.parameter import ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.generator.transforms.base import Transform


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
        self.transform_params: Set[str] = {
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
            for p_name in self.transform_params:
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = math.log10(param)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_params:
                # pyre: p_cast is declared to have type `RangeParameter` but
                # pyre-fixme[9]: is used as type `ae.lazarus.ae.core.parameter.Parameter`.
                p_cast: RangeParameter = p
                p_cast.set_log_scale(False).update_range(
                    lower=math.log10(p_cast.lower), upper=math.log10(p_cast.upper)
                )
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_params:
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = math.pow(10, param)
        return observation_features
