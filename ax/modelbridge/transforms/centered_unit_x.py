#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.docutils import copy_doc


class CenteredUnitX(Transform):
    """Map X to [-1, 1]^d for RangeParameter of type float and not log scale.

    Currently does not support linear constraints, but could in the future be
    adjusted to transform them too, since this is a linear operation.

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
        self.bounds: Dict[str, Tuple[float, float]] = {}
        for p_name, p in search_space.parameters.items():
            if (
                isinstance(p, RangeParameter)
                and p.parameter_type == ParameterType.FLOAT
                and not p.log_scale
            ):
                self.bounds[p_name] = (p.lower, p.upper)

    @copy_doc(Transform.transform_observation_features)
    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = -1 + 2 * (param - l) / (u - l)
        return observation_features

    @copy_doc(Transform.transform_search_space)
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.bounds and isinstance(p, RangeParameter):
                p.update_range(lower=-1.0, upper=1.0)
            if p.target_value is not None:
                l, u = self.bounds[p_name]
                new_tval = -1 + 2 * (p.target_value - l) / (u - l)  # pyre-ignore [16]
                p._target_value = new_tval
        for c in search_space.parameter_constraints:
            for p_name in c.constraint_dict:
                if p_name in self.bounds:
                    raise ValueError("Does not support parameter constraints")
        return search_space

    @copy_doc(Transform.untransform_observation_features)
    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                # pyre: param is declared to have type `float` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                param: float = obsf.parameters[p_name]
                obsf.parameters[p_name] = ((param + 1) / 2) * (u - l) + l
        return observation_features
