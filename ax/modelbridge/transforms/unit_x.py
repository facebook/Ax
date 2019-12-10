#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class UnitX(Transform):
    """Map X to [0, 1]^d for RangeParameter of type float and not log scale.

    Uses bounds l <= x <= u, sets x_tilde_i = (x_i - l_i) / (u_i - l_i).
    Constraints wTx <= b are converted to gTx_tilde <= h, where
    g_i = w_i (u_i - l_i) and h = b - wTl.

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

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = (param - l) / (u - l)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.bounds and isinstance(p, RangeParameter):
                p.update_range(lower=0.0, upper=1.0)
            if p.target_value is not None:
                l, u = self.bounds[p_name]
                p._target_value = (p.target_value - l) / (u - l)  # pyre-ignore [16]
        new_constraints: List[ParameterConstraint] = []
        for c in search_space.parameter_constraints:
            constraint_dict: Dict[str, float] = {}
            bound = float(c.bound)
            for p_name, w in c.constraint_dict.items():
                # p is RangeParameter, but may not be transformed (Int or log)
                if p_name in self.bounds:
                    l, u = self.bounds[p_name]
                    constraint_dict[p_name] = w * (u - l)
                    bound -= w * l
                else:
                    constraint_dict[p_name] = w
            new_constraints.append(
                ParameterConstraint(constraint_dict=constraint_dict, bound=bound)
            )
        search_space.set_parameter_constraints(new_constraints)
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                # pyre: param is declared to have type `float` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                param: float = obsf.parameters[p_name]
                obsf.parameters[p_name] = param * (u - l) + l
        return observation_features
