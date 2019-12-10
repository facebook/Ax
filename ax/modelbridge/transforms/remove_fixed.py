#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class RemoveFixed(Transform):
    """Remove fixed parameters.

    Fixed parameters should not be included in the SearchSpace.
    This transform removes these parameters, leaving only tunable parameters.

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
        self.fixed_parameters: Dict[str, FixedParameter] = {
            p_name: p
            for p_name, p in search_space.parameters.items()
            if isinstance(p, FixedParameter)
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, fixed_p in self.fixed_parameters.items():
                if p_name in obsf.parameters:
                    if obsf.parameters[p_name] != fixed_p.value:
                        raise ValueError(
                            "Fixed parameter with out of design value passed "
                            "to RemoveFixed."
                        )
                    obsf.parameters.pop(p_name)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        tunable_parameters: List[Union[ChoiceParameter, RangeParameter]] = []
        for p in search_space.parameters.values():
            if p.name not in self.fixed_parameters:
                # If it's not in fixed_parameters, it must be a tunable param.
                # pyre: p_ is declared to have type `Union[ChoiceParameter,
                # pyre: RangeParameter]` but is used as type `ax.core.
                # pyre-fixme[9]: parameter.Parameter`.
                p_: Union[ChoiceParameter, RangeParameter] = p
                tunable_parameters.append(p_)
        return SearchSpace(
            # Expected `List[ax.core.parameter.Parameter]` for 2nd parameter
            # `parameters` to call `ax.core.search_space.SearchSpace.__init__`
            # but got `List[Union[ChoiceParameter, RangeParameter]]`.
            # pyre-fixme[6]:
            parameters=tunable_parameters,
            parameter_constraints=[
                pc.clone() for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, p in self.fixed_parameters.items():
                obsf.parameters[p_name] = p.value
        return observation_features
