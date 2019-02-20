#!/usr/bin/env python3

from typing import Dict, List, Optional, Union

from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.generator.transforms.base import Transform


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
        self.fixed_params: Dict[str, FixedParameter] = {
            p_name: p
            for p_name, p in search_space.parameters.items()
            if isinstance(p, FixedParameter)
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.fixed_params:
                if p_name in obsf.parameters:
                    obsf.parameters.pop(p_name)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        tunable_parameters: List[Union[ChoiceParameter, RangeParameter]] = []
        for p in search_space.parameters.values():
            if p.name not in self.fixed_params:
                # If it's not in fixed_params, it must be a tunable param.
                # pyre: p_ is declared to have type `Union[ChoiceParameter,
                # pyre: RangeParameter]` but is used as type `ae.lazarus.ae.core.
                # pyre-fixme[9]: parameter.Parameter`.
                p_: Union[ChoiceParameter, RangeParameter] = p
                tunable_parameters.append(p_)
        return SearchSpace(
            # Expected `List[ae.lazarus.ae.core.parameter.Parameter]` for 2nd parameter
            # `parameters` to call `ae.lazarus.ae.core.search_space.SearchSpace.__init__`
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
            for p_name, p in self.fixed_params.items():
                obsf.parameters[p_name] = p.value
        return observation_features
