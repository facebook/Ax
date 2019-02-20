#!/usr/bin/env python3

from typing import Dict, List, Optional

from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig, TParamValue
from ae.lazarus.ae.generator.transforms.base import Transform


class OrderedChoiceEncode(Transform):
    """Convert ordered ChoiceParameters to unit length RangeParameters.

    Parameters will be transformed to an integer RangeParameter,
    mapped from the original choice domain to a contiguous range from [0, n_choices].
    Does not transform task parameters.

    In the inverse transform, parameters will be mapped back onto the original domain.

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
        self.encoded_params: Dict[str, Dict[TParamValue, int]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_ordered and not p.is_task:
                self.encoded_params[p.name] = {
                    original_value: transformed_value
                    for transformed_value, original_value in enumerate(p.values)
                }
        self.encoded_params_inverse: Dict[str, Dict[int, TParamValue]] = {
            p_name: {
                transformed_value: original_value
                for original_value, transformed_value in transforms.items()
            }
            for p_name, transforms in self.encoded_params.items()
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoded_params:
                if p_name in obsf.parameters:
                    obsf.parameters[p_name] = self.encoded_params[p_name][
                        obsf.parameters[p_name]
                    ]
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p in search_space.parameters.values():
            if p.name in self.encoded_params:
                # TypeAssert. Only ChoiceParameters present here.
                # pyre: p_ is declared to have type `ChoiceParameter` but is
                # pyre-fixme[9]: used as type `Parameter`.
                p_: ChoiceParameter = p
                # Choice(|K|) => Range(0, K-1)
                transformed_parameters[p.name] = RangeParameter(
                    name=p_.name,
                    parameter_type=ParameterType.INT,
                    lower=0,
                    upper=len(p_.values) - 1,
                )
            else:
                transformed_parameters[p.name] = p
        return SearchSpace(
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone() for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, reverse_transform in self.encoded_params_inverse.items():
                # pyre: pval is declared to have type `int` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                pval: int = obsf.parameters[p_name]
                obsf.parameters[p_name] = reverse_transform[pval]
        return observation_features
