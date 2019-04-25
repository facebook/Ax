#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig, TParamValue
from ax.modelbridge.transforms.base import Transform


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
        self.encoded_parameters: Dict[str, Dict[TParamValue, int]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_ordered and not p.is_task:
                self.encoded_parameters[p.name] = {
                    original_value: transformed_value
                    for transformed_value, original_value in enumerate(p.values)
                }
        self.encoded_parameters_inverse: Dict[str, Dict[int, TParamValue]] = {
            p_name: {
                transformed_value: original_value
                for original_value, transformed_value in transforms.items()
            }
            for p_name, transforms in self.encoded_parameters.items()
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoded_parameters:
                if p_name in obsf.parameters:
                    obsf.parameters[p_name] = self.encoded_parameters[p_name][
                        obsf.parameters[p_name]
                    ]
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p in search_space.parameters.values():
            if p.name in self.encoded_parameters:
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
            for p_name, reverse_transform in self.encoded_parameters_inverse.items():
                # pyre: pval is declared to have type `int` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                pval: int = obsf.parameters[p_name]
                obsf.parameters[p_name] = reverse_transform[pval]
        return observation_features
