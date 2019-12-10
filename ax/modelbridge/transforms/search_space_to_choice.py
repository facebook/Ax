#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from ax.core.arm import Arm
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class SearchSpaceToChoice(Transform):
    """Replaces the search space with a single choice parameter, whose values
    are the signatures of the arms observed in the data.

    This transform is meant to be used with ThompsonSampler.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if any(p.is_fidelity for p in search_space.parameters.values()):
            raise ValueError(
                "Cannot perform SearchSpaceToChoice conversion if fidelity "
                "parameters are present"
            )
        self.parameter_name = "arms"
        self.signature_to_parameterization = {
            Arm(parameters=obsf.parameters).signature: obsf.parameters
            for obsf in observation_features
        }

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        values = list(self.signature_to_parameterization.keys())
        if len(values) > 1:
            parameter = ChoiceParameter(
                name=self.parameter_name,
                parameter_type=ParameterType.STRING,
                values=values,
            )
        else:
            parameter = FixedParameter(
                name=self.parameter_name,
                parameter_type=ParameterType.STRING,
                value=values[0],
            )
        return SearchSpace(parameters=[parameter])

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            obsf.parameters = {
                self.parameter_name: Arm(parameters=obsf.parameters).signature
            }
        return observation_features

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            signature = obsf.parameters[self.parameter_name]
            obsf.parameters = self.signature_to_parameterization[signature]
        return observation_features
