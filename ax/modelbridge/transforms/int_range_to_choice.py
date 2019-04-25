#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional, Set

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


class IntRangeToChoice(Transform):
    """Convert a RangeParameter of type int to a ChoiceParameter.

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
            if isinstance(p, RangeParameter) and p.parameter_type == ParameterType.INT
        }

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: List[Parameter] = []
        for p in search_space.parameters.values():
            if p.name in self.transform_parameters:
                # pyre: p_cast is declared to have type `RangeParameter` but
                # pyre-fixme[9]: is used as type `Parameter`.
                p_cast: RangeParameter = p
                transformed_parameters.append(
                    ChoiceParameter(
                        name=p_cast.name,
                        parameter_type=p_cast.parameter_type,
                        # Expected `List[Optional[typing.Union[bool, float, str]]]` for
                        # 4th parameter `values` to call
                        # `ax.core.parameter.ChoiceParameter.__init__` but got
                        # `List[int]`.
                        # pyre-fixme[6]:
                        values=list(range(p_cast.lower, p_cast.upper + 1)),
                    )
                )
            else:
                transformed_parameters.append(p)
        return SearchSpace(
            parameters=transformed_parameters,
            parameter_constraints=[
                pc.clone() for pc in search_space.parameter_constraints
            ],
        )
