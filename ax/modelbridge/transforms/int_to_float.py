#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional, Set

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.rounding import randomized_round


class IntToFloat(Transform):
    """Convert a RangeParameter of type int to type float.

    Uses either randomized_rounding or default python rounding,
    depending on 'rounding' flag.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        self.rounding = "strict"
        if config is not None:
            self.rounding = config.get("rounding", "strict")

        # Identify parameters that should be transformed
        self.transform_parameters: Set[str] = {
            p_name
            for p_name, p in search_space.parameters.items()
            if isinstance(p, RangeParameter) and p.parameter_type == ParameterType.INT
        }

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_parameters:
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `int` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: int = obsf.parameters[p_name]
                    obsf.parameters[p_name] = float(param)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: List[Parameter] = []
        for p in search_space.parameters.values():
            # Refine type, since we've only added RangeParameters above.
            if p.name in self.transform_parameters:
                # pyre: p_cast is declared to have type `RangeParameter` but
                # pyre-fixme[9]: is used as type `Parameter`.
                p_cast: RangeParameter = p
                transformed_parameters.append(
                    RangeParameter(
                        name=p_cast.name,
                        parameter_type=ParameterType.FLOAT,
                        lower=p_cast.lower,
                        upper=p_cast.upper,
                        log_scale=p_cast.log_scale,
                        digits=p_cast.digits,
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

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_parameters:
                # pyre: param is declared to have type `float` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                param: float = obsf.parameters.get(p_name)
                if self.rounding == "strict":
                    obsf.parameters[p_name] = int(round(param))  # TODO: T41938776
                else:
                    obsf.parameters[p_name] = randomized_round(param)
        return observation_features
