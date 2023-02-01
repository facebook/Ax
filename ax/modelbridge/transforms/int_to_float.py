#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.rounding import (
    contains_constrained_integer,
    randomized_round_parameters,
)
from ax.modelbridge.transforms.utils import construct_new_search_space
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


DEFAULT_MAX_ROUND_ATTEMPTS = 10_000


class IntToFloat(Transform):
    """Convert a RangeParameter of type int to type float.

    Uses either randomized_rounding or default python rounding,
    depending on 'rounding' flag.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        self.search_space: SearchSpace = not_none(
            search_space, "IntToFloat requires search space"
        )
        self.rounding: str = "strict"
        if config is not None:
            self.rounding = config.get("rounding", "strict")  # pyre-ignore
            # pyre-fixme[4]: Attribute must be annotated.
            self.max_round_attempts = config.get(
                "max_round_attempts", DEFAULT_MAX_ROUND_ATTEMPTS
            )
        else:
            self.max_round_attempts = DEFAULT_MAX_ROUND_ATTEMPTS

        # Identify parameters that should be transformed
        self.transform_parameters: Set[str] = {
            p_name
            for p_name, p in self.search_space.parameters.items()
            if isinstance(p, RangeParameter) and p.parameter_type == ParameterType.INT
        }
        if contains_constrained_integer(self.search_space, self.transform_parameters):
            self.rounding = "randomized"

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

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters and isinstance(p, RangeParameter):
                transformed_parameters[p_name] = RangeParameter(
                    name=p_name,
                    parameter_type=ParameterType.FLOAT,
                    # +/- 0.5 ensures that sampling
                    # 1) floating point numbers from (quasi-)Uniform(0,1)
                    # 2) unnormalizing to the raw search space
                    # 3) rounding
                    # results in uniform (quasi-)random integers
                    lower=p.lower - 0.49999,
                    upper=p.upper + 0.49999,
                    log_scale=p.log_scale,
                    digits=p.digits,
                    is_fidelity=p.is_fidelity,
                    target_value=p.target_value,  # casting happens in constructor
                )
            else:
                transformed_parameters[p.name] = p
        return construct_new_search_space(
            search_space=search_space,
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone_with_transformed_parameters(
                    transformed_parameters=transformed_parameters
                )
                for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            present_params = self.transform_parameters.intersection(
                obsf.parameters.keys()
            )
            if self.rounding == "strict":
                for p_name in present_params:
                    # pyre: param is declared to have type `float` but is used as
                    # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters.get(p_name)
                    obsf.parameters[p_name] = int(round(param))  # TODO: T41938776
            else:
                round_attempts = 0
                rounded_parameters = randomized_round_parameters(
                    obsf.parameters, self.transform_parameters
                )
                # Try to round up to max_round_attempt times)
                while (
                    not self.search_space.check_membership(
                        rounded_parameters, check_all_parameters_present=False
                    )
                    and round_attempts < self.max_round_attempts
                ):
                    rounded_parameters = randomized_round_parameters(
                        obsf.parameters, present_params
                    )
                    round_attempts += 1
                if not self.search_space.check_membership(
                    rounded_parameters, check_all_parameters_present=False
                ):
                    logger.warning(
                        f"Unable to round {obsf.parameters}"
                        f"to meet parameter constraints of {self.search_space}"
                    )
                    # This means we failed to randomly round the observation to
                    # something that satisfies the search space bounds and parameter
                    # constraints. We use strict rounding in order to get a candidate
                    # that satisfies the search space bounds, but this candidate may
                    # not satisfy the parameter constraints.
                    for p_name in present_params:
                        param = obsf.parameters.get(p_name)
                        obsf.parameters[p_name] = int(round(param))  # pyre-ignore
                else:  # Update observation if rounding was successful
                    for p_name in present_params:
                        obsf.parameters[p_name] = rounded_parameters[p_name]

        return observation_features
