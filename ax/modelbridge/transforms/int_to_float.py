#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import Parameter, ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.rounding import (
    contains_constrained_integer,
    randomized_round_parameters,
)
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


logger = get_logger(__name__)


DEFAULT_MAX_ROUND_ATTEMPTS = 10000


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
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        self.search_space = search_space
        self.rounding = "strict"
        if config is not None:
            self.rounding = config.get("rounding", "strict")
            self.max_round_attempts = config.get(
                "max_round_attempts", DEFAULT_MAX_ROUND_ATTEMPTS
            )
        else:
            self.max_round_attempts = DEFAULT_MAX_ROUND_ATTEMPTS

        # Identify parameters that should be transformed
        self.transform_parameters: Set[str] = {
            p_name
            for p_name, p in search_space.parameters.items()
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
        new_kwargs = {
            "parameters": list(transformed_parameters.values()),
            "parameter_constraints": [
                pc.clone_with_transformed_parameters(
                    transformed_parameters=transformed_parameters
                )
                for pc in search_space.parameter_constraints
            ],
        }
        if isinstance(search_space, RobustSearchSpace):
            new_kwargs["environmental_variables"] = list(
                search_space._environmental_variables.values()
            )
            # pyre-ignore Incompatible parameter type [6]
            new_kwargs["parameter_distributions"] = search_space.parameter_distributions
        # pyre-ignore Incompatible parameter type [6]
        return search_space.__class__(**new_kwargs)

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            if self.rounding == "strict":
                for p_name in self.transform_parameters:
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
                    not self.search_space.check_membership(rounded_parameters)
                    and round_attempts < self.max_round_attempts
                ):
                    rounded_parameters = randomized_round_parameters(
                        obsf.parameters, self.transform_parameters
                    )
                    round_attempts += 1
                # Update observation with successful rounding or log warning.
                for p_name in self.transform_parameters:
                    obsf.parameters[p_name] = rounded_parameters[p_name]
                if not self.search_space.check_membership(rounded_parameters):
                    logger.warning(
                        f"Unable to round {obsf.parameters}"
                        f"to meet constraints of {self.search_space}"
                    )
        return observation_features
