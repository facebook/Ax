#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class IntRangeToChoice(Transform):
    """Convert a RangeParameter of type int to a ordered ChoiceParameter.

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
        # Identify parameters that should be transformed
        self.transform_parameters: Set[str] = {
            p_name
            for p_name, p in search_space.parameters.items()
            if isinstance(p, RangeParameter) and p.parameter_type == ParameterType.INT
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters and isinstance(p, RangeParameter):
                # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
                values = list(range(p.lower, p.upper + 1))
                target_value = (
                    None
                    if p.target_value is None
                    else next(i for i, v in enumerate(values) if v == p.target_value)
                )
                transformed_parameters[p_name] = ChoiceParameter(
                    name=p_name,
                    parameter_type=p.parameter_type,
                    # Expected `List[Optional[typing.Union[bool, float, str]]]` for
                    # 4th parameter `values` to call
                    # `ax.core.parameter.ChoiceParameter.__init__` but got
                    # `List[int]`.
                    # pyre-fixme[6]:
                    values=values,
                    is_ordered=True,
                    is_fidelity=p.is_fidelity,
                    target_value=target_value,
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
