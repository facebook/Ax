#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig, TParamValue
from ax.modelbridge.transforms.ordered_choice_encode import OrderedChoiceEncode


class TaskEncode(OrderedChoiceEncode):
    """Convert task ChoiceParameters to unit length RangeParameters.

    Parameters will be transformed to an integer RangeParameter,
    mapped from the original choice domain to a contiguous range from [0, n_choices].

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
            if isinstance(p, ChoiceParameter) and p.is_task:
                if p.is_fidelity:
                    raise ValueError(
                        f"Task parameter {p.name} cannot simultaneously be "
                        "fideliy parameter"
                    )
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
