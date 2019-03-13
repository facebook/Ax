#!/usr/bin/env python3

from typing import Dict, List, Optional

from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.parameter import ChoiceParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig, TParamValue
from ae.lazarus.ae.modelbridge.transforms.ordered_choice_encode import (
    OrderedChoiceEncode,
)


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
        self.encoded_params: Dict[str, Dict[TParamValue, int]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_task:
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
