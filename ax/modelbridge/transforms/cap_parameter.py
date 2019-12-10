#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.typeutils import checked_cast


class CapParameter(Transform):
    """Cap parameter range(s) to given values. Expects a configuration of form
    { parameter_name -> new_upper_range_value }.

    This transform only transforms the search space.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        self.config = config or {}
        self.transform_parameters = {  # Only transform parameters in config.
            p_name for p_name in search_space.parameters if p_name in self.config
        }

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters:
                if not isinstance(p, RangeParameter):
                    raise NotImplementedError(
                        "Can only cap range parameters currently."
                    )
                checked_cast(RangeParameter, p).update_range(
                    upper=self.config.get(p_name)
                )
        return search_space
