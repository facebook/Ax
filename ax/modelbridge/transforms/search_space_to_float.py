#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from ax.core.arm import Arm
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform


class SearchSpaceToFloat(Transform):
    """Replaces the search space with a single range parameter, whose values
    are derived from the signature of the arms.

    NOTE: This will have collisions and so should not be used whenever unique
    observation features need to be preserved. Its purpose is to enable
    forward transforms for any search space regardless of parameterization.

    Transform is done in-place.
    """

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        parameter = RangeParameter(
            name="HASH_PARAM",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1e12,
        )
        return SearchSpace(parameters=[parameter])

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            sig = Arm(parameters=obsf.parameters).signature
            val = float(int(sig, 16) % 1_000_000_000_000)
            obsf.parameters = {"HASH_PARAM": val}
        return observation_features
