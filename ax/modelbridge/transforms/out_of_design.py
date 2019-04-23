#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List

from ax.core.observation import ObservationFeatures
from ax.modelbridge.transforms.base import Transform


class OutOfDesign(Transform):
    """Replace out of design arms with empty parameter maps.

    If any parameter values are out of design, the entire arm should be replaced
    with an empty parameter map (used by the model to signal out-of-design points).

    Transform is done in-place.
    """

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        # Wipe null values => arm is out of design.
        for obsf in observation_features:
            if None in obsf.parameters.values():
                obsf.parameters = {}
        return observation_features
