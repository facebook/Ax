#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from typing import Iterator

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.map_key_to_float import MapKeyToFloat
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


WIDTHS = [2.0, 4.0, 8.0]
HEIGHTS = [4.0, 2.0, 8.0]
STEPS_ENDS = [1, 5, 3]


def _enumerate() -> Iterator[tuple[int, float, float, float]]:
    yield from (
        (trial_index, width, height, float(i + 1))
        for trial_index, (width, height, steps_end) in enumerate(
            zip(WIDTHS, HEIGHTS, STEPS_ENDS)
        )
        for i in range(steps_end)
    )


class MapKeyToFloatTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="width",
                    parameter_type=ParameterType.FLOAT,
                    lower=1,
                    upper=20,
                ),
                RangeParameter(
                    name="height",
                    parameter_type=ParameterType.FLOAT,
                    lower=1,
                    upper=20,
                ),
            ]
        )

        self.observations = []
        for trial_index, width, height, steps in _enumerate():
            obs_feat = ObservationFeatures(
                trial_index=trial_index,
                parameters={"width": width, "height": height},
                metadata={
                    "foo": 42,
                    MapKeyToFloat.DEFAULT_MAP_KEY: steps,
                },
            )
            obs_data = ObservationData(
                metric_names=[], means=np.array([]), covariance=np.empty((0, 0))
            )
            self.observations.append(Observation(features=obs_feat, data=obs_data))

        # does not require explicitly specifying `config`
        self.t = MapKeyToFloat(
            observations=self.observations,
        )

    def test_Init(self) -> None:
        self.assertEqual(len(self.t._parameter_list), 1)

        p = self.t._parameter_list[0]

        self.assertEqual(p.name, MapKeyToFloat.DEFAULT_MAP_KEY)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 1.0)
        self.assertEqual(p.upper, 5.0)
        self.assertTrue(p.log_scale)

        # test that one is able to override default config
        with self.subTest(msg="override default config"):
            t = MapKeyToFloat(
                observations=self.observations,
                config={
                    "parameters": {MapKeyToFloat.DEFAULT_MAP_KEY: {"log_scale": False}}
                },
            )
            self.assertDictEqual(t.parameters, {"step": {"log_scale": False}})

            self.assertEqual(len(t._parameter_list), 1)

            p = t._parameter_list[0]

            self.assertEqual(p.name, MapKeyToFloat.DEFAULT_MAP_KEY)
            self.assertEqual(p.parameter_type, ParameterType.FLOAT)
            self.assertEqual(p.lower, 1.0)
            self.assertEqual(p.upper, 5.0)
            self.assertFalse(p.log_scale)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        self.assertSetEqual(
            set(ss2.parameters),
            {"height", "width", MapKeyToFloat.DEFAULT_MAP_KEY},
        )

        p = assert_is_instance(
            ss2.parameters[MapKeyToFloat.DEFAULT_MAP_KEY], RangeParameter
        )

        self.assertEqual(p.name, MapKeyToFloat.DEFAULT_MAP_KEY)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 1.0)
        self.assertEqual(p.upper, 5.0)
        self.assertTrue(p.log_scale)

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [obs.features for obs in self.observations]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)

        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    trial_index=trial_index,
                    parameters={
                        "width": width,
                        "height": height,
                        MapKeyToFloat.DEFAULT_MAP_KEY: steps,
                    },
                    metadata={"foo": 42},
                )
                for trial_index, width, height, steps in _enumerate()
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def test_TransformObservationFeaturesWithEmptyParameters(self) -> None:
        obsf = ObservationFeatures(parameters={})
        self.t.transform_observation_features([obsf])

        p = self.t._parameter_list[0]
        self.assertEqual(
            obsf,
            ObservationFeatures(parameters={MapKeyToFloat.DEFAULT_MAP_KEY: p.upper}),
        )
