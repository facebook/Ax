#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterator
from copy import deepcopy

import numpy as np
from ax.core.experiment import Experiment
from ax.core.map_metric import MapMetric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.modelbridge import Adapter
from ax.modelbridge.transforms.map_key_to_float import MapKeyToFloat
from ax.models.base import Generator
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


WIDTHS = [2.0, 4.0, 8.0]
HEIGHTS = [4.0, 2.0, 8.0]
STEP_ENDS = [1, 5, 3]
DEFAULT_MAP_KEY: str = MapMetric.map_key_info.key


def _enumerate() -> Iterator[tuple[int, float, float, float]]:
    yield from (
        (trial_index, width, height, float(i + 1))
        for trial_index, (width, height, step_end) in enumerate(
            zip(WIDTHS, HEIGHTS, STEP_ENDS)
        )
        for i in range(step_end)
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
        optimization_config = OptimizationConfig(
            objective=Objective(metric=MapMetric(name="map_metric"), minimize=True)
        )
        adapter = Adapter(
            experiment=Experiment(
                search_space=self.search_space, optimization_config=optimization_config
            ),
            model=Generator(),
        )

        self.observations = []
        for trial_index, width, height, step in _enumerate():
            obs_feat = ObservationFeatures(
                trial_index=trial_index,
                parameters={"width": width, "height": height},
                metadata={
                    "foo": 42,
                    DEFAULT_MAP_KEY: step,
                },
            )
            obs_data = ObservationData(
                metric_names=[], means=np.array([]), covariance=np.empty((0, 0))
            )
            self.observations.append(Observation(features=obs_feat, data=obs_data))

        # does not require explicitly specifying `config`
        self.t = MapKeyToFloat(observations=self.observations, modelbridge=adapter)

    def test_Init(self) -> None:
        # Check for error if adapter & parameters are not provided.
        with self.assertRaisesRegex(UserInputError, "optimization config"):
            MapKeyToFloat(observations=self.observations)

        # Check for default initialization
        self.assertEqual(len(self.t._parameter_list), 1)
        p = self.t._parameter_list[0]
        self.assertEqual(p.name, DEFAULT_MAP_KEY)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 1.0)
        self.assertEqual(p.upper, 5.0)
        self.assertTrue(p.log_scale)

        # test that one is able to override default config
        with self.subTest(msg="override default config"):
            t = MapKeyToFloat(
                observations=self.observations,
                config={"parameters": {DEFAULT_MAP_KEY: {"log_scale": False}}},
            )
            self.assertDictEqual(t.parameters, {"step": {"log_scale": False}})

            self.assertEqual(len(t._parameter_list), 1)

            p = t._parameter_list[0]

            self.assertEqual(p.name, DEFAULT_MAP_KEY)
            self.assertEqual(p.parameter_type, ParameterType.FLOAT)
            self.assertEqual(p.lower, 1.0)
            self.assertEqual(p.upper, 5.0)
            self.assertFalse(p.log_scale)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        self.assertSetEqual(
            set(ss2.parameters),
            {"height", "width", DEFAULT_MAP_KEY},
        )

        p = assert_is_instance(ss2.parameters[DEFAULT_MAP_KEY], RangeParameter)

        self.assertEqual(p.name, DEFAULT_MAP_KEY)
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
                        DEFAULT_MAP_KEY: step,
                    },
                    metadata={"foo": 42},
                )
                for trial_index, width, height, step in _enumerate()
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def test_TransformObservationFeaturesWithEmptyMetadata(self) -> None:
        # undefined metadata
        obsf = ObservationFeatures(
            trial_index=42,
            parameters={"width": 1.0, "height": 2.0},
            metadata=None,
        )
        self.t.transform_observation_features([obsf])
        self.assertEqual(
            obsf,
            ObservationFeatures(
                trial_index=42,
                parameters={
                    "width": 1.0,
                    "height": 2.0,
                    DEFAULT_MAP_KEY: 5.0,
                },
                metadata={},
            ),
        )
        # empty metadata
        obsf = ObservationFeatures(
            trial_index=42,
            parameters={"width": 1.0, "height": 2.0},
            metadata={},
        )
        self.t.transform_observation_features([obsf])
        self.assertEqual(
            obsf,
            ObservationFeatures(
                trial_index=42,
                parameters={
                    "width": 1.0,
                    "height": 2.0,
                    DEFAULT_MAP_KEY: 5.0,
                },
                metadata={},
            ),
        )

    def test_TransformObservationFeaturesWithEmptyParameters(self) -> None:
        obsf = ObservationFeatures(parameters={})
        self.t.transform_observation_features([obsf])

        p = self.t._parameter_list[0]
        self.assertEqual(
            obsf,
            ObservationFeatures(parameters={DEFAULT_MAP_KEY: p.upper}),
        )

    def test_with_different_map_key(self) -> None:
        observations = [
            Observation(
                features=ObservationFeatures(
                    trial_index=0,
                    parameters={"width": width, "height": height},
                    metadata={"timestamp": timestamp},
                ),
                data=ObservationData(
                    metric_names=[], means=np.array([]), covariance=np.empty((0, 0))
                ),
            )
            for width, height, timestamp in (
                (0.0, 1.0, 12345.0),
                (0.1, 0.9, 12346.0),
            )
        ]
        t = MapKeyToFloat(
            observations=observations,
            config={"parameters": {"timestamp": {"log_scale": False}}},
        )
        self.assertEqual(t.parameters, {"timestamp": {"log_scale": False}})
        self.assertEqual(len(t._parameter_list), 1)
        tf_obs_ft = t.transform_observation_features(
            [obs.features for obs in observations]
        )
        self.assertEqual(
            tf_obs_ft[0].parameters, {"width": 0.0, "height": 1.0, "timestamp": 12345.0}
        )
        self.assertEqual(
            tf_obs_ft[1].parameters, {"width": 0.1, "height": 0.9, "timestamp": 12346.0}
        )
