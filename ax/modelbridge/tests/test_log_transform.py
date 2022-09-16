#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.log import Log
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class LogTransformTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        self.t = Log(
            search_space=self.search_space,
            observations=[],
        )
        self.search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )

    def testInit(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"x"})

    def testTransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "c"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": math.log10(2.2), "a": 2, "b": "c"})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["x"], float))
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        # pyre-fixme[16]: `Parameter` has no attribute `lower`.
        self.assertEqual(ss2.parameters["x"].lower, math.log10(1))
        # pyre-fixme[16]: `Parameter` has no attribute `upper`.
        self.assertEqual(ss2.parameters["x"].upper, math.log10(3))
        t2 = Log(
            search_space=self.search_space_with_target,
            observations=[],
        )
        t2.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, math.log10(3)
        )

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space(lb=1.0, use_discrete=True)
        # pyre-fixme[16]: `Parameter` has no attribute `set_log_scale`.
        rss.parameters["y"].set_log_scale(True)
        # Transform a non-distributional parameter.
        t = Log(
            search_space=rss,
            observations=[],
        )
        t.transform_search_space(rss)
        # pyre-fixme[16]: Optional type has no attribute `log_scale`.
        self.assertFalse(rss.parameters.get("y").log_scale)
        # Error with distributional parameter.
        rss.parameters["x"].set_log_scale(True)
        t = Log(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)
