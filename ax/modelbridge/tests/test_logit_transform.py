#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.transforms.logit import Logit
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space
from scipy.special import expit, logit


class LogitTransformTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=0.9,
                    upper=0.999,
                    parameter_type=ParameterType.FLOAT,
                    logit_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        self.t = Logit(
            search_space=self.search_space,
            observations=[],
        )
        self.search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=0.1,
                    upper=0.3,
                    parameter_type=ParameterType.FLOAT,
                    logit_scale=True,
                    is_fidelity=True,
                    target_value=0.123,
                )
            ]
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _create_logit_parameter(self, lower, upper, log_scale=False):
        return RangeParameter(
            "x",
            lower=lower,
            upper=upper,
            parameter_type=ParameterType.FLOAT,
            log_scale=log_scale,
            logit_scale=True,
        )

    def testInit(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"x"})

    def testTransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 0.95, "a": 2, "b": "c"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": logit(0.95), "a": 2, "b": "c"})],
        )
        # Untransform
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        x_true = expit(logit(0.95))
        self.assertAlmostEqual(x_true, 0.95)  # Need to be careful with rounding here
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": x_true, "a": 2, "b": "c"})],
        )

    def testInvalidSettings(self) -> None:
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.1, upper=0.9, log_scale=True)
        self.assertEqual("Can't use both log and logit.", str(cm.exception))

        str_exc = "Logit requires lower > 0 and upper < 1"
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.0, upper=0.5)
        self.assertEqual(str_exc, str(cm.exception))
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.3, upper=1.0)
        self.assertEqual(str_exc, str(cm.exception))
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.5, upper=10.0)
        self.assertEqual(str_exc, str(cm.exception))

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        # pyre-fixme[16]: `Parameter` has no attribute `lower`.
        self.assertEqual(ss2.parameters["x"].lower, logit(0.9))
        # pyre-fixme[16]: `Parameter` has no attribute `upper`.
        self.assertEqual(ss2.parameters["x"].upper, logit(0.999))
        t2 = Logit(
            search_space=self.search_space_with_target,
            observations=[],
        )
        ss_target = deepcopy(self.search_space_with_target)
        t2.transform_search_space(ss_target)
        self.assertEqual(ss_target.parameters["x"].target_value, logit(0.123))
        self.assertEqual(ss_target.parameters["x"].lower, logit(0.1))
        self.assertEqual(ss_target.parameters["x"].upper, logit(0.3))

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space(use_discrete=True)
        # pyre-fixme[16]: `Parameter` has no attribute `set_logit_scale`.
        rss.parameters["y"].set_logit_scale(True)
        # Transform a non-distributional parameter.
        t = Logit(
            search_space=rss,
            observations=[],
        )
        t.transform_search_space(rss)
        # pyre-fixme[16]: Optional type has no attribute `logit_scale`.
        self.assertFalse(rss.parameters.get("y").logit_scale)
        # Error with distributional parameter.
        rss.parameters["x"].set_logit_scale(True)
        t = Logit(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)
