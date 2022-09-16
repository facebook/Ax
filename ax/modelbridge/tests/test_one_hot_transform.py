#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.one_hot import OH_PARAM_INFIX, OneHot
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class OneHotTransformTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                ChoiceParameter(
                    "c",
                    parameter_type=ParameterType.BOOL,
                    values=[True, False],
                    is_ordered=False,
                ),
                ChoiceParameter(
                    "d",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5)
            ],
        )
        self.t = OneHot(
            search_space=self.search_space,
            observations=[],
        )
        self.t2 = OneHot(
            search_space=self.search_space,
            observations=[],
            config={"rounding": "randomized"},
        )

        self.transformed_features = ObservationFeatures(
            parameters={
                "x": 2.2,
                "a": 2,
                "b" + OH_PARAM_INFIX + "_0": 0,
                "b" + OH_PARAM_INFIX + "_1": 1,
                "b" + OH_PARAM_INFIX + "_2": 0,
                # Only two choices => one parameter.
                "c" + OH_PARAM_INFIX: 0,
                "d": 10.0,
            }
        )
        self.observation_features = ObservationFeatures(
            parameters={"x": 2.2, "a": 2, "b": "b", "c": False, "d": 10.0}
        )

    def testInit(self) -> None:
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c"])
        self.assertEqual(list(self.t2.encoded_parameters.keys()), ["b", "c"])

    def testTransformObservationFeatures(self) -> None:
        observation_features = [self.observation_features]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [self.transformed_features])
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test partial transform
        obs_ft3 = [ObservationFeatures(parameters={"x": 2.2, "b": "b"})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3[0],
            ObservationFeatures(
                parameters={
                    "x": 2.2,
                    "b" + OH_PARAM_INFIX + "_0": 0,
                    "b" + OH_PARAM_INFIX + "_1": 1,
                    "b" + OH_PARAM_INFIX + "_2": 0,
                }
            ),
        )
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testRandomizedTransform(self) -> None:
        observation_features = [self.observation_features]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t2.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [self.transformed_features])
        obs_ft2 = self.t2.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameter type fixed.
        self.assertEqual(ss2.parameters["x"].parameter_type, ParameterType.FLOAT)
        self.assertEqual(ss2.parameters["a"].parameter_type, ParameterType.INT)
        self.assertEqual(
            ss2.parameters["b" + OH_PARAM_INFIX + "_0"].parameter_type,
            ParameterType.FLOAT,
        )
        self.assertEqual(
            ss2.parameters["b" + OH_PARAM_INFIX + "_1"].parameter_type,
            ParameterType.FLOAT,
        )
        self.assertEqual(
            ss2.parameters["c" + OH_PARAM_INFIX].parameter_type, ParameterType.FLOAT
        )
        self.assertEqual(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)

        # Parameter range fixed to [0,1].
        # pyre-fixme[16]: `Parameter` has no attribute `lower`.
        self.assertEqual(ss2.parameters["b" + OH_PARAM_INFIX + "_0"].lower, 0.0)
        # pyre-fixme[16]: `Parameter` has no attribute `upper`.
        self.assertEqual(ss2.parameters["b" + OH_PARAM_INFIX + "_1"].upper, 1.0)
        self.assertEqual(ss2.parameters["c" + OH_PARAM_INFIX].lower, 0.0)
        self.assertEqual(ss2.parameters["c" + OH_PARAM_INFIX].upper, 1.0)

        # Ensure we error if we try to transform a fidelity parameter
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_fidelity=True,
                    target_value="c",
                )
            ]
        )
        t = OneHot(search_space=ss3, observations=[])
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        # Transform a non-distributional parameter.
        t = OneHot(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(len(rss_new.parameters.keys()), 4)
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertNotIn("c", rss_new.parameters)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = OneHot(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(len(rss_new.parameters.keys()), 4)
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertNotIn("c", rss_new.parameters)
