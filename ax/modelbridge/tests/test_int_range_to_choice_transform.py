#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class IntRangeToChoiceTransformTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter("a", lower=1, upper=5, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[],
        )
        self.t = IntRangeToChoice(
            search_space=self.search_space,
            observations=[],
        )

    def testInit(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"a"})

    def testTransformObservationFeatures(self) -> None:
        observation_features = [ObservationFeatures(parameters={"a": 2, "b": "b"})]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, [ObservationFeatures(parameters={"a": 2, "b": "b"})])
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertTrue(isinstance(ss2.parameters["a"], ChoiceParameter))
        # pyre-fixme[16]: `Parameter` has no attribute `values`.
        self.assertTrue(ss2.parameters["a"].values, [1, 2, 3, 4, 5])

    def test_w_robust_search_space(self) -> None:
        rss = get_robust_search_space()
        # Transform a non-distributional parameter.
        t = IntRangeToChoice(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertIsInstance(rss_new.parameters.get("z"), ChoiceParameter)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = IntRangeToChoice(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertIsInstance(rss_new.parameters.get("z"), ChoiceParameter)
        # Error with distributional parameter.
        rss = get_robust_search_space(use_discrete=True)
        t = IntRangeToChoice(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)

    def test_num_choices(self) -> None:
        parameters = {
            "a": RangeParameter(
                "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
            ),
            "b": RangeParameter(
                "b", lower=1, upper=2, parameter_type=ParameterType.INT
            ),
            "c": ChoiceParameter(
                "c", parameter_type=ParameterType.STRING, values=["x1", "x2", "x3"]
            ),
            "d": RangeParameter(
                "d", lower=1, upper=9, parameter_type=ParameterType.INT
            ),
            "e": RangeParameter(
                "e", lower=3, upper=5, parameter_type=ParameterType.INT
            ),
        }
        search_space = SearchSpace(parameters=parameters.values())  # pyre-ignore[6]

        # Don't specify max_choices (should be set to inf)
        t = IntRangeToChoice(search_space=search_space)
        new_search_space = t.transform_search_space(search_space=search_space)
        self.assertEqual(len(new_search_space.parameters), len(parameters))
        self.assertEqual(t.max_choices, float("inf"))
        self.assertEqual(new_search_space.parameters["a"], parameters["a"])
        self.assertEqual(
            new_search_space.parameters["b"],
            ChoiceParameter(
                "b", values=[1, 2], is_ordered=True, parameter_type=ParameterType.INT
            ),
        )
        self.assertEqual(new_search_space.parameters["c"], parameters["c"])
        self.assertEqual(
            new_search_space.parameters["d"],
            ChoiceParameter(
                "d",
                values=list(range(1, 10)),  # pyre-ignore
                is_ordered=True,
                parameter_type=ParameterType.INT,
            ),
        )
        self.assertEqual(
            new_search_space.parameters["e"],
            ChoiceParameter(
                "e", values=[3, 4, 5], is_ordered=True, parameter_type=ParameterType.INT
            ),
        )

        # Set max_choices so parameter d isn't transformed
        t = IntRangeToChoice(search_space=search_space, config={"max_choices": 5})
        new_search_space = t.transform_search_space(search_space=search_space)
        self.assertEqual(len(new_search_space.parameters), len(parameters))
        self.assertEqual(t.max_choices, 5)
        self.assertEqual(new_search_space.parameters["a"], parameters["a"])
        self.assertEqual(
            new_search_space.parameters["b"],
            ChoiceParameter(
                "b", values=[1, 2], is_ordered=True, parameter_type=ParameterType.INT
            ),
        )
        self.assertEqual(new_search_space.parameters["c"], parameters["c"])
        self.assertEqual(new_search_space.parameters["d"], parameters["d"])
        self.assertEqual(
            new_search_space.parameters["e"],
            ChoiceParameter(
                "e", values=[3, 4, 5], is_ordered=True, parameter_type=ParameterType.INT
            ),
        )
