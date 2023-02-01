#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest import mock

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import OrderConstraint, SumConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class IntToFloatTransformTest(TestCase):
    def setUp(self) -> None:
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.FLOAT),
            RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
            RangeParameter("d", lower=1, upper=3, parameter_type=ParameterType.INT),
            ChoiceParameter(
                "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
            ),
        ]
        self.search_space = SearchSpace(
            # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
            #  `List[Union[ChoiceParameter, RangeParameter]]`.
            parameters=parameters,
            parameter_constraints=[
                OrderConstraint(
                    lower_parameter=parameters[0], upper_parameter=parameters[1]
                )
            ],
        )
        self.t = IntToFloat(
            search_space=self.search_space,
            observations=[],
        )
        self.t2 = IntToFloat(
            search_space=self.search_space,
            observations=[],
            config={"rounding": "randomized"},
        )

    def testInit(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"a", "d"})

    def testTransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 3})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 3})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        # Let the transformed space be a float, verify it becomes an int.
        obs_ft3 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 2.9})
        ]
        obs_ft3 = self.t.untransform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3, observation_features)

        # Test forward transform on partial observation
        obs_ft4 = [ObservationFeatures(parameters={"x": 2.2, "d": 3})]
        obs_ft4 = self.t.transform_observation_features(obs_ft4)
        self.assertEqual(obs_ft4, [ObservationFeatures(parameters={"x": 2.2, "d": 3})])
        self.assertTrue(isinstance(obs_ft4[0].parameters["d"], float))
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

        # test untransforming integer params that are outside of the range, but within
        # 0.5 of the range limit
        obs_ft6 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 0.6, "b": "b", "d": 3.3})
        ]
        obs_ft6 = self.t.untransform_observation_features(obs_ft6)
        self.assertEqual(
            obs_ft6,
            [ObservationFeatures(parameters={"x": 2.2, "a": 1, "b": "b", "d": 3})],
        )

    def testTransformObservationFeaturesRandomized(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t2.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 4})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t2.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertTrue(ss2.parameters["a"].parameter_type, ParameterType.FLOAT)
        self.assertTrue(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)

    def testRoundingWithConstrainedIntRanges(self) -> None:
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=3, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
            #  `List[RangeParameter]`.
            parameters=parameters,
            parameter_constraints=[
                # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
                #  `List[RangeParameter]`.
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=5)
            ],
        )
        t = IntToFloat(
            search_space=constrained_int_search_space,
            observations=[],
        )
        self.assertEqual(t.rounding, "randomized")
        observation_features = [ObservationFeatures(parameters={"x": 2.6, "y": 2.6})]
        self.assertTrue(
            constrained_int_search_space.check_membership(
                t.untransform_observation_features(
                    observation_features=observation_features
                )[0].parameters
            )
        )

    @mock.patch(
        "ax.modelbridge.transforms.int_to_float.DEFAULT_MAX_ROUND_ATTEMPTS", 100
    )
    def testRoundingWithImpossiblyConstrainedIntRanges(self) -> None:
        parameters = [
            RangeParameter("x", lower=1, upper=5, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=5, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
            #  `List[RangeParameter]`.
            parameters=parameters,
            parameter_constraints=[
                # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
                #  `List[RangeParameter]`.
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=3)
            ],
        )
        t = IntToFloat(
            search_space=constrained_int_search_space,
            observations=[],
        )
        self.assertEqual(t.rounding, "randomized")
        observation_features = [ObservationFeatures(parameters={"x": 2.6, "y": 2.6})]
        self.assertFalse(
            constrained_int_search_space.check_membership(
                t.untransform_observation_features(
                    observation_features=observation_features
                )[0].parameters
            )
        )
        # Round something that is outside the search space and make sure it satisfies
        # the domain bounds even if it doesn't satisfy the parameter constraints.
        for _ in range(10):
            observation_features = [
                ObservationFeatures(parameters={"x": 0.51, "y": 4.44})
            ]
            untransformed_t = t.untransform_observation_features(
                observation_features=observation_features
            )[0].parameters
            self.assertEqual(untransformed_t, {"x": 1, "y": 4})

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        # Transform a non-distributional parameter.
        t = IntToFloat(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `parameter_type`.
            rss_new.parameters.get("z").parameter_type,
            ParameterType.FLOAT,
        )
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = IntToFloat(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertEqual(
            rss_new.parameters.get("z").parameter_type, ParameterType.FLOAT
        )
        # Error with distributional parameter.
        rss = get_robust_search_space(use_discrete=True)
        t = IntToFloat(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)
