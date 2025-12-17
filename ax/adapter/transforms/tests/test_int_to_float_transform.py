#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest import mock

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.int_to_float import IntToFloat
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint, SumConstraint
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal, assert_series_equal
from pyre_extensions import assert_is_instance


class IntToFloatTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        parameters: list[Parameter] = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.FLOAT),
            RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
            RangeParameter("d", lower=1, upper=3, parameter_type=ParameterType.INT),
            ChoiceParameter(
                "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
            ),
        ]
        self.search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[ParameterConstraint(inequality="x <= a")],
        )
        self.t = IntToFloat(search_space=self.search_space)
        self.t2 = IntToFloat(
            search_space=self.search_space, config={"rounding": "randomized"}
        )
        self.t3 = IntToFloat(search_space=self.search_space, config={"min_choices": 3})
        self.search_space_with_log = self.search_space.clone()
        assert_is_instance(
            self.search_space_with_log.parameters["a"], RangeParameter
        )._log_scale = True
        self.t4 = IntToFloat(
            search_space=self.search_space_with_log,
            config={"min_choices": 3},
        )

    def test_Init(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"a", "d"})

    def test_transform_observation_features(self) -> None:
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

        # With min_choices. Only d should be transformed.
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t3.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 3})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], int))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t3.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        # With log_scale & min_choices. Both a & d should get transformed.
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t4.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 3})],
        )
        self.assertTrue(isinstance(obs_ft2[0].parameters["a"], float))
        self.assertTrue(isinstance(obs_ft2[0].parameters["d"], float))
        obs_ft2 = self.t4.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        # Let the transformed space be a float, verify it becomes an int.
        obs_ft3 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 2.9})
        ]
        obs_ft3 = self.t.untransform_observation_features(obs_ft3)
        self.assertEqual(obs_ft3, observation_features)

        # With min_choices. Only d should become an int.
        obs_ft3 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 2.9})
        ]
        obs_ft3 = self.t3.untransform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 3})],
        )

        # With log_scale & min_choices. Both a & d should become ints.
        obs_ft3 = [
            ObservationFeatures(parameters={"x": 2.2, "a": 2.2, "b": "b", "d": 2.9})
        ]
        obs_ft3 = self.t4.untransform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3,
            [ObservationFeatures(parameters={"x": 2.2, "a": 2, "b": "b", "d": 3})],
        )

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

    def test_TransformObservationFeaturesRandomized(self) -> None:
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

    def test_transform_search_space(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertTrue(ss2.parameters["a"].parameter_type, ParameterType.FLOAT)
        self.assertTrue(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)
        # With min_choices, only d should be transformed.
        ss2 = deepcopy(self.search_space)
        ss2 = self.t3.transform_search_space(ss2)
        self.assertTrue(ss2.parameters["a"].parameter_type, ParameterType.INT)
        self.assertTrue(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)
        # With log_scale & min_choices. Both a & d should get transformed.
        ss2 = deepcopy(self.search_space)
        ss2 = self.t4.transform_search_space(ss2)
        self.assertTrue(ss2.parameters["a"].parameter_type, ParameterType.FLOAT)
        self.assertTrue(ss2.parameters["d"].parameter_type, ParameterType.FLOAT)

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"x": 1.0, "a": 1, "b": "a", "d": 1},
            {"x": 1.5, "a": 2, "b": "b", "d": 3},
            {"x": 1.7, "a": 3, "b": "c", "d": 5},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Check that the transformed data has float types for 'a' and 'd'.
        self.assertEqual(transformed_data.arm_data["a"].dtype, np.float64)
        self.assertEqual(transformed_data.arm_data["d"].dtype, np.float64)

        # Check that the values are the same (just converted to float).
        assert_series_equal(
            transformed_data.arm_data["a"], experiment_data.arm_data["a"].astype(float)
        )
        assert_series_equal(
            transformed_data.arm_data["d"], experiment_data.arm_data["d"].astype(float)
        )

        # Check that other columns remain unchanged.
        assert_series_equal(
            transformed_data.arm_data["x"], experiment_data.arm_data["x"]
        )
        assert_series_equal(
            transformed_data.arm_data["b"], experiment_data.arm_data["b"]
        )

        # Check that observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

        # Test with min_choices=3 (only 'd' should be transformed).
        transformed_data = self.t3.transform_experiment_data(
            experiment_data=experiment_data
        )

        # Check that only 'd' is converted to float.
        self.assertNotEqual(transformed_data.arm_data["a"].dtype, np.float64)
        self.assertEqual(transformed_data.arm_data["d"].dtype, np.float64)

        # Test with log_scale parameter (both 'a' and 'd' should be transformed).
        transformed_data = self.t4.transform_experiment_data(
            experiment_data=experiment_data
        )

        # Check that both 'a' and 'd' are converted to float.
        self.assertEqual(transformed_data.arm_data["a"].dtype, np.float64)
        self.assertEqual(transformed_data.arm_data["d"].dtype, np.float64)

    def test_RoundingWithConstrainedIntRanges(self) -> None:
        parameters = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=3, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
                #  `List[RangeParameter]`.
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=5)
            ],
        )
        t = IntToFloat(search_space=constrained_int_search_space)
        self.assertEqual(t.rounding, "randomized")
        observation_features = [ObservationFeatures(parameters={"x": 2.6, "y": 2.6})]
        self.assertTrue(
            constrained_int_search_space.check_membership(
                t.untransform_observation_features(
                    observation_features=observation_features
                )[0].parameters
            )
        )

        # test empty parameters
        observation_features = [ObservationFeatures(parameters={})]
        untransformed_t = t.untransform_observation_features(
            observation_features=observation_features
        )[0].parameters
        self.assertEqual(untransformed_t, {})
        # test wrong number of parameters
        observation_features = [
            ObservationFeatures(
                parameters={
                    "x": 1,
                }
            )
        ]
        msg = (
            "Either all parameters must be provided or no parameters"
            " should be provided, when there are parameter"
            " constraints involving integers."
        )
        with self.assertRaisesRegex(ValueError, msg):
            t.untransform_observation_features(
                observation_features=observation_features
            )

    @mock.patch("ax.adapter.transforms.int_to_float.DEFAULT_MAX_ROUND_ATTEMPTS", 100)
    def test_RoundingWithImpossiblyConstrainedIntRanges(self) -> None:
        parameters = [
            RangeParameter("x", lower=1, upper=5, parameter_type=ParameterType.INT),
            RangeParameter("y", lower=1, upper=5, parameter_type=ParameterType.INT),
        ]
        constrained_int_search_space = SearchSpace(
            parameters=parameters,
            parameter_constraints=[
                # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
                #  `List[RangeParameter]`.
                SumConstraint(parameters=parameters, is_upper_bound=True, bound=3)
            ],
        )
        t = IntToFloat(search_space=constrained_int_search_space)
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
