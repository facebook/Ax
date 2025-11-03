#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.logit import Logit
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.special import expit, logit


class LogitTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.t = Logit(search_space=self.search_space)
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

    def _create_logit_parameter(
        self, lower: float, upper: float, log_scale: bool = False
    ) -> RangeParameter:
        return RangeParameter(
            "x",
            lower=lower,
            upper=upper,
            parameter_type=ParameterType.FLOAT,
            log_scale=log_scale,
            logit_scale=True,
        )

    def test_Init(self) -> None:
        self.assertEqual(self.t.transform_parameters, {"x"})

    def test_TransformObservationFeatures(self) -> None:
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

    def test_InvalidSettings(self) -> None:
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.1, upper=0.9, log_scale=True)
        self.assertEqual("x can't use both log and logit.", str(cm.exception))

        str_exc = "x logit requires lower > 0 and upper < 1"
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.0, upper=0.5)
        self.assertEqual(str_exc, str(cm.exception))
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.3, upper=1.0)
        self.assertEqual(str_exc, str(cm.exception))
        with self.assertRaises(UserInputError) as cm:
            self._create_logit_parameter(lower=0.5, upper=10.0)
        self.assertEqual(str_exc, str(cm.exception))

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        # pyre-fixme[16]: `Parameter` has no attribute `lower`.
        self.assertEqual(ss2.parameters["x"].lower, logit(0.9))
        # pyre-fixme[16]: `Parameter` has no attribute `upper`.
        self.assertEqual(ss2.parameters["x"].upper, logit(0.999))
        t2 = Logit(search_space=self.search_space_with_target)
        ss_target = deepcopy(self.search_space_with_target)
        t2.transform_search_space(ss_target)
        self.assertEqual(ss_target.parameters["x"].target_value, logit(0.123))
        self.assertEqual(ss_target.parameters["x"].lower, logit(0.1))
        self.assertEqual(ss_target.parameters["x"].upper, logit(0.3))

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"x": 0.2, "a": 1, "b": "a"},
            {"x": 0.5, "a": 2, "b": "b"},
            {"x": 0.7, "a": 3, "b": "c"},
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

        # Check that `x` has been log-transformed.
        assert_series_equal(
            transformed_data.arm_data["x"], logit(experiment_data.arm_data["x"])
        )

        # Check that other columns remain unchanged.
        assert_series_equal(
            transformed_data.arm_data["a"], experiment_data.arm_data["a"]
        )
        assert_series_equal(
            transformed_data.arm_data["b"], experiment_data.arm_data["b"]
        )

        # Check that observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )
