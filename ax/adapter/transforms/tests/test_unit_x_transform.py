#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.unit_x import UnitX
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class UnitXTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "y", lower=1, upper=2, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "z",
                    lower=1,
                    upper=2,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "y": 1}, bound=0.5),
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5),
            ],
        )
        self.t = UnitX(search_space=self.search_space)
        self.search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )

    def test_Init(self) -> None:
        self.assertEqual(self.t.bounds, {"x": (1.0, 3.0), "y": (1.0, 2.0)})

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 2, "y": 2, "z": 2, "a": 2, "b": "b"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={"x": 0.5, "y": 1.0, "z": 2, "a": 2, "b": "b"}
                )
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform partial observation
        obs_ft3 = [ObservationFeatures(parameters={"x": 3.0, "z": 2})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3[0],
            ObservationFeatures(parameters={"x": 1.0, "z": 2}),
        )
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameters transformed
        true_bounds = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (1.0, 2.0),
            "a": (1.0, 2.0),
        }
        for p_name, (l, u) in true_bounds.items():
            self.assertEqual(
                assert_is_instance(ss2.parameters[p_name], RangeParameter).lower, l
            )
            self.assertEqual(
                assert_is_instance(ss2.parameters[p_name], RangeParameter).upper, u
            )
        self.assertEqual(
            assert_is_instance(ss2.parameters["b"], ChoiceParameter).values,
            ["a", "b", "c"],
        )
        self.assertEqual(len(ss2.parameters), 5)
        # Constraints transformed
        self.assertEqual(
            ss2.parameter_constraints[0].constraint_dict, {"x": -1.0, "y": 1.0}
        )
        self.assertEqual(ss2.parameter_constraints[0].bound, 0.0)
        self.assertEqual(
            ss2.parameter_constraints[1].constraint_dict, {"x": -1.0, "a": 1.0}
        )
        self.assertEqual(ss2.parameter_constraints[1].bound, 1.0)

        # Test transform of target value
        t = UnitX(search_space=self.search_space_with_target)
        t.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, 1.0
        )

    def test_TransformNewSearchSpace(self) -> None:
        new_ss = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1.5, upper=2.0, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "y", lower=1.25, upper=2.0, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "z",
                    lower=1.0,
                    upper=1.5,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter(
                    "a", lower=0.0, upper=2, parameter_type=ParameterType.INT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "y": 1}, bound=0.5),
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5),
            ],
        )
        self.t.transform_search_space(new_ss)
        # Parameters transformed
        true_bounds = {
            "x": [0.25, 0.5],
            "y": [0.25, 1.0],
            "z": [1.0, 1.5],
            "a": [0, 2],
        }
        for p_name, (l, u) in true_bounds.items():
            p = assert_is_instance(new_ss.parameters[p_name], RangeParameter)
            self.assertEqual(p.lower, l)
            self.assertEqual(p.upper, u)
        self.assertEqual(
            assert_is_instance(new_ss.parameters["b"], ChoiceParameter).values,
            ["a", "b", "c"],
        )
        self.assertEqual(len(new_ss.parameters), 5)
        # # Constraints transformed
        self.assertEqual(
            new_ss.parameter_constraints[0].constraint_dict, {"x": -1.0, "y": 1.0}
        )
        self.assertEqual(new_ss.parameter_constraints[0].bound, 0.0)
        self.assertEqual(
            new_ss.parameter_constraints[1].constraint_dict, {"x": -1.0, "a": 1.0}
        )
        self.assertEqual(new_ss.parameter_constraints[1].bound, 1.0)

        # Test transform of target value
        t = UnitX(search_space=self.search_space_with_target)
        new_search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=2,
                    parameter_type=ParameterType.FLOAT,
                    is_fidelity=True,
                    target_value=2,
                )
            ]
        )
        t.transform_search_space(new_search_space_with_target)
        self.assertEqual(new_search_space_with_target.parameters["x"].target_value, 0.5)

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"x": 1.0, "y": 1.5, "z": 1.0, "a": 1, "b": "b"},
            {"x": 2.0, "y": 2.0, "z": 2.0, "a": 2, "b": "b"},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Check that `x` and `y` have been transformed.
        expected = DataFrame(
            index=transformed_data.arm_data.index,
            data={
                "x": [0.0, 0.5],
                "y": [0.5, 1.0],
            },
            columns=["x", "y"],
        )
        assert_frame_equal(transformed_data.arm_data[["x", "y"]], expected)

        # Remaining columns are unchanged.
        # "z" is log-scale and "a" is in, so they're not transformed.
        cols = ["z", "a", "b", "metadata"]
        assert_frame_equal(
            transformed_data.arm_data[cols], experiment_data.arm_data[cols]
        )
        # Observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )
