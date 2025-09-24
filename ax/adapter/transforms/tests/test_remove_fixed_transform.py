#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace, RobustSearchSpace, SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_robust_search_space,
)
from pandas.testing import assert_frame_equal, assert_series_equal


class RemoveFixedTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                FixedParameter("c", parameter_type=ParameterType.STRING, value="a"),
                DerivedParameter(
                    "d",
                    parameter_type=ParameterType.FLOAT,
                    expression_str="2.0 * a + 1.0",
                ),
            ]
        )
        self.t = RemoveFixed(search_space=self.search_space)

        # The ASCII tree is generated from https://www.text-tree-generator.com/
        # This gaint tree is here to bump up the test coverage.
        # root: FixedParameter
        # ├── (0) parent1: ChoiceParameter
        # │       ├── (0) child1: RangeParameter
        # │       ├── (1) middle: DerivedParameter
        # │       └── (1) child2: FixedParameter
        # │               └── (0) grandchild: FixedParameter
        # │                       ├── ("yee-haw") great_grandchild1: DerivedParameter
        # │                       └── ("yee-haw") great_grandchild2: RangeParameter
        # └── (1) parent2: ChoiceParameter
        #         ├── (0) child3: FixedParameter
        #         └── (1) child4: FixedParameter
        self.hierarchical_search_space = HierarchicalSearchSpace(
            parameters=[
                FixedParameter(
                    "root",
                    parameter_type=ParameterType.INT,
                    value=0,
                    dependents={0: ["parent1", "parent2"]},
                ),
                ChoiceParameter(
                    "parent1",
                    parameter_type=ParameterType.INT,
                    values=[0, 1],
                    dependents={0: ["child1"], 1: ["child2", "the_middle_child"]},
                ),
                ChoiceParameter(
                    "parent2",
                    parameter_type=ParameterType.INT,
                    values=[0, 1],
                    dependents={0: ["child3"], 1: ["child4"]},
                ),
                RangeParameter(
                    "child1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
                ),
                DerivedParameter(
                    "the_middle_child",
                    parameter_type=ParameterType.FLOAT,
                    expression_str="child1 + 1.0",
                ),
                FixedParameter(
                    "child2",
                    parameter_type=ParameterType.INT,
                    value=0,
                    dependents={0: ["grandchild"]},
                ),
                FixedParameter(
                    "grandchild",
                    parameter_type=ParameterType.STRING,
                    value="yee-haw",
                    dependents={"yee-haw": ["great_grandchild1", "great_grandchild2"]},
                ),
                DerivedParameter(
                    "great_grandchild1",
                    parameter_type=ParameterType.FLOAT,
                    expression_str="great_grandchild2 + 1.0",
                ),
                RangeParameter(
                    "great_grandchild2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
                FixedParameter(
                    "child3",
                    parameter_type=ParameterType.STRING,
                    value="hellow world",
                ),
                FixedParameter(
                    "child4",
                    parameter_type=ParameterType.STRING,
                    value="hellow world again",
                ),
            ]
        )
        self.t_hss = RemoveFixed(search_space=self.hierarchical_search_space)

    def test_Init(self) -> None:
        self.assertEqual(list(self.t.fixed_or_derived_parameters.keys()), ["c", "d"])

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a", "d": 5.4})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2, [ObservationFeatures(parameters={"a": 2.2, "b": "b"})]
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        observation_features = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "a", "d": 5.4})
        ]
        observation_features_different = [
            ObservationFeatures(parameters={"a": 2.2, "b": "b", "c": "b", "d": 5.4})
        ]
        # Fixed parameter is out of design. It will still get removed.
        t_obs = self.t.transform_observation_features(observation_features)
        t_obs_different = self.t.transform_observation_features(
            observation_features_different
        )
        self.assertEqual(t_obs, t_obs_different)

        # Test untransform with empty parameters (status quo case)
        # This would previously fail on p.compute(parameters=obsf.parameters)
        # when obsf.parameters is {} for DerivedParameter
        empty_obs_features = [ObservationFeatures(parameters={})]
        result = self.t.untransform_observation_features(empty_obs_features)
        # Should return unchanged empty observation features
        self.assertEqual(result, [ObservationFeatures(parameters={})])

    def test_TransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        for name in ("c", "d"):
            self.assertIsNone(ss2.parameters.get(name))

        # Test if dependents are removed properly.
        hss = self.hierarchical_search_space.clone()
        hss = self.t_hss.transform_search_space(hss)
        self.assertEqual(
            set(hss.parameters), {"parent1", "parent2", "child1", "great_grandchild2"}
        )
        self.assertEqual(
            hss.parameters["parent1"].dependents,
            {0: ["child1"], 1: ["great_grandchild2"]},
        )
        # Both children of `parent2` got removed. It's not hierarchical anymore.
        self.assertFalse(hss.parameters["parent2"].is_hierarchical)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        # Transform a non-distributional parameter.
        t = RemoveFixed(search_space=rss)
        rss = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(len(rss.parameter_distributions), 2)
        self.assertNotIn("d", rss.parameters)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            # pyre-fixme[16]: `SearchSpace` has no attribute `num_samples`.
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        rss.add_parameter(
            FixedParameter("d", parameter_type=ParameterType.STRING, value="a"),
        )
        t = RemoveFixed(search_space=rss)
        rss = t.transform_search_space(rss)
        self.assertIsInstance(rss, RobustSearchSpace)
        self.assertEqual(len(rss.parameters.keys()), 4)
        self.assertEqual(len(rss.parameter_distributions), 2)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(len(rss._environmental_variables), 2)
        self.assertNotIn("d", rss.parameters)

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"a": 1.0, "b": "a", "c": "a", "d": 3.0},
            {"a": 2.0, "b": "b", "c": "a", "d": 5.0},
            {"a": 3.0, "b": "c", "c": "a", "d": 7.0},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        copy_experiment_data = deepcopy(experiment_data)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=copy_experiment_data
        )
        for name in ("c", "d"):
            self.assertIn(name, experiment_data.arm_data)
            # Check that `c` has been removed.
            self.assertNotIn(name, transformed_data.arm_data)

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
        self.assertIs(
            transformed_data.observation_data, copy_experiment_data.observation_data
        )

        # Test with no fixed features.
        search_space = self.t.transform_search_space(search_space=self.search_space)
        t = RemoveFixed(search_space=search_space)
        copy_transformed_data = deepcopy(transformed_data)
        transformed_data_2 = t.transform_experiment_data(
            experiment_data=copy_transformed_data
        )
        self.assertEqual(transformed_data_2, transformed_data)
        self.assertIs(
            transformed_data_2.observation_data, copy_transformed_data.observation_data
        )
        self.assertIsNot(transformed_data_2.arm_data, copy_transformed_data.arm_data)
        assert_frame_equal(transformed_data_2.arm_data, copy_transformed_data.arm_data)
