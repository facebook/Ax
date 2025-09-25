#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.metadata_to_task import MetadataToTask
from ax.core.observation import observations_from_data
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class MetadataToTaskTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=1, upper=20
                )
            ]
        )
        self.experiment = get_experiment_with_observations(
            observations=[[0.0], [1.0]],
            search_space=self.search_space,
            parameterizations=[{"x": 0.0}, {"x": 1.0}],
            candidate_metadata=[
                {Keys.TASK_FEATURE_NAME.value: 0},
                {Keys.TASK_FEATURE_NAME.value: 1},
            ],
        )
        self.observations = observations_from_data(
            experiment=self.experiment, data=self.experiment.lookup_data()
        )
        self.experiment_data = extract_experiment_data(
            experiment=self.experiment, data_loader_config=DataLoaderConfig()
        )
        self.transform_config: TConfig = {"task_values": [0, 1]}
        self.t = MetadataToTask(
            experiment_data=self.experiment_data, config=self.transform_config
        )

    def test_Init(self) -> None:
        self.assertEqual(len(self.t._parameter_list), 1)
        p = self.t._parameter_list[0]
        # check that the parameter options are specified in a sensible manner
        # by default if the user does not specify them explicitly
        self.assertEqual(p.name, Keys.TASK_FEATURE_NAME.value)
        self.assertEqual(p.parameter_type, ParameterType.INT)
        self.assertIsInstance(p, ChoiceParameter)
        self.assertEqual(p.values, [0, 1])
        self.assertFalse(p.is_fidelity)
        self.assertTrue(p.is_task)
        self.assertEqual(p.target_value, 0)

    def test_TransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)

        self.assertSetEqual(
            set(ss2.parameters.keys()),
            {"x", Keys.TASK_FEATURE_NAME.value},
        )

        p = assert_is_instance(
            ss2.parameters[Keys.TASK_FEATURE_NAME.value], ChoiceParameter
        )
        self.assertEqual(p.name, Keys.TASK_FEATURE_NAME.value)
        self.assertEqual(p.parameter_type, ParameterType.INT)
        self.assertEqual(p.values, [0, 1])
        self.assertFalse(p.is_fidelity)
        self.assertTrue(p.is_task)
        self.assertEqual(p.target_value, 0)

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [obs.features for obs in self.observations]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        expected_obs_ft2 = []
        for i, obs_ft in enumerate(observation_features):
            new_obs_ft = obs_ft.clone(
                replace_parameters={
                    **obs_ft.parameters,
                    Keys.TASK_FEATURE_NAME.value: i,
                }
            )
            del new_obs_ft.metadata[Keys.TASK_FEATURE_NAME]
            expected_obs_ft2.append(new_obs_ft)
        self.assertEqual(obs_ft2, expected_obs_ft2)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def test_transform_experiment_data(self) -> None:
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Check that arm data now has a new column for the transformed parameter.
        expected_task_values = [0, 1]
        self.assertEqual(
            transformed_data.arm_data[Keys.TASK_FEATURE_NAME.value].tolist(),
            expected_task_values,
        )
        # Remaining columns are unchanged.
        assert_frame_equal(
            transformed_data.arm_data.drop(columns=[Keys.TASK_FEATURE_NAME.value]),
            self.experiment_data.arm_data,
        )
        # Observation data is not changed.
        assert_frame_equal(
            transformed_data.observation_data,
            self.experiment_data.observation_data,
        )
