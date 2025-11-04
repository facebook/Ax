#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import DataRequiredError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal, assert_series_equal


class SearchSpaceToChoiceTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )
        parameterizations: list[TParameterization] = [
            {"a": 2, "b": "a"},
            {"a": 3, "b": "b"},
            {"a": 3, "b": "c"},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        self.experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        self.observation_features = [
            ObservationFeatures(parameters=p) for p in parameterizations
        ]
        self.signature_to_parameterization = {
            Arm(parameters=p).signature: p for p in parameterizations
        }
        self.transformed_features = [
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 2, "b": "a"}).signature}
            ),
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 3, "b": "b"}).signature}
            ),
            ObservationFeatures(
                parameters={"arms": Arm(parameters={"a": 3, "b": "c"}).signature}
            ),
        ]
        self.observations = [
            Observation(
                data=ObservationData([], np.array([]), np.empty((0, 0))), features=obsf
            )
            for obsf in self.observation_features
        ]
        self.t = SearchSpaceToChoice(
            search_space=self.search_space, experiment_data=self.experiment_data
        )
        # Convert first observation to experiment data for t2
        experiment_single = get_experiment_with_observations(
            observations=[[1.0]],
            search_space=self.search_space,
            parameterizations=[{"a": 1, "b": "a"}],
        )
        experiment_data_single = extract_experiment_data(
            experiment=experiment_single, data_loader_config=DataLoaderConfig()
        )
        self.t2 = SearchSpaceToChoice(
            search_space=self.search_space, experiment_data=experiment_data_single
        )
        self.t3 = SearchSpaceToChoice(
            search_space=self.search_space,
            experiment_data=self.experiment_data,
            config={"use_ordered": True},
        )

    def test_validation(self) -> None:
        # Test with no data.
        with self.assertRaisesRegex(DataRequiredError, "non-empty data"):
            SearchSpaceToChoice(search_space=self.search_space)
        # Test with empty experiment data.
        with self.assertRaisesRegex(DataRequiredError, "non-empty data"):
            SearchSpaceToChoice(
                search_space=self.search_space,
                experiment_data=extract_experiment_data(
                    experiment=Experiment(search_space=self.search_space),
                    data_loader_config=DataLoaderConfig(),
                ),
            )

        # Test error if there are fidelities.
        ss = SearchSpace(
            parameters=[
                RangeParameter(
                    "a",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "fidelity"):
            SearchSpaceToChoice(search_space=ss, experiment_data=self.experiment_data)

    def test_TransformSearchSpace(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = ChoiceParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            values=list(self.t.signature_to_parameterization.keys()),
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

        # With use_ordered
        ss2 = self.search_space.clone()
        ss2 = self.t3.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = ChoiceParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            values=list(self.t.signature_to_parameterization.keys()),
            is_ordered=True,
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

    def test_TransformSearchSpaceWithFixedParam(self) -> None:
        ss2 = self.search_space.clone()
        ss2 = self.t2.transform_search_space(ss2)
        self.assertEqual(len(ss2.parameters), 1)
        expected_parameter = FixedParameter(
            name="arms",
            parameter_type=ParameterType.STRING,
            value=list(self.t2.signature_to_parameterization.keys())[0],
        )
        self.assertEqual(ss2.parameters.get("arms"), expected_parameter)

    def test_TransformObservationFeatures(self) -> None:
        obs_ft2 = deepcopy(self.observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.transformed_features)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, self.observation_features)

        # Testing transform empty parameters dict
        # Both transform and untransform should leave the param dict intact
        empty_obs_param = ObservationFeatures(parameters={}, trial_index=0)
        tsfm_empty_obs_param = self.t.transform_observation_features([empty_obs_param])[
            0
        ]
        self.assertEqual(tsfm_empty_obs_param, empty_obs_param)
        untsfm_empty_obs_param = self.t.untransform_observation_features(
            [tsfm_empty_obs_param]
        )[0]
        self.assertEqual(untsfm_empty_obs_param, empty_obs_param)

    def test_transform_experiment_data(self) -> None:
        # Empty data is returned unchanged.
        empty_data = extract_experiment_data(
            experiment=Experiment(search_space=self.search_space),
            data_loader_config=DataLoaderConfig(),
        )
        copy_empty_data = deepcopy(empty_data)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=copy_empty_data
        )
        self.assertIs(copy_empty_data, transformed_data)
        self.assertEqual(transformed_data, empty_data)

        # Data is transformed to the signature.
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Columns only include arms and metadata.
        self.assertEqual(set(transformed_data.arm_data), {"arms", "metadata"})
        # Metadata is unchanged.
        assert_series_equal(
            transformed_data.arm_data["metadata"],
            self.experiment_data.arm_data["metadata"],
        )
        # Arms are replaced by signatures.
        expected_arms = list(self.signature_to_parameterization)
        self.assertEqual(transformed_data.arm_data["arms"].tolist(), expected_arms)
        # Observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, self.experiment_data.observation_data
        )
