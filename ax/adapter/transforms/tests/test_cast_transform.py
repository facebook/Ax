#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest.mock import patch

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import ExperimentData, extract_experiment_data
from ax.adapter.transforms.cast import Cast
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_timestamp_map_metric,
    get_experiment_with_observations,
    get_hierarchical_search_space,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal


class CastTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1.0, upper=5.0, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "b",
                    lower=1.0,
                    upper=5.0,
                    digits=2,
                    parameter_type=ParameterType.FLOAT,
                ),
                ChoiceParameter(
                    "c", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
                FixedParameter(name="d", parameter_type=ParameterType.INT, value=2),
            ],
            parameter_constraints=[],
        )
        self.t = Cast(search_space=self.search_space)
        self.hss = get_hierarchical_search_space()
        self.t_hss = Cast(search_space=self.hss)
        self.obs_feats_hss = ObservationFeatures(
            parameters={
                "model": "Linear",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
                "num_boost_rounds": 12,
            },
            trial_index=9,
            metadata=None,
        )
        self.obs_feats_hss_2 = ObservationFeatures(
            parameters={
                "model": "XGBoost",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
                "num_boost_rounds": 12,
            },
            trial_index=10,
            metadata=None,
        )
        self.obs_data = ObservationData(
            metric_signatures=["m1"],
            means=np.array([1.0]),
            covariance=np.array([[1.0]]),
        )

    def test_invalid_config(self) -> None:
        with self.assertRaisesRegex(UserInputError, "Unexpected config"):
            Cast(search_space=self.search_space, config={"flatten_hs": "foo"})

    def test_transform_observations_and_features(self) -> None:
        # Verify running the transform on already-casted features does nothing
        observation_features = [
            ObservationFeatures(parameters={"a": 1.2345, "b": 2.34, "c": "a", "d": 2})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

        # Test with transform_observations.
        obs = Observation(features=obs_ft2[0], data=self.obs_data, arm_name="arm")
        (tf_obs,) = self.t.transform_observations([obs])
        self.assertEqual(tf_obs.features, observation_features[0])
        self.assertEqual(tf_obs.data, self.obs_data)
        self.assertEqual(tf_obs.arm_name, "arm")

        # Check that the transform casts the parameter values when necessary.
        observation_features = [
            ObservationFeatures(parameters={"a": 1, "b": 2, "c": "a", "d": 2.1})
        ]
        expected = [
            ObservationFeatures(parameters={"a": 1.0, "b": 2.0, "c": "a", "d": 2})
        ]
        self.assertEqual(
            self.t.transform_observation_features(
                observation_features=observation_features
            ),
            expected,
        )

    def test_untransform_observation_features(self) -> None:
        # Verify running the transform on uncasted values properly converts them
        # (e.g. typing, rounding)
        observation_features = [
            ObservationFeatures(parameters={"a": 1, "b": 2.3466789, "c": "a", "d": 2.0})
        ]
        observation_features = self.t.untransform_observation_features(
            observation_features
        )
        self.assertEqual(
            observation_features,
            [ObservationFeatures(parameters={"a": 1.0, "b": 2.35, "c": "a", "d": 2})],
        )

    def test_flatten_hss_setting(self) -> None:
        t = Cast(search_space=self.hss)
        self.assertTrue(t.flatten_hss)
        t = Cast(search_space=self.hss, config={"flatten_hss": False})
        self.assertFalse(t.flatten_hss)
        self.assertFalse(self.t.flatten_hss)  # `self.t` does not have HSS
        self.assertTrue(self.t_hss.flatten_hss)  # `self.t_hss` does have HSS

    def test_transform_search_space_HSS(self) -> None:
        with patch.object(
            self.hss, "flatten", wraps=self.hss.flatten
        ) as mock_hss_flatten:
            flattened_search_space = self.t_hss.transform_search_space(
                search_space=self.hss
            )
        mock_hss_flatten.assert_called_once()
        self.assertIsNot(flattened_search_space, self.hss)
        self.assertFalse(flattened_search_space.is_hierarchical)

    def test_transform_observation_features_HSS(self) -> None:
        # Untransform the observation features first to cast them and
        # save their full parameterization in metadata.
        obs_feats = self.t_hss.untransform_observation_features(
            observation_features=[self.obs_feats_hss]
        )
        with patch.object(
            self.t_hss.search_space,
            "flatten_observation_features",
            wraps=self.t_hss.search_space.flatten_observation_features,
        ) as mock_flatten_obsf:
            transformed_obs_feats = self.t_hss.transform_observation_features(
                observation_features=obs_feats
            )
        mock_flatten_obsf.assert_called_once()

        for obsf in transformed_obs_feats:
            # Check that transformed obs feats have all the parameters
            for p_name in self.t_hss.search_space.parameters:
                self.assertIn(p_name, obsf.parameters)
            # Check that full parameterization is recorded in metadata
            self.assertEqual(
                # pyre-fixme[16]: Optional type has no attribute `get`.
                obsf.metadata.get(Keys.FULL_PARAMETERIZATION),
                self.obs_feats_hss.parameters,
            )

        # Perform one more roundtrip so parameterizations are cast to HSS.
        obs_feats = self.t_hss.untransform_observation_features(
            observation_features=transformed_obs_feats
        )
        new_transformed_obs_feats = self.t_hss.transform_observation_features(
            observation_features=obs_feats
        )
        for obsf in new_transformed_obs_feats:
            # Check that transformed obs feats have all the parameters
            for p_name in self.t_hss.search_space.parameters:
                self.assertIn(p_name, obsf.parameters)
            # Check that full parameterization is recorded in metadata
            self.assertEqual(
                obsf.metadata.get(Keys.FULL_PARAMETERIZATION),
                self.obs_feats_hss.parameters,
            )

    def test_transform_observation_features_HSS_dummy_values_settings(self) -> None:
        t = Cast(
            search_space=self.hss,
            config={
                "inject_dummy_values_to_complete_flat_parameterization": True,
            },
        )
        self.assertTrue(t.inject_dummy_values_to_complete_flat_parameterization)
        with patch.object(
            t.search_space,
            "flatten_observation_features",
            wraps=t.search_space.flatten_observation_features,
        ) as mock_flatten_obsf:
            t.transform_observation_features(observation_features=[self.obs_feats_hss])
        mock_flatten_obsf.assert_called_once()
        self.assertTrue(
            mock_flatten_obsf.call_args.kwargs[
                "inject_dummy_values_to_complete_flat_parameterization"
            ]
        )

    def test_untransform_observation_features_HSS(self) -> None:
        # Test transformation in one subtree of HSS.
        with patch.object(
            self.t_hss.search_space,
            "cast_observation_features",
            wraps=self.t_hss.search_space.cast_observation_features,
        ) as mock_cast_obsf:
            obs_feats = self.t_hss.untransform_observation_features(
                observation_features=[self.obs_feats_hss]
            )
        mock_cast_obsf.assert_called_once()

        self.assertEqual(len(obs_feats), 1)
        obsf = obs_feats[0]
        self.assertEqual(
            obsf.parameters,
            {
                "model": "Linear",
                "learning_rate": 0.01,
                "l2_reg_weight": 0.0001,
            },
        )
        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `get`.
            obsf.metadata.get(Keys.FULL_PARAMETERIZATION),
            self.obs_feats_hss.parameters,
        )

        # Test transformation in other subtree of HSS.
        obs_feats_2 = self.t_hss.untransform_observation_features(
            observation_features=[self.obs_feats_hss_2]
        )
        self.assertEqual(len(obs_feats_2), 1)
        obsf = obs_feats_2[0]
        self.assertEqual(
            obsf.parameters,
            {
                "model": "XGBoost",
                "num_boost_rounds": 12,
            },
        )
        self.assertEqual(
            obsf.metadata.get(Keys.FULL_PARAMETERIZATION),
            self.obs_feats_hss_2.parameters,
        )

    def test_cast_parameter_type_and_none(self) -> None:
        # This test covers removal of observations with Nones, casting
        # to correct parameter type and rounding to digits for RangeParameters.
        search_space = SearchSpace(
            parameters=[
                ChoiceParameter(
                    name="choice",
                    parameter_type=ParameterType.STRING,
                    values=["1", "2", "3"],
                ),
                RangeParameter(
                    name="range",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=5.0,
                    digits=1,
                ),
            ]
        )
        t = Cast(search_space=search_space)
        obs_features = [
            ObservationFeatures(parameters={"choice": None, "range": 5.0}),
            ObservationFeatures(parameters={"choice": 1, "range": 3}),
            ObservationFeatures(parameters={"choice": "2", "range": 3.567}),
        ]
        observations = [
            Observation(
                features=ft.clone(), data=deepcopy(self.obs_data), arm_name=f"{i}"
            )
            for i, ft in enumerate(obs_features)
        ]
        tf_obs_features = t.transform_observation_features(
            observation_features=obs_features
        )
        self.assertEqual(
            tf_obs_features,
            [
                ObservationFeatures(parameters={"choice": "1", "range": 3.0}),
                ObservationFeatures(parameters={"choice": "2", "range": 3.6}),
            ],
        )
        tf_observations = t.transform_observations(observations)
        expected = [
            Observation(
                features=ObservationFeatures(parameters={"choice": "1", "range": 3.0}),
                data=self.obs_data,
                arm_name="1",
            ),
            Observation(
                features=ObservationFeatures(parameters={"choice": "2", "range": 3.6}),
                data=self.obs_data,
                arm_name="2",
            ),
        ]
        self.assertEqual(tf_observations, expected)

    def test_transform_experiment_data_flatten(self) -> None:
        # Tests for flattening of hierarchical parameterizations.
        columns = [
            "model",
            "learning_rate",
            "l2_reg_weight",
            "num_boost_rounds",
            "metadata",
        ]
        arm_data = DataFrame.from_dict(  # Same data used in `setUp`.
            {
                (0, "0_0"): {
                    "model": "Linear",
                    "learning_rate": 0.01,
                    "l2_reg_weight": 0.0001,
                    "metadata": {
                        Keys.FULL_PARAMETERIZATION: {
                            "model": "Linear",
                            "learning_rate": 0.01,
                            "l2_reg_weight": 0.0001,
                            "num_boost_rounds": 12,
                        }
                    },
                },
                (1, "1_0"): {
                    "model": "XGBoost",
                    "num_boost_rounds": 12,
                    "metadata": {
                        Keys.FULL_PARAMETERIZATION: {
                            "model": "XGBoost",
                            "learning_rate": 0.01,
                            "l2_reg_weight": 0.0001,
                            "num_boost_rounds": 12,
                        }
                    },
                },
            },
            orient="index",
            columns=columns,
        )
        arm_data.index.names = ["trial_index", "arm_name"]
        experiment_data = ExperimentData(
            arm_data=arm_data, observation_data=DataFrame()
        )
        transformed = self.t_hss.transform_experiment_data(
            experiment_data=experiment_data
        )
        expected_arm_data = DataFrame.from_dict(
            {
                (0, "0_0"): {
                    "model": "Linear",
                    "learning_rate": 0.01,
                    "l2_reg_weight": 0.0001,
                    "num_boost_rounds": 12,
                    "metadata": {
                        Keys.FULL_PARAMETERIZATION: {
                            "model": "Linear",
                            "learning_rate": 0.01,
                            "l2_reg_weight": 0.0001,
                            "num_boost_rounds": 12,
                        }
                    },
                },
                (1, "1_0"): {
                    "model": "XGBoost",
                    "learning_rate": 0.01,
                    "l2_reg_weight": 0.0001,
                    "num_boost_rounds": 12,
                    "metadata": {
                        Keys.FULL_PARAMETERIZATION: {
                            "model": "XGBoost",
                            "learning_rate": 0.01,
                            "l2_reg_weight": 0.0001,
                            "num_boost_rounds": 12,
                        }
                    },
                },
            },
            orient="index",
            columns=columns,
        )
        expected_arm_data.index.names = ["trial_index", "arm_name"]
        expected_arm_data["num_boost_rounds"] = expected_arm_data[
            "num_boost_rounds"
        ].astype("Int64")
        assert_frame_equal(transformed.arm_data, expected_arm_data)

    def test_transform_experiment_data_flatten_with_missing_columns(self) -> None:
        columns = ["model", "learning_rate", "l2_reg_weight", "metadata"]
        arm_data = (
            DataFrame.from_dict(  # Data intentionally missing `num_boost_rounds`.
                {
                    (0, "0_0"): {
                        "model": "Linear",
                        "learning_rate": 0.01,
                        "l2_reg_weight": 0.0001,
                        "metadata": {
                            Keys.FULL_PARAMETERIZATION: {
                                "model": "Linear",
                                "learning_rate": 0.01,
                                "l2_reg_weight": 0.0001,
                            }
                        },
                    }
                },
                orient="index",
                columns=columns,
            )
        )
        arm_data.index.names = ["trial_index", "arm_name"]
        experiment_data = ExperimentData(
            arm_data=arm_data, observation_data=DataFrame()
        )
        transformed = self.t_hss.transform_experiment_data(
            experiment_data=experiment_data
        )
        expected_columns = set(columns + ["num_boost_rounds"])
        self.assertEqual(set(transformed.arm_data.columns), expected_columns)
        # Test with empty DF w/ missing columns.
        arm_data = arm_data.iloc[:0]
        arm_data.index.names = ["trial_index", "arm_name"]
        experiment_data = ExperimentData(
            arm_data=arm_data, observation_data=DataFrame()
        )
        transformed = self.t_hss.transform_experiment_data(
            experiment_data=experiment_data
        )
        self.assertEqual(set(transformed.arm_data.columns), expected_columns)

    def test_transform_experiment_data_cast(self) -> None:
        # Test for casting to the correct data type and dropping of Nones.
        experiment = get_experiment_with_observations(
            observations=[[0.0], [1.0], [2.0]],
            search_space=SearchSpace(
                parameters=[
                    RangeParameter(
                        name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=5
                    ),
                    RangeParameter(
                        name="y", parameter_type=ParameterType.FLOAT, lower=0, upper=5
                    ),
                    RangeParameter(
                        name="z", parameter_type=ParameterType.INT, lower=0, upper=5
                    ),
                ]
            ),
            parameterizations=[
                {"x": 1, "y": None},
                {"x": 2, "y": 2.0},
                {"x": 3, "y": 3},
            ],
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        transformed = Cast(
            search_space=experiment.search_space
        ).transform_experiment_data(experiment_data=deepcopy(experiment_data))
        # Arm data should drop row 0 and cast to float.
        # The missing column for `z` should be added and populated with NaNs.
        expected_arm_data = (
            experiment_data.arm_data.copy(deep=True)
            .iloc[[1, 2]]
            .astype({"x": float, "y": float})
        )
        expected_arm_data["z"] = None
        expected_arm_data["z"] = expected_arm_data["z"].astype("Int64")
        expected_arm_data = expected_arm_data[["x", "y", "z", "metadata"]]
        assert_frame_equal(transformed.arm_data, expected_arm_data)
        # Observation data should drop row 0.
        expected_obs_data = experiment_data.observation_data.copy(deep=True).iloc[
            [1, 2]
        ]
        assert_frame_equal(transformed.observation_data, expected_obs_data)

    def test_transform_experiment_data_cast_map_data(self) -> None:
        # Check that indexing for removal of NaNs works correctly with MapData.
        experiment = get_branin_experiment_with_timestamp_map_metric(
            with_trials_and_data=True
        )
        # Add some data for the last trial as well.
        experiment.fetch_data()
        # Update the last trial to mark parameterization as None.
        experiment.trials[2].arms[0]._parameters["x1"] = None

        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(
                fit_only_completed_map_metrics=False,
                latest_rows_per_group=None,
            ),
        )
        transformed_data = Cast(
            search_space=experiment.search_space
        ).transform_experiment_data(experiment_data=deepcopy(experiment_data))
        # Arm data should only include first three rows.
        assert_frame_equal(transformed_data.arm_data, experiment_data.arm_data.iloc[:2])
        # Observation data should include all but rows for last trial.
        assert_frame_equal(
            transformed_data.observation_data,
            experiment_data.observation_data.iloc[:-2],
        )
