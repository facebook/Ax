#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest import mock

import pandas as pd
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.time_as_feature import TimeAsFeature
from ax.core.observation import ObservationFeatures
from ax.core.observation_utils import observations_from_data
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import unixtime_to_pandas_ts
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_robust_search_space,
)
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class TimeAsFeatureTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=4, parameter_type=ParameterType.FLOAT
                )
            ]
        )
        experiment = get_experiment_with_observations(
            observations=[[0.1], [0.2], [0.3], [0.4]],
            search_space=self.search_space,
            parameterizations=[{"x": i + 1} for i in range(4)],
            additional_data_columns=[
                {
                    "start_time": unixtime_to_pandas_ts(float(i)),
                    "end_time": unixtime_to_pandas_ts(float(i + 1 + i)),
                }
                for i in range(4)
            ],
        )
        self.training_obs = observations_from_data(
            experiment=experiment, data=experiment.lookup_data()
        )
        self.training_feats = [obs.features for obs in self.training_obs]
        self.time_return_value = 5.0
        time_patcher = mock.patch(
            "ax.adapter.transforms.time_as_feature.time",
            return_value=self.time_return_value,
        )
        self.time_patcher = time_patcher.start()
        self.addCleanup(time_patcher.stop)

        self.experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        self.t = TimeAsFeature(
            search_space=self.search_space,
            experiment_data=self.experiment_data,
        )

    def test_init__(self) -> None:
        self.assertEqual(self.t.current_time, self.time_return_value)
        self.assertEqual(self.t.min_duration, 1.0)
        self.assertEqual(self.t.max_duration, 4.0)
        self.assertEqual(self.t.duration_range, 3.0)
        self.assertEqual(self.t.min_start_time, 0.0)
        self.assertEqual(self.t.max_start_time, 3.0)

        # Test validation with missing start time data.
        experiment = get_experiment_with_observations(
            observations=[[0.1], [0.2]],
            search_space=self.search_space,
            additional_data_columns=[
                {
                    "start_time": unixtime_to_pandas_ts(0.0),
                    "end_time": unixtime_to_pandas_ts(1.0),
                },
                {
                    "start_time": pd.NaT,
                    "end_time": pd.NaT,
                },
            ],
        )
        experiment_data_no_time = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        with self.assertRaisesRegex(
            ValueError,
            "Unable to use TimeAsFeature since not all observations have "
            "start time specified.",
        ):
            TimeAsFeature(
                search_space=self.search_space,
                experiment_data=experiment_data_no_time,
            )

        # Create experiment data with just one observation.
        experiment = get_experiment_with_observations(
            observations=[[0.1]],
            search_space=self.search_space,
            parameterizations=[{"x": 1}],
            additional_data_columns=[
                {
                    "start_time": unixtime_to_pandas_ts(0.0),
                    "end_time": unixtime_to_pandas_ts(1.0),
                }
            ],
        )
        experiment_data_one = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        t2 = TimeAsFeature(
            search_space=self.search_space,
            experiment_data=experiment_data_one,
        )
        self.assertEqual(t2.duration_range, 1.0)

    def test_TransformObservationFeatures(self) -> None:
        obs_ft1 = deepcopy(self.training_feats)
        obs_ft_trans1 = deepcopy(self.training_feats)
        for i, obs in enumerate(obs_ft_trans1):
            obs.parameters.update({"start_time": float(i), "duration": 1 / 3 * i})
        obs_ft1 = self.t.transform_observation_features(obs_ft1)
        self.assertEqual(obs_ft1, obs_ft_trans1)
        obs_ft1 = self.t.untransform_observation_features(obs_ft1)
        self.assertEqual(obs_ft1, self.training_feats)
        # test transforming observation features that do not have
        # start_time/end_time
        obsf = [ObservationFeatures({"x": 2.5})]
        obsf_trans = self.t.transform_observation_features(obsf)
        self.assertEqual(
            obsf_trans[0],
            ObservationFeatures(
                {"x": 2.5, "duration": 0.5, "start_time": self.time_return_value}
            ),
        )
        # test untransforming observation features that do not have
        # start/end time (important for fixed features in MOO when un-
        # transforming objective thresholds)
        obsf_trans = [ObservationFeatures({"x": 2.5})]
        obsf_untrans = self.t.untransform_observation_features(obsf_trans)
        self.assertEqual(obsf_untrans, obsf_trans)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x", "start_time", "duration"})
        p = assert_is_instance(ss2.parameters["start_time"], RangeParameter)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 0.0)
        self.assertEqual(p.upper, 3.0)
        p = assert_is_instance(ss2.parameters["duration"], RangeParameter)
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 0.0)
        self.assertEqual(p.upper, 1.0)

    def test_w_robust_search_space(self) -> None:
        rss = get_robust_search_space()
        # Raises an error in __init__.
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            TimeAsFeature(
                search_space=rss,
                experiment_data=self.experiment_data,
            )

    def test_with_experiment_data(self) -> None:
        experiment = get_experiment_with_observations(
            parameterizations=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
            observations=[[1.0], [2.0], [3.0]],
            additional_data_columns=[
                {
                    "start_time": unixtime_to_pandas_ts(0.0),
                    "end_time": unixtime_to_pandas_ts(1.0),
                },
                {
                    "start_time": unixtime_to_pandas_ts(1.0),
                    "end_time": unixtime_to_pandas_ts(3.0),
                },
                {"start_time": unixtime_to_pandas_ts(2.0)},
            ],
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        # Check transform initialization.
        t = TimeAsFeature(
            search_space=experiment.search_space,
            experiment_data=experiment_data,
        )
        # from time() mock in `setUp`.
        self.assertEqual(t.current_time, 5.0)
        self.assertEqual(t.min_duration, 1.0)
        self.assertEqual(t.max_duration, 3.0)
        self.assertEqual(t.duration_range, 2.0)
        self.assertEqual(t.min_start_time, 0.0)
        self.assertEqual(t.max_start_time, 2.0)
        # Transform experiment data.
        transformed_data = t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )
        # Observation data is unmodified.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )
        # Arm data has start_time and duration columns.
        self.assertEqual(
            set(transformed_data.arm_data.columns),
            {"x", "metadata", "start_time", "duration"},
        )
        assert_frame_equal(
            transformed_data.arm_data[["x", "metadata"]],
            experiment_data.arm_data[["x", "metadata"]],
        )
        # Check that start_time and duration are correct.
        self.assertEqual(
            transformed_data.arm_data["start_time"].tolist(), [0.0, 1.0, 2.0]
        )
        self.assertEqual(
            transformed_data.arm_data["duration"].tolist(), [0.0, 0.5, 1.0]
        )
