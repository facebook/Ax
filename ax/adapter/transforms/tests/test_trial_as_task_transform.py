#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

import numpy as np
from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.trial_as_task import TrialAsTask
from ax.core.arm import Arm
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.exceptions.core import UnsupportedError
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_robust_search_space
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class TrialAsTaskTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.exp = get_branin_experiment(with_status_quo=True, with_batch=True)
        self.adapter = Adapter(
            search_space=self.exp.search_space,
            generator=Generator(),
            experiment=self.exp,
        )
        self.exp.new_batch_trial().add_arm(
            Arm(parameters={"x1": 0, "x2": 0}, name="status_quo")
        ).add_arm(Arm(parameters={"x1": 1, "x2": 1}))
        self.exp.new_batch_trial().add_arm(
            Arm(parameters={"x1": 0, "x2": 0}, name="status_quo")
        ).add_arm(Arm(parameters={"x1": 3, "x2": 3}))
        for t in self.exp.trials.values():
            t.mark_running(no_runner_required=True)
        self.exp.trials[0].mark_completed()
        self.exp.fetch_data()

        self.experiment_data = extract_experiment_data(
            experiment=self.exp, data_loader_config=DataLoaderConfig()
        )
        self.training_feats = [
            ObservationFeatures({"x1": 1, "x2": 1}, trial_index=0),
            ObservationFeatures({"x1": 2, "x2": 2}, trial_index=0),
            ObservationFeatures({"x1": 3, "x2": 3}, trial_index=1),
            ObservationFeatures({"x1": 4, "x2": 4}, trial_index=2),
        ]
        self.training_obs = [
            Observation(
                data=ObservationData(
                    metric_names=[], means=np.array([]), covariance=np.empty((0, 0))
                ),
                features=obsf,
            )
            for obsf in self.training_feats
        ]

        self.t = TrialAsTask(
            search_space=self.exp.search_space,
            experiment_data=self.experiment_data,
            adapter=self.adapter,
        )
        self.bm = {
            "bp1": {0: "v1", 1: "v2", 2: "v3"},
            "bp2": {0: "u1", 1: "u1", 2: "u2"},
        }

        self.t2 = TrialAsTask(
            search_space=self.exp.search_space,
            observations=self.training_obs,
            adapter=self.adapter,
            config={"trial_level_map": self.bm},
        )
        self.t3 = TrialAsTask(
            search_space=self.exp.search_space,
            observations=self.training_obs,
            adapter=self.adapter,
            config={"trial_level_map": {}},
        )
        # test string trial indices
        self.bm2 = {
            p_name: {str(k): v for k, v in value_dict.items()}
            for p_name, value_dict in self.bm.items()
        }
        self.t4 = TrialAsTask(
            search_space=self.exp.search_space,
            observations=self.training_obs,
            adapter=self.adapter,
            config={"trial_level_map": self.bm2, "target_trial": 2},
        )

    def test_Init(self) -> None:
        self.assertEqual(
            self.t.trial_level_map, {"TRIAL_PARAM": {i: str(i) for i in range(3)}}
        )
        self.assertEqual(self.t.inverse_map, {str(i): i for i in range(3)})
        # based on `get_target_trial_index``, the longest running trial is trial 1
        self.assertEqual(self.t.target_values, {"TRIAL_PARAM": "1"})
        self.assertEqual(self.t2.trial_level_map, self.bm)
        self.assertIsNone(self.t2.inverse_map)
        self.assertEqual(self.t2.target_values, {"bp1": "v2", "bp2": "u1"})
        # check that strings were converted to integers
        self.assertEqual(self.t4.trial_level_map, self.bm)
        self.assertIsNone(self.t4.inverse_map)
        self.assertEqual(self.t4.target_values, {"bp1": "v3", "bp2": "u2"})
        # Test validation
        obsf = ObservationFeatures({"x1": 2, "x2": 2})
        obs = Observation(
            data=ObservationData([], np.array([]), np.empty((0, 0))), features=obsf
        )
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.exp.search_space,
                observations=self.training_obs + [obs],
                adapter=self.adapter,
            )
        bm = {"p": {0: "y", 1: "z"}}
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.exp.search_space,
                observations=self.training_obs,
                adapter=self.adapter,
                config={"trial_level_map": bm},
            )

    def test_TransformObservationFeatures(self) -> None:
        obs_ft1 = deepcopy(self.training_feats)
        obs_ft2 = deepcopy(self.training_feats)
        obs_ft_trans1 = [
            ObservationFeatures({"x1": 1, "x2": 1, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x1": 2, "x2": 2, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x1": 3, "x2": 3, "TRIAL_PARAM": "1"}),
            ObservationFeatures({"x1": 4, "x2": 4, "TRIAL_PARAM": "2"}),
        ]
        obs_ft_trans2 = [
            ObservationFeatures({"x1": 1, "x2": 1, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x1": 2, "x2": 2, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x1": 3, "x2": 3, "bp1": "v2", "bp2": "u1"}),
            ObservationFeatures({"x1": 4, "x2": 4, "bp1": "v3", "bp2": "u2"}),
        ]
        obs_ft1 = self.t.transform_observation_features(obs_ft1)
        self.assertEqual(obs_ft1, obs_ft_trans1)
        obs_ft2 = self.t2.transform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, obs_ft_trans2)
        obs_ft1 = self.t.untransform_observation_features(obs_ft1)
        self.assertEqual(obs_ft1, self.training_feats)
        obs_ft3 = self.t2.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft3[0], ObservationFeatures({}))
        obs_ft4 = deepcopy(self.training_feats)
        obs_ft4 = self.t3.untransform_observation_features(obs_ft4)
        self.assertEqual(obs_ft4, self.training_feats)

    def test_TransformObservationFeaturesWithoutTrialIndex(self) -> None:
        obs_ft_no_trial_index = deepcopy(self.training_feats)
        obs_ft_no_trial_index.append(
            ObservationFeatures(
                {"x1": 20, "x2": 20},
            )
        )
        obs_ft_trans = [
            ObservationFeatures({"x1": 1, "x2": 1, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x1": 2, "x2": 2, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x1": 3, "x2": 3, "TRIAL_PARAM": "1"}),
            ObservationFeatures({"x1": 4, "x2": 4, "TRIAL_PARAM": "2"}),
            ObservationFeatures({"x1": 20, "x2": 20, "TRIAL_PARAM": "2"}),
        ]
        obs_ft_trans2 = [
            ObservationFeatures({"x1": 1, "x2": 1, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x1": 2, "x2": 2, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x1": 3, "x2": 3, "bp1": "v2", "bp2": "u1"}),
            ObservationFeatures({"x1": 4, "x2": 4, "bp1": "v3", "bp2": "u2"}),
            ObservationFeatures({"x1": 20, "x2": 20, "bp1": "v3", "bp2": "u2"}),
        ]

        # test can transform and untransform with no config
        obs_ft_no_trial_index_transformed = self.t.transform_observation_features(
            obs_ft_no_trial_index
        )
        self.assertEqual(obs_ft_no_trial_index_transformed, obs_ft_trans)
        untransformed = self.t.untransform_observation_features(
            obs_ft_no_trial_index_transformed
        )
        # test can transform and untransform with config trial level map
        self.assertEqual(untransformed, obs_ft_no_trial_index)
        obs_ft_no_index_transformed_2 = self.t2.transform_observation_features(
            obs_ft_no_trial_index
        )
        self.assertEqual(obs_ft_no_index_transformed_2, obs_ft_trans2)
        # can transform and untransform are equal with empty config
        obs_ft4 = self.t3.untransform_observation_features(obs_ft_no_trial_index)
        self.assertEqual(obs_ft4, obs_ft_no_trial_index)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.exp.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x1", "x2", "TRIAL_PARAM"})
        p = assert_is_instance(ss2.parameters["TRIAL_PARAM"], ChoiceParameter)
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"0", "1", "2"})
        self.assertTrue(p.is_task)
        self.assertFalse(p.is_ordered)
        self.assertEqual(p.target_value, "1")
        ss2 = deepcopy(self.exp.search_space)
        ss2 = self.t2.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x1", "x2", "bp1", "bp2"})
        p = ss2.parameters["bp1"]
        self.assertTrue(isinstance(p, ChoiceParameter))
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"v1", "v2", "v3"})
        self.assertTrue(p.is_task)
        self.assertFalse(p.is_ordered)
        self.assertEqual(p.target_value, "v2")
        p = ss2.parameters["bp2"]
        self.assertTrue(isinstance(p, ChoiceParameter))
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"u1", "u2"})
        self.assertTrue(p.is_task)
        self.assertTrue(p.is_ordered)  # 2 choices so always ordered
        self.assertEqual(p.target_value, "u1")
        t = TrialAsTask(
            search_space=self.exp.search_space,
            observations=self.training_obs,
            adapter=self.adapter,
            config={
                "trial_level_map": {
                    "trial_index": {0: 10, 1: 11, 2: 12},
                },
                "target_trial": 1,
            },
        )
        ss2 = deepcopy(self.exp.search_space)
        ss2 = t.transform_search_space(ss2)
        p = assert_is_instance(ss2.parameters["trial_index"], ChoiceParameter)
        self.assertEqual(p.parameter_type, ParameterType.INT)
        self.assertEqual(set(p.values), {10, 11, 12})
        self.assertTrue(p.is_ordered)
        self.assertTrue(p.is_task)
        self.assertEqual(p.target_value, 11)

    def test_w_robust_search_space(self) -> None:
        rss = get_robust_search_space()
        # Raises an error in __init__.
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            TrialAsTask(
                search_space=rss,
                observations=self.training_obs,
                adapter=self.adapter,
            )

    def test_less_than_two_trials(self) -> None:
        # test transform is a no-op with less than two trials
        exp = get_branin_experiment()
        exp.new_trial().add_arm(Arm(parameters={"x1": 1, "x2": 1}))
        adapter = Adapter(
            search_space=exp.search_space,
            generator=Generator(),
            experiment=exp,
        )
        training_obs = self.training_obs[:1]
        t = TrialAsTask(
            search_space=exp.search_space,
            observations=training_obs,
            adapter=adapter,
        )
        self.assertEqual(t.trial_level_map, {})
        training_feats = [training_obs[0].features]
        training_feats_clone = deepcopy(training_feats)
        self.assertEqual(
            t.transform_observation_features(training_feats_clone), training_feats
        )
        self.assertEqual(
            t.untransform_observation_features(training_feats), training_feats_clone
        )
        ss2 = exp.search_space.clone()
        self.assertEqual(t.transform_search_space(ss2), exp.search_space)

    def test_less_than_two_levels(self) -> None:
        # test transform is a no-op with less than two trials
        exp = get_branin_experiment()
        exp.new_trial().add_arm(Arm(parameters={"x1": 1, "x2": 1}))
        exp.new_trial().add_arm(Arm(parameters={"x1": 2, "x2": 2}))
        adapter = Adapter(
            search_space=exp.search_space,
            generator=Generator(),
            experiment=exp,
        )
        training_obs = self.training_obs[:1]
        t = TrialAsTask(
            search_space=exp.search_space,
            observations=training_obs,
            adapter=adapter,
            config={"trial_level_map": {"t": {0: "v1", 1: "v1"}}},
        )
        self.assertEqual(t.trial_level_map, {})
        training_feats = [training_obs[0].features]
        training_feats_clone = deepcopy(training_feats)
        self.assertEqual(
            t.transform_observation_features(training_feats_clone), training_feats
        )
        self.assertEqual(
            t.untransform_observation_features(training_feats), training_feats_clone
        )
        ss2 = exp.search_space.clone()
        self.assertEqual(t.transform_search_space(ss2), exp.search_space)

    def test_transform_experiment_data(self) -> None:
        # Experiment data has 16 rows for trial 0 and 2 rows each for trials 1 & 2.
        def make_expected(values: tuple[str, str, str]) -> list[str]:
            return [values[0]] * 16 + [values[1]] * 2 + [values[2]] * 2

        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Check that arm data has trial parameter added.
        self.assertEqual(
            transformed_data.arm_data["TRIAL_PARAM"].to_list(),
            make_expected(values=("0", "1", "2")),
        )
        # Check that other columns are unchanged.
        assert_frame_equal(
            transformed_data.arm_data.drop(columns="TRIAL_PARAM"),
            self.experiment_data.arm_data,
        )
        # Check that observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, self.experiment_data.observation_data
        )

        # Test with alternative transform config.
        transformed_data = self.t2.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        self.assertEqual(
            transformed_data.arm_data["bp1"].to_list(),
            make_expected(values=("v1", "v2", "v3")),
        )
        self.assertEqual(
            transformed_data.arm_data["bp2"].to_list(),
            make_expected(values=("u1", "u1", "u2")),
        )
