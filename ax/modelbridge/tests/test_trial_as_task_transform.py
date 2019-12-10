#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.utils.common.testutils import TestCase


class TrialAsTaskTransformTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=4, parameter_type=ParameterType.FLOAT
                )
            ]
        )
        self.training_feats = [
            ObservationFeatures({"x": 1}, trial_index=0),
            ObservationFeatures({"x": 2}, trial_index=0),
            ObservationFeatures({"x": 3}, trial_index=1),
            ObservationFeatures({"x": 4}, trial_index=2),
        ]
        self.t = TrialAsTask(
            search_space=self.search_space,
            observation_features=self.training_feats,
            observation_data=None,
        )
        self.bm = {
            "bp1": {0: "v1", 1: "v2", 2: "v3"},
            "bp2": {0: "u1", 1: "u1", 2: "u2"},
        }

        self.t2 = TrialAsTask(
            search_space=self.search_space,
            observation_features=self.training_feats,
            observation_data=None,
            config={"trial_level_map": self.bm},
        )
        self.t3 = TrialAsTask(
            search_space=self.search_space,
            observation_features=self.training_feats,
            observation_data=None,
            config={"trial_level_map": {}},
        )

    def testInit(self):
        self.assertEqual(
            self.t.trial_level_map, {"TRIAL_PARAM": {i: str(i) for i in range(3)}}
        )
        self.assertEqual(self.t.inverse_map, {str(i): i for i in range(3)})
        self.assertEqual(self.t2.trial_level_map, self.bm)
        self.assertIsNone(self.t2.inverse_map)
        # Test validation
        obsf = ObservationFeatures({"x": 2})
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.search_space,
                observation_features=self.training_feats + [obsf],
                observation_data=None,
            )
        bm = {"p": {0: "x1", 1: "x2"}}
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.search_space,
                observation_features=self.training_feats,
                observation_data=None,
                config={"trial_level_map": bm},
            )

    def testTransformObservationFeatures(self):
        obs_ft1 = deepcopy(self.training_feats)
        obs_ft2 = deepcopy(self.training_feats)
        obs_ft_trans1 = [
            ObservationFeatures({"x": 1, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x": 2, "TRIAL_PARAM": "0"}),
            ObservationFeatures({"x": 3, "TRIAL_PARAM": "1"}),
            ObservationFeatures({"x": 4, "TRIAL_PARAM": "2"}),
        ]
        obs_ft_trans2 = [
            ObservationFeatures({"x": 1, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x": 2, "bp1": "v1", "bp2": "u1"}),
            ObservationFeatures({"x": 3, "bp1": "v2", "bp2": "u1"}),
            ObservationFeatures({"x": 4, "bp1": "v3", "bp2": "u2"}),
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

    def testTransformSearchSpace(self):
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x", "TRIAL_PARAM"})
        p = ss2.parameters["TRIAL_PARAM"]
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"0", "1", "2"})
        self.assertTrue(p.is_task)
        ss2 = deepcopy(self.search_space)
        ss2 = self.t2.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x", "bp1", "bp2"})
        p = ss2.parameters["bp1"]
        self.assertTrue(isinstance(p, ChoiceParameter))
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"v1", "v2", "v3"})
        self.assertTrue(p.is_task)
        p = ss2.parameters["bp2"]
        self.assertTrue(isinstance(p, ChoiceParameter))
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        self.assertEqual(set(p.values), {"u1", "u2"})
        self.assertTrue(p.is_task)
