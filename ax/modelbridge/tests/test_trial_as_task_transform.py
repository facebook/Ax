#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class TrialAsTaskTransformTest(TestCase):
    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=4, parameter_type=ParameterType.FLOAT
                )
            ]
        )
        self.training_feats = [
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures({"x": 1}, trial_index=0),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures({"x": 2}, trial_index=0),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures({"x": 3}, trial_index=1),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures({"x": 4}, trial_index=2),
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
            search_space=self.search_space,
            observations=self.training_obs,
        )
        self.bm = {
            "bp1": {0: "v1", 1: "v2", 2: "v3"},
            "bp2": {0: "u1", 1: "u1", 2: "u2"},
        }

        self.t2 = TrialAsTask(
            search_space=self.search_space,
            observations=self.training_obs,
            config={"trial_level_map": self.bm},
        )
        self.t3 = TrialAsTask(
            search_space=self.search_space,
            observations=self.training_obs,
            config={"trial_level_map": {}},
        )

    def testInit(self) -> None:
        self.assertEqual(
            self.t.trial_level_map, {"TRIAL_PARAM": {i: str(i) for i in range(3)}}
        )
        self.assertEqual(self.t.inverse_map, {str(i): i for i in range(3)})
        self.assertEqual(self.t2.trial_level_map, self.bm)
        self.assertIsNone(self.t2.inverse_map)
        # Test validation
        obsf = ObservationFeatures({"x": 2})
        obs = Observation(
            data=ObservationData([], np.array([]), np.empty((0, 0))), features=obsf
        )
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.search_space,
                observations=self.training_obs + [obs],
            )
        bm = {"p": {0: "x1", 1: "x2"}}
        with self.assertRaises(ValueError):
            TrialAsTask(
                search_space=self.search_space,
                observations=self.training_obs,
                config={"trial_level_map": bm},
            )

    def testTransformObservationFeatures(self) -> None:
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

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        self.assertEqual(set(ss2.parameters.keys()), {"x", "TRIAL_PARAM"})
        p = ss2.parameters["TRIAL_PARAM"]
        self.assertEqual(p.parameter_type, ParameterType.STRING)
        # pyre-fixme[16]: `Parameter` has no attribute `values`.
        self.assertEqual(set(p.values), {"0", "1", "2"})
        # pyre-fixme[16]: `Parameter` has no attribute `is_task`.
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

    def test_w_robust_search_space(self) -> None:
        rss = get_robust_search_space()
        # Raises an error in __init__.
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            TrialAsTask(
                search_space=rss,
                observations=[],
            )
