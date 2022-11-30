#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.utils import (
    ClosestLookupDict,
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_multi_objective_optimization_config

OBSERVATION_DATA = [
    Observation(
        features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}),
        data=ObservationData(
            means=np.array([1.0, 2.0, 6.0]),
            covariance=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
            metric_names=["m1", "m2", "m3"],
        ),
        arm_name="1_1",
    )
]


class TransformUtilsTest(TestCase):
    def test_closest_lookup_dict(self) -> None:
        # test empty lookup
        d = ClosestLookupDict()
        with self.assertRaises(RuntimeError):
            d[0]
        # basic test
        keys = (1.0, 2, 4)
        vals = ("a", "b", "c")
        d = ClosestLookupDict(zip(keys, vals))
        for k, v in zip(keys, vals):
            self.assertEqual(d[k], v)
        self.assertEqual(d[2.5], "b")
        self.assertEqual(d[0], "a")
        self.assertEqual(d[6], "c")
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 1st param expected `Number` but got `str`.
            d["str_key"] = 3

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=(OBSERVATION_DATA),
    )
    def test_derelativize_optimization_config_with_raw_status_quo(self, _) -> None:
        optimization_config = get_multi_objective_optimization_config()
        dummy_search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0, 20),
                RangeParameter("y", ParameterType.FLOAT, 0, 20),
            ]
        )
        modelbridge = ModelBridge(
            search_space=dummy_search_space,
            model=None,
            transforms=[],
            experiment=Experiment(dummy_search_space, "test"),
            data=Data(),
            optimization_config=optimization_config,
            status_quo_name="1_1",
        )
        new_opt_config = derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config,
            modelbridge=modelbridge,
            observations=OBSERVATION_DATA,
        )
        expected_bound_values = {"m1": 0.9975, "m2": 1.995, "m3": 5.985}
        for oc in new_opt_config.all_constraints:
            self.assertFalse(oc.relative)
            expected_bound_value = expected_bound_values[oc.metric.name]
            self.assertEqual(oc.bound, expected_bound_value)
