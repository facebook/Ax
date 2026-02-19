#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter import Adapter
from ax.adapter.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_multi_objective_optimization_config,
)


class TransformUtilsTest(TestCase):
    def test_derelativize_optimization_config_with_raw_status_quo(self) -> None:
        optimization_config = get_multi_objective_optimization_config()
        experiment = get_experiment_with_observations(
            observations=[[1.0, 2.0, 6.0]], optimization_config=optimization_config
        )
        experiment._status_quo = experiment.trials[0].arms[0]

        adapter = Adapter(experiment=experiment, generator=Generator())
        new_opt_config = derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config, adapter=adapter
        )
        expected_bound_values = {"m1": 0.9975, "m2": 1.995, "m3": 5.985}
        for oc in new_opt_config.all_constraints:
            self.assertFalse(oc.relative)
            expected_bound_value = expected_bound_values[oc.metric.name]
            self.assertEqual(oc.bound, expected_bound_value)
