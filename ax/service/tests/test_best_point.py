# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

from ax.core.optimization_config import MultiObjectiveOptimizationConfig

from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_experiment_with_trial,
)


class TestBestPointMixin(TestCase):
    def test_get_trace(self) -> None:
        # Alias for easier access.
        get_trace = BestPointMixin._get_trace

        # Single objective, minimize.
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]], minimize=True
        )
        self.assertEqual(get_trace(exp), [11, 10, 9, 9, 5])
        # Same experiment with maximize via new optimization config.
        opt_conf = not_none(exp.optimization_config).clone()
        opt_conf.objective.minimize = False
        self.assertEqual(get_trace(exp, opt_conf), [11, 11, 11, 15, 15])

        # Scalarized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [2, 2], [3, 3]],
            scalarized=True,
        )
        self.assertEqual(get_trace(exp), [2, 4, 6])

        # Multi objective.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 100], [1, 2], [3, 3], [2, 4], [2, 1]],
        )
        self.assertEqual(get_trace(exp), [1, 1, 2, 9, 11, 11])

        # W/o ObjectiveThresholds (infering ObjectiveThresholds from nadir point)
        checked_cast(
            MultiObjectiveOptimizationConfig, exp.optimization_config
        ).objective_thresholds = []
        self.assertEqual(get_trace(exp), [0.0, 0.0, 2.0, 8.0, 11.0, 11.0])

        # W/ constraints.
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 1, 1]],
            constrained=True,
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ first objective being minimized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 2], [3, 3], [-2, 4], [2, 1]], minimize=True
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(get_trace(exp), [])

    def test_get_hypervolume(self) -> None:
        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(BestPointMixin._get_hypervolume(exp, Mock()), 0.0)
