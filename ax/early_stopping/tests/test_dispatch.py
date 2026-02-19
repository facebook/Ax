#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.early_stopping.dispatch import get_default_ess_or_none
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
)
from pyre_extensions import none_throws


class TestGetDefaultEss(TestCase):
    def test_get_default_ess_single_objective_unconstrained(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric()
        strategy = none_throws(get_default_ess_or_none(experiment=exp))
        self.assertIsInstance(strategy, PercentileEarlyStoppingStrategy)

        # Verify configuration.
        self.assertEqual(strategy.percentile_threshold, 50)
        self.assertEqual(strategy.min_curves, 3)
        self.assertEqual(strategy.min_progression, 0.2)
        self.assertEqual(strategy.max_progression, 0.9)
        self.assertTrue(strategy.normalize_progressions)
        self.assertEqual(strategy.n_best_trials_to_complete, 3)
        self.assertTrue(strategy.check_safe)

    def test_get_default_ess_null_conditions(self) -> None:
        # Checks that None is returned for currently unsupported conditions.
        for exp in [
            get_branin_experiment(has_optimization_config=True),  # No MapMetric.
            get_branin_experiment(has_optimization_config=False),  # No opt config.
            get_branin_experiment_with_timestamp_map_metric(  # Outcome constraint.
                with_outcome_constraint=True
            ),
            # Multi-objective.
            get_branin_experiment_with_timestamp_map_metric(multi_objective=True),
        ]:
            self.assertIsNone(get_default_ess_or_none(experiment=exp))
