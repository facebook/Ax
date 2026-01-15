#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.early_stopping.experiment_replay import (
    estimate_hypothetical_early_stopping_savings,
)
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
)
from pyre_extensions import none_throws


class TestEstimateHypotheticalEss(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Experiment with MapMetric for tests that need a valid default ESS.
        self.exp = get_branin_experiment_with_timestamp_map_metric()
        self.metric = none_throws(self.exp.optimization_config).objective.metric

    def test_estimate_hypothetical_ess_no_default_strategy(self) -> None:
        """Test that UnsupportedError is raised when no default ESS is available."""
        # Non-MapMetric experiment has no default ESS.
        exp = get_branin_experiment(has_optimization_config=True)
        metric = none_throws(exp.optimization_config).objective.metric

        with self.assertRaises(UnsupportedError) as e:
            estimate_hypothetical_early_stopping_savings(
                experiment=exp,
                metric=metric,
            )

        self.assertIn(
            "No default early stopping strategy available",
            str(e.exception),
        )

    def test_estimate_hypothetical_ess_no_progression_data(self) -> None:
        """Test that UnsupportedError is raised when experiment has no progression
        data."""
        with patch(
            "ax.early_stopping.experiment_replay.replay_experiment",
            return_value=None,
        ):
            with self.assertRaises(UnsupportedError) as e:
                estimate_hypothetical_early_stopping_savings(
                    experiment=self.exp,
                    metric=self.metric,
                )

            self.assertIn(
                "Experiment data does not have progression data for replay",
                str(e.exception),
            )

    def test_estimate_hypothetical_ess_success(self) -> None:
        """Test that savings are returned when replay succeeds."""
        with (
            patch(
                "ax.early_stopping.experiment_replay.replay_experiment",
            ) as mock_replay,
            patch(
                "ax.early_stopping.experiment_replay.estimate_early_stopping_savings",
                return_value=0.25,
            ) as mock_estimate,
        ):
            result = estimate_hypothetical_early_stopping_savings(
                experiment=self.exp,
                metric=self.metric,
            )

            self.assertEqual(result, 0.25)
            mock_replay.assert_called_once()
            mock_estimate.assert_called_once()

    def test_estimate_hypothetical_ess_exception(self) -> None:
        """Test that exceptions from replay propagate to the caller."""
        with patch(
            "ax.early_stopping.experiment_replay.replay_experiment",
            side_effect=ValueError("Experiment's name is None."),
        ):
            with self.assertRaises(ValueError) as e:
                estimate_hypothetical_early_stopping_savings(
                    experiment=self.exp,
                    metric=self.metric,
                )

            self.assertIn("Experiment's name is None.", str(e.exception))
