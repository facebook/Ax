# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

from ax.adapter.random import RandomAdapter
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.predictable_metrics import (
    HEALTHCHECK_DESCRIPTION,
    PredictableMetricsAnalysis,
)
from ax.generators.random.base import RandomGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestPredictableMetricsAnalysis(TestCase):
    def test_passes_when_all_metrics_predictable(self) -> None:
        # GIVEN an experiment and generation strategy with good R² scores
        experiment = get_branin_experiment(with_completed_trial=True)
        mock_gs = MagicMock()
        mock_adapter = MagicMock()
        mock_gs.adapter = mock_adapter

        with patch(
            "ax.analysis.healthcheck.predictable_metrics."
            "compute_model_fit_metrics_from_adapter"
        ) as mock_compute:
            mock_compute.return_value = {
                "coefficient_of_determination": {"branin": 0.85}
            }

            # WHEN healthcheck is run
            healthcheck = PredictableMetricsAnalysis(model_fit_threshold=0.0)
            card = healthcheck.compute(
                experiment=experiment,
                generation_strategy=mock_gs,
            )

            # THEN it returns PASS status
            self.assertEqual(card.name, "PredictableMetricsAnalysis")
            self.assertEqual(card.title, "Predictable Metrics Success")
            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertIn(HEALTHCHECK_DESCRIPTION, card.subtitle)
            self.assertIn("All metrics are predictable", card.subtitle)
            self.assertEqual(len(card.df), 0)

    def test_warns_when_metrics_unpredictable(self) -> None:
        # GIVEN an experiment and generation strategy with poor R² scores
        experiment = get_branin_experiment(with_completed_trial=True)
        mock_gs = MagicMock()
        mock_adapter = MagicMock()
        mock_gs.adapter = mock_adapter

        with patch(
            "ax.analysis.healthcheck.predictable_metrics."
            "compute_model_fit_metrics_from_adapter"
        ) as mock_compute:
            mock_compute.return_value = {
                "coefficient_of_determination": {"branin": -0.15}
            }

            # WHEN healthcheck is run
            healthcheck = PredictableMetricsAnalysis(model_fit_threshold=0.0)
            card = healthcheck.compute(
                experiment=experiment,
                generation_strategy=mock_gs,
            )

            # THEN it returns WARNING status with R² in dataframe
            self.assertEqual(card.name, "PredictableMetricsAnalysis")
            self.assertEqual(card.title, "Predictable Metrics Warning")
            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn(HEALTHCHECK_DESCRIPTION, card.subtitle)
            self.assertIn("behaving unpredictably", card.subtitle)
            self.assertIn("branin", card.subtitle)
            self.assertIn("-0.1500", card.subtitle)
            # Verify dataframe contains R² values
            self.assertEqual(len(card.df), 1)
            self.assertIn("branin", card.df["Metric"].tolist())
            self.assertEqual(card.df["R²"].iloc[0], -0.15)

    def test_threshold_and_guidance_parameters(self) -> None:
        # GIVEN different configurations
        experiment = get_branin_experiment(with_completed_trial=True)
        mock_gs = MagicMock()
        mock_adapter = MagicMock()
        mock_gs.adapter = mock_adapter

        # Test with custom threshold
        with self.subTest("custom_threshold"):
            with patch(
                "ax.analysis.healthcheck.predictable_metrics"
                ".compute_model_fit_metrics_from_adapter"
            ) as mock_compute:
                mock_compute.return_value = {
                    "coefficient_of_determination": {"branin": 0.3}
                }

                healthcheck = PredictableMetricsAnalysis(
                    model_fit_threshold=0.5, guidance_message=None
                )
                card = healthcheck.compute(
                    experiment=experiment,
                    generation_strategy=mock_gs,
                )

                self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
                self.assertIn("R² < 0.50", card.subtitle)

        # Test with guidance message
        with self.subTest("with_guidance"):
            with patch(
                "ax.analysis.healthcheck.predictable_metrics"
                ".compute_model_fit_metrics_from_adapter"
            ) as mock_compute:
                mock_compute.return_value = {
                    "coefficient_of_determination": {"branin": -0.1}
                }

                healthcheck = PredictableMetricsAnalysis(
                    model_fit_threshold=0.0,
                    guidance_message=" Contact support@example.com",
                )
                card = healthcheck.compute(
                    experiment=experiment,
                    generation_strategy=mock_gs,
                )

                self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
                self.assertIn("Contact support@example.com", card.subtitle)

        # Test passing threshold
        with self.subTest("passes_threshold"):
            with patch(
                "ax.analysis.healthcheck.predictable_metrics"
                ".compute_model_fit_metrics_from_adapter"
            ) as mock_compute:
                mock_compute.return_value = {
                    "coefficient_of_determination": {"branin": 0.6}
                }

                healthcheck = PredictableMetricsAnalysis(
                    model_fit_threshold=0.5, guidance_message=None
                )
                card = healthcheck.compute(
                    experiment=experiment,
                    generation_strategy=mock_gs,
                )

                self.assertEqual(card.get_status(), HealthcheckStatus.PASS)

    def test_random_adapter_warning(self) -> None:
        # GIVEN an experiment with RandomAdapter
        experiment = get_branin_experiment(with_completed_trial=True)
        mock_gs = MagicMock()
        mock_gs.adapter = RandomAdapter(
            experiment=experiment,
            generator=RandomGenerator(),
        )

        # WHEN healthcheck is run
        healthcheck = PredictableMetricsAnalysis()
        card = healthcheck.compute(
            experiment=experiment,
            generation_strategy=mock_gs,
        )

        # THEN it returns WARNING about RandomAdapter
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("RandomAdapter", card.subtitle)
        self.assertIn("no model is being used", card.subtitle)

    def test_validate_applicable_state(self) -> None:
        # GIVEN a healthcheck instance
        healthcheck = PredictableMetricsAnalysis()
        experiment = get_branin_experiment()
        mock_gs = MagicMock()

        with self.subTest("no_experiment"):
            # WHEN experiment is None
            # THEN validation fails
            error = healthcheck.validate_applicable_state(
                experiment=None, generation_strategy=mock_gs
            )
            self.assertIsNotNone(error)
            self.assertIn("Experiment", error)

        with self.subTest("no_generation_strategy"):
            # WHEN generation_strategy is None
            # THEN validation fails
            error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=None
            )
            self.assertIsNotNone(error)
            self.assertIn("GenerationStrategy", error)

        with self.subTest("valid_inputs"):
            # WHEN both experiment and generation_strategy are provided
            # THEN validation passes
            error = healthcheck.validate_applicable_state(
                experiment=experiment, generation_strategy=mock_gs
            )
            self.assertIsNone(error)
