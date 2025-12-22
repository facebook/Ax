# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.adapter.random import RandomAdapter
from ax.adapter.registry import Generators
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.predictable_metrics import (
    DEFAULT_MODEL_FIT_THRESHOLD,
    HEALTHCHECK_DESCRIPTION,
    HEALTHCHECK_TITLE,
    PredictableMetricsAnalysis,
)
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generators.random.base import RandomGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize


class TestPredictableMetricsAnalysis(TestCase):
    """Tests for PredictableMetricsAnalysis healthcheck."""

    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment(with_completed_trial=True)
        self.generation_strategy = self._create_generation_strategy()

    def _create_generation_strategy(self, fit: bool = True) -> GenerationStrategy:
        """Create a generation strategy for testing.

        Args:
            fit: Whether to fit the generation strategy. Defaults to True.

        Returns:
            A GenerationStrategy, fitted if fit=True.
        """
        generation_strategy = GenerationStrategy(
            name="test_gs",
            nodes=[
                GenerationNode(
                    name="test_node",
                    generator_specs=[
                        GeneratorSpec(
                            generator_enum=Generators.BOTORCH_MODULAR,
                        )
                    ],
                )
            ],
        )
        generation_strategy.experiment = self.experiment
        if fit:
            generation_strategy._curr._fit(experiment=self.experiment)
        return generation_strategy

    def _create_gs_with_random_adapter(self) -> GenerationStrategy:
        """Create a generation strategy with RandomAdapter for testing."""
        generation_strategy = GenerationStrategy(
            name="random_gs",
            nodes=[
                GenerationNode(
                    name="random_node",
                    generator_specs=[
                        GeneratorSpec(
                            generator_enum=Generators.SOBOL,
                        )
                    ],
                )
            ],
        )
        generation_strategy.experiment = self.experiment
        # Manually set a RandomAdapter on the GeneratorSpec
        # to test the RandomAdapter check
        generation_strategy._curr.generator_specs[0]._fitted_adapter = RandomAdapter(
            experiment=self.experiment,
            generator=RandomGenerator(),
        )
        return generation_strategy

    def test_passes_when_all_metrics_predictable(self) -> None:
        """Test that healthcheck returns PASS when all metrics are predictable."""
        with patch(
            "ax.analysis.healthcheck.predictable_metrics.warn_if_unpredictable_metrics"
        ) as mock_warn:
            # Mock returns None when all metrics are predictable
            mock_warn.return_value = None

            healthcheck = PredictableMetricsAnalysis(
                model_fit_threshold=DEFAULT_MODEL_FIT_THRESHOLD
            )
            card = healthcheck.compute(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )

            self.assertEqual(card.name, "PredictableMetricsAnalysis")
            self.assertEqual(card.title, f"{HEALTHCHECK_TITLE} Success")
            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertIn(HEALTHCHECK_DESCRIPTION, card.subtitle)
            self.assertIn("All metrics are predictable", card.subtitle)

    def test_warns_when_metrics_unpredictable(self) -> None:
        """Test that healthcheck returns WARNING when metrics are unpredictable."""
        warning_msg = "The metric 'branin' is behaving unpredictably (R² = -0.15)."

        with patch(
            "ax.analysis.healthcheck.predictable_metrics.warn_if_unpredictable_metrics"
        ) as mock_warn:
            mock_warn.return_value = warning_msg

            healthcheck = PredictableMetricsAnalysis(
                model_fit_threshold=DEFAULT_MODEL_FIT_THRESHOLD
            )
            card = healthcheck.compute(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )

            self.assertEqual(card.name, "PredictableMetricsAnalysis")
            self.assertEqual(card.title, f"{HEALTHCHECK_TITLE} Warning")
            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn(HEALTHCHECK_DESCRIPTION, card.subtitle)
            self.assertIn("behaving unpredictably", card.subtitle)
            self.assertIn("branin", card.subtitle)

    def test_custom_threshold_passed_to_underlying_function(self) -> None:
        """Test that custom threshold is passed to warn_if_unpredictable_metrics."""
        custom_threshold = 0.5

        with patch(
            "ax.analysis.healthcheck.predictable_metrics.warn_if_unpredictable_metrics"
        ) as mock_warn:
            mock_warn.return_value = None

            healthcheck = PredictableMetricsAnalysis(
                model_fit_threshold=custom_threshold
            )
            healthcheck.compute(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )

            # Verify threshold was passed correctly
            mock_warn.assert_called_once_with(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
                model_fit_threshold=custom_threshold,
            )

    def test_guidance_message_appended_on_warning(self) -> None:
        """Test that guidance message is appended to subtitle on warning."""
        guidance = "Contact support@example.com for assistance."
        warning_msg = "The metric 'branin' is behaving unpredictably."

        with patch(
            "ax.analysis.healthcheck.predictable_metrics.warn_if_unpredictable_metrics"
        ) as mock_warn:
            mock_warn.return_value = warning_msg

            healthcheck = PredictableMetricsAnalysis(
                model_fit_threshold=DEFAULT_MODEL_FIT_THRESHOLD,
                guidance_message=guidance,
            )
            card = healthcheck.compute(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn(guidance, card.subtitle)

    def test_guidance_message_not_shown_on_pass(self) -> None:
        """Test that guidance message is not shown when status is PASS."""
        guidance = "Contact support@example.com for assistance."

        with patch(
            "ax.analysis.healthcheck.predictable_metrics.warn_if_unpredictable_metrics"
        ) as mock_warn:
            mock_warn.return_value = None

            healthcheck = PredictableMetricsAnalysis(
                model_fit_threshold=DEFAULT_MODEL_FIT_THRESHOLD,
                guidance_message=guidance,
            )
            card = healthcheck.compute(
                experiment=self.experiment,
                generation_strategy=self.generation_strategy,
            )

            self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
            self.assertNotIn(guidance, card.subtitle)

    def test_random_adapter_not_applicable(self) -> None:
        """Test that validation fails for RandomAdapter."""
        gs = self._create_gs_with_random_adapter()

        healthcheck = PredictableMetricsAnalysis()
        error = healthcheck.validate_applicable_state(
            experiment=self.experiment,
            generation_strategy=gs,
        )

        self.assertIsNotNone(error)
        self.assertIn("RandomAdapter", error)
        self.assertIn("no model to evaluate", error)

    def test_validate_applicable_state_no_experiment(self) -> None:
        """Test that validation fails when experiment is None."""
        healthcheck = PredictableMetricsAnalysis()

        error = healthcheck.validate_applicable_state(
            experiment=None,
            generation_strategy=self.generation_strategy,
        )

        self.assertIsNotNone(error)
        self.assertIn("Experiment", error)

    def test_validate_applicable_state_no_generation_strategy(self) -> None:
        """Test that validation fails when generation_strategy is None."""
        healthcheck = PredictableMetricsAnalysis()

        error = healthcheck.validate_applicable_state(
            experiment=self.experiment,
            generation_strategy=None,
        )

        self.assertIsNotNone(error)
        self.assertIn("GenerationStrategy", error)

    def test_validate_applicable_state_valid_inputs(self) -> None:
        """Test that validation passes with valid experiment and generation strategy."""
        healthcheck = PredictableMetricsAnalysis()

        error = healthcheck.validate_applicable_state(
            experiment=self.experiment,
            generation_strategy=self.generation_strategy,
        )

        self.assertIsNone(error)

    @mock_botorch_optimize
    def test_adapter_resolved_from_generation_strategy(self) -> None:
        """Test that adapter is resolved by fitting when initially None."""
        generation_strategy = self._create_generation_strategy(fit=False)
        # Adapter should be None before fitting
        self.assertIsNone(generation_strategy.adapter)

        healthcheck = PredictableMetricsAnalysis()

        error = healthcheck.validate_applicable_state(
            experiment=self.experiment,
            generation_strategy=generation_strategy,
        )

        # After validate_applicable_state, the adapter should be resolved
        self.assertIsNotNone(generation_strategy.adapter)
        self.assertIsNone(error)
