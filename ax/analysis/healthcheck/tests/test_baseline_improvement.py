# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import pandas as pd
from ax.analysis.healthcheck.baseline_improvement import BaselineImprovementAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.data import Data
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


class TestBaselineImprovementAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Single-objective experiment with a batch trial (has multiple arms: 0_0, 0_1)
        # minimize=True for consistency with get_branin_experiment_with_multi_objective
        self.experiment = get_branin_experiment(with_batch=True, minimize=True)
        self.experiment.trials[0].mark_running(no_runner_required=True)
        self.experiment.trials[0].mark_completed()

        # Multi-objective experiment with status_quo
        # minimize=True by default
        self.moo_experiment = get_branin_experiment_with_multi_objective(
            with_batch=True, with_status_quo=True
        )
        self.moo_experiment.trials[0].mark_running(no_runner_required=True)
        self.moo_experiment.trials[0].mark_completed()

    def _attach_data(
        self,
        metric_values: dict[str, list[tuple[float, float]]],
        arm_names: list[str],
        experiment: None = None,
    ) -> None:
        """Attach metric data to the experiment."""
        exp = experiment if experiment is not None else self.experiment

        rows = []
        for metric_name, values in metric_values.items():
            for i, (mean, sem) in enumerate(values):
                rows.append(
                    {
                        "arm_name": arm_names[i],
                        "metric_name": metric_name,
                        "mean": mean,
                        "sem": sem,
                        "trial_index": 0,
                        "n": 100,
                        "metric_signature": metric_name,
                    }
                )
        exp.attach_data(Data(df=pd.DataFrame(rows)))

    def test_status_outcomes(self) -> None:
        """Test PASS/FAIL status based on improvement."""
        # minimize=True: lower is better
        test_cases = [
            # (baseline_mean, comparison_mean, expected_status, description)
            (100.0, 50.0, HealthcheckStatus.PASS, "improved (lower)"),
            (50.0, 100.0, HealthcheckStatus.FAIL, "not improved (higher)"),
        ]

        for baseline_mean, comparison_mean, expected_status, desc in test_cases:
            with self.subTest(desc=desc):
                self._attach_data(
                    {"branin": [(baseline_mean, 0.1), (comparison_mean, 0.1)]},
                    arm_names=["0_0", "0_1"],
                )
                analysis = BaselineImprovementAnalysis(
                    comparison_arm_names=["0_1"],
                    baseline_arm_name="0_0",
                )
                card = analysis.compute(experiment=self.experiment)
                self.assertEqual(card.get_status(), expected_status)

    def test_multi_objective_partial_improvement(self) -> None:
        """Test WARNING status when only some objectives improve."""
        # minimize=True for both objectives (lower is better):
        # branin_a: 100 -> 50, improved (decreased)
        # branin_b: 50 -> 100, NOT improved (increased)
        self._attach_data(
            {
                "branin_a": [(100.0, 0.1), (50.0, 0.1)],
                "branin_b": [(50.0, 0.1), (100.0, 0.1)],
            },
            arm_names=["status_quo", "0_0"],
            experiment=self.moo_experiment,
        )

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["0_0"],
            baseline_arm_name="status_quo",
        )
        card = analysis.compute(experiment=self.moo_experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("1 out of 2", card.subtitle)

    def test_documentation_link(self) -> None:
        """Test documentation_link is appended correctly."""
        # minimize=True: baseline=50, comparison=100 -> NOT improved (FAIL status)
        self._attach_data(
            {"branin": [(50.0, 0.1), (100.0, 0.1)]}, arm_names=["0_0", "0_1"]
        )
        doc_link = "https://example.com/wiki"

        test_cases = [
            # (baseline_arm_name, expected_substring, description)
            (
                "0_0",
                "For more information on performance measurement",
                "explicit baseline",
            ),
            (None, "no explicit baseline was provided", "auto-selected baseline"),
        ]

        for baseline_arm_name, expected_substring, desc in test_cases:
            with self.subTest(desc=desc):
                analysis = BaselineImprovementAnalysis(
                    comparison_arm_names=["0_1"],
                    baseline_arm_name=baseline_arm_name,
                    documentation_link=doc_link,
                )
                card = analysis.compute(experiment=self.experiment)
                self.assertIn(expected_substring, card.subtitle)
                self.assertIn(doc_link, card.subtitle)

    def test_validation_errors(self) -> None:
        """Test validation fails for invalid states."""
        analysis = BaselineImprovementAnalysis()

        with self.subTest(desc="no experiment"):
            error_msg = analysis.validate_applicable_state(experiment=None)
            self.assertIsNotNone(error_msg)
            self.assertIn("experiment", error_msg.lower())

        with self.subTest(desc="no optimization config"):
            self.experiment._optimization_config = None
            error_msg = analysis.validate_applicable_state(experiment=self.experiment)
            self.assertIsNotNone(error_msg)
            self.assertIn("optimizationconfig", error_msg.lower())

    def test_compute_raises_without_trials(self) -> None:
        """Test compute raises UserInputError when no trials exist."""
        # Remove trials from self.experiment
        self.experiment._trials = {}
        analysis = BaselineImprovementAnalysis()

        with self.assertRaisesRegex(
            UserInputError, "Could not find valid baseline arm"
        ):
            analysis.compute(experiment=self.experiment)

    def test_dataframe_structure(self) -> None:
        """Test returned DataFrame has expected columns and content."""
        self._attach_data(
            {"branin": [(50.0, 0.1), (100.0, 0.1)]}, arm_names=["0_0", "0_1"]
        )

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["0_1"],
            baseline_arm_name="0_0",
        )
        card = analysis.compute(experiment=self.experiment)

        self.assertIn("Metric", card.df.columns)
        self.assertIn("Status", card.df.columns)
        self.assertIn("Details", card.df.columns)
        self.assertEqual(len(card.df), 1)

    def test_no_improvement_message_parameter(self) -> None:
        """Test custom no_improvement_message is displayed on FAIL status."""
        self._attach_data(
            {"branin": [(50.0, 0.1), (100.0, 0.1)]}, arm_names=["0_0", "0_1"]
        )

        custom_message = "Custom failure message - contact support."

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["0_1"],
            baseline_arm_name="0_0",
            no_improvement_message=custom_message,
        )
        card = analysis.compute(experiment=self.experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn(custom_message, card.subtitle)
