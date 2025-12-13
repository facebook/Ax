# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.analysis.healthcheck.baseline_improvement import BaselineImprovementAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.evaluations_to_data import DataType, raw_evaluations_to_data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestBaselineImprovementAnalysis(TestCase):
    def setUp(self) -> None:
        """Setup common test fixtures."""
        super().setUp()
        self.experiment = get_branin_experiment(minimize=True)

        # Create two trials with arms
        self.trial_0 = self.experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0})]
            )
        )
        self.trial_1 = self.experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="1_0", parameters={"x1": 1.0, "x2": 1.0})]
            )
        )
        self.trial_0.mark_running(no_runner_required=True)
        self.trial_0.mark_completed()
        self.trial_1.mark_running(no_runner_required=True)
        self.trial_1.mark_completed()

    def _attach_improvement_data(
        self, baseline_val: float, comparison_val: float
    ) -> None:
        """Helper to attach data showing specific baseline vs comparison values."""
        data = Data.from_multiple_data(
            [
                raw_evaluations_to_data(
                    raw_data={"0_0": {"branin": (baseline_val, 0.0)}},
                    metric_name_to_signature={"branin": "branin"},
                    trial_index=0,
                    data_type=DataType.DATA,
                ),
                raw_evaluations_to_data(
                    raw_data={"1_0": {"branin": (comparison_val, 0.0)}},
                    metric_name_to_signature={"branin": "branin"},
                    trial_index=1,
                    data_type=DataType.DATA,
                ),
            ]
        )
        self.experiment.attach_data(data)

    def _create_multi_objective_experiment(self) -> Experiment:
        """Helper to create experiment with multi-objective config."""
        experiment = get_branin_experiment()
        metrics = [Metric(name="m1"), Metric(name="m2")]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=metrics[0], minimize=False),  # maximize
                    Objective(metric=metrics[1], minimize=True),  # minimize
                ]
            )
        )
        experiment._optimization_config = optimization_config
        return experiment

    def test_status_levels_with_subtests(self) -> None:
        """Test PASS/WARNING/FAIL status logic using subtests."""
        # Attach baseline data (higher is worse for branin)
        self._attach_improvement_data(baseline_val=100.0, comparison_val=100.0)

        test_cases = [
            ("all_improved", 50.0, HealthcheckStatus.PASS, "All 1 objective"),
            ("not_improved", 150.0, HealthcheckStatus.FAIL, "None of the 1 objective"),
        ]

        for name, comp_val, expected_status, expected_text in test_cases:
            with self.subTest(name=name):
                # Update comparison data
                self._attach_improvement_data(
                    baseline_val=100.0, comparison_val=comp_val
                )

                analysis = BaselineImprovementAnalysis(
                    comparison_arm_names=["1_0"],
                    baseline_arm_name="0_0",
                )
                card = analysis.compute(experiment=self.experiment)

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_text, card.subtitle)

    def test_multi_objective_with_subtests(self) -> None:
        """Test multi-objective scenarios using subtests."""
        experiment = self._create_multi_objective_experiment()

        # Create trials
        trial_0 = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0})]
            )
        )
        trial_1 = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="1_0", parameters={"x1": 1.0, "x2": 1.0})]
            )
        )
        trial_0.mark_running(no_runner_required=True)
        trial_0.mark_completed()
        trial_1.mark_running(no_runner_required=True)
        trial_1.mark_completed()

        test_cases = [
            (
                "all_improved",
                {
                    "m1": (20.0, 0.0),
                    "m2": (40.0, 0.0),
                },  # m1 increased (good), m2 decreased (good)
                HealthcheckStatus.PASS,
                "All 2 objective(s) improved",
                2,
            ),
            (
                "some_improved",
                {"m1": (20.0, 0.0), "m2": (100.0, 0.0)},  # m1 improved, m2 not
                HealthcheckStatus.WARNING,
                "1 out of 2 objective(s) improved",
                1,
            ),
            (
                "none_improved",
                {"m1": (5.0, 0.0), "m2": (100.0, 0.0)},  # Both worse
                HealthcheckStatus.FAIL,
                "None of the 2 objective(s) improved",
                0,
            ),
        ]

        # Baseline data: m1=10 (maximize), m2=50 (minimize)
        baseline_data = raw_evaluations_to_data(
            raw_data={"0_0": {"m1": (10.0, 0.0), "m2": (50.0, 0.0)}},
            metric_name_to_signature={"m1": "m1", "m2": "m2"},
            trial_index=0,
            data_type=DataType.DATA,
        )
        experiment.attach_data(baseline_data)

        for name, comp_metrics, status, text, num_improved in test_cases:
            with self.subTest(name=name):
                # Update comparison data
                comp_data = raw_evaluations_to_data(
                    raw_data={"1_0": comp_metrics},
                    metric_name_to_signature={"m1": "m1", "m2": "m2"},
                    trial_index=1,
                    data_type=DataType.DATA,
                )
                experiment.attach_data(comp_data)

                analysis = BaselineImprovementAnalysis(
                    comparison_arm_names=["1_0"],
                    baseline_arm_name="0_0",
                )
                card = analysis.compute(experiment=experiment)

                self.assertEqual(card.get_status(), status)
                self.assertIn(text, card.subtitle)
                self.assertEqual(
                    card.get_aditional_attrs()["num_objectives_improved"],
                    num_improved,
                )

    def test_footer_notes_parameter(self) -> None:
        """Test that footer_notes are appended to subtitle."""
        self._attach_improvement_data(baseline_val=100.0, comparison_val=50.0)

        custom_footer = "For more info, see: https://example.com/wiki"

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["1_0"],
            baseline_arm_name="0_0",
            footer_notes=custom_footer,
        )
        card = analysis.compute(experiment=self.experiment)

        self.assertIn(custom_footer, card.subtitle)
        self.assertTrue(card.subtitle.endswith(custom_footer))

    def test_footer_notes_none(self) -> None:
        """Test that None footer_notes doesn't add anything."""
        self._attach_improvement_data(baseline_val=100.0, comparison_val=50.0)

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["1_0"],
            baseline_arm_name="0_0",
            footer_notes=None,
        )
        card = analysis.compute(experiment=self.experiment)

        # Should not end with extra newlines or footer content
        self.assertNotIn("https://", card.subtitle)
        self.assertFalse(card.subtitle.endswith("\n\n"))

    def test_validate_applicable_state_no_experiment(self) -> None:
        """Test validation fails when experiment is None."""
        analysis = BaselineImprovementAnalysis()

        error_msg = analysis.validate_applicable_state(experiment=None)

        self.assertIsNotNone(error_msg)
        self.assertIn("experiment", error_msg.lower())

    def test_validate_applicable_state_no_optimization_config(self) -> None:
        """Test validation fails when no optimization config."""
        experiment = get_branin_experiment()
        experiment._optimization_config = None

        analysis = BaselineImprovementAnalysis()
        error_msg = analysis.validate_applicable_state(experiment=experiment)

        self.assertIsNotNone(error_msg)
        self.assertIn("optimization_config", error_msg)

    def test_validate_applicable_state_no_data(self) -> None:
        """Test validation fails when no data available."""
        experiment = get_branin_experiment()
        trial = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0})]
            )
        )
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

        analysis = BaselineImprovementAnalysis(comparison_arm_names=["0_0"])
        error_msg = analysis.validate_applicable_state(experiment=experiment)

        self.assertIsNotNone(error_msg)
        self.assertIn("baseline comparison values", error_msg)

    def test_validate_applicable_state_success(self) -> None:
        """Test validation passes with valid experiment."""
        self._attach_improvement_data(baseline_val=100.0, comparison_val=50.0)

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["1_0"],
            baseline_arm_name="0_0",
        )
        error_msg = analysis.validate_applicable_state(experiment=self.experiment)

        self.assertIsNone(error_msg)

    def test_compute_raises_when_validation_fails(self) -> None:
        """Test compute raises UserInputError when validation would fail."""
        experiment = get_branin_experiment()
        experiment._optimization_config = None

        analysis = BaselineImprovementAnalysis()

        with self.assertRaisesRegex(UserInputError, "Could not select baseline arm"):
            analysis.compute(experiment=experiment)

    def test_baseline_selection_modes(self) -> None:
        """Test baseline selection with explicit vs auto-selected baseline."""
        self._attach_improvement_data(baseline_val=100.0, comparison_val=50.0)

        test_cases = [
            ("auto_first_trial", None, "first trial's first arm"),
            ("explicit_baseline", "0_0", "Baseline arm:"),
        ]

        for name, baseline_name, expected_text in test_cases:
            with self.subTest(name=name):
                analysis = BaselineImprovementAnalysis(
                    comparison_arm_names=["1_0"],
                    baseline_arm_name=baseline_name,
                )
                card = analysis.compute(experiment=self.experiment)

                self.assertIn(expected_text, card.subtitle)

    def test_dataframe_structure(self) -> None:
        """Test that the returned DataFrame has correct structure."""
        experiment = self._create_multi_objective_experiment()

        trial_0 = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0})]
            )
        )
        trial_1 = experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[Arm(name="1_0", parameters={"x1": 1.0, "x2": 1.0})]
            )
        )
        trial_0.mark_running(no_runner_required=True)
        trial_0.mark_completed()
        trial_1.mark_running(no_runner_required=True)
        trial_1.mark_completed()

        # One objective improved, one not
        data = Data.from_multiple_data(
            [
                raw_evaluations_to_data(
                    raw_data={"0_0": {"m1": (10.0, 0.0), "m2": (50.0, 0.0)}},
                    metric_name_to_signature={"m1": "m1", "m2": "m2"},
                    trial_index=0,
                    data_type=DataType.DATA,
                ),
                raw_evaluations_to_data(
                    raw_data={"1_0": {"m1": (20.0, 0.0), "m2": (100.0, 0.0)}},
                    metric_name_to_signature={"m1": "m1", "m2": "m2"},
                    trial_index=1,
                    data_type=DataType.DATA,
                ),
            ]
        )
        experiment.attach_data(data)

        analysis = BaselineImprovementAnalysis(
            comparison_arm_names=["1_0"],
            baseline_arm_name="0_0",
        )
        card = analysis.compute(experiment=experiment)

        df = card.df
        self.assertEqual(len(df), 2)
        self.assertIn("Metric", df.columns)
        self.assertIn("Status", df.columns)
        self.assertIn("Details", df.columns)

        # Check status symbols
        statuses = df["Status"].tolist()
        self.assertIn("✓ Improved", statuses)
        self.assertIn("✗ Not Improved", statuses)
