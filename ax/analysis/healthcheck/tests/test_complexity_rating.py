# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.adapter_utils import can_map_to_binary, is_unordered_choice
from ax.analysis.healthcheck.complexity_rating import ComplexityRatingAnalysis
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestComplexityRatingAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()

    def test_wheelhouse_configuration(self) -> None:
        summary = summarize_experiment(self.experiment, max_trials=100)
        healthcheck = ComplexityRatingAnalysis()
        card = healthcheck.compute()

        self.assertEqual(card.name, "ComplexityRatingAnalysis")
        self.assertEqual(card.title, "Support Tier Healthcheck")
        self.assertTrue(card.is_passing())
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertIn("Wheelhouse", card.subtitle)
        self.assertIn("should not run into any problems", card.subtitle)
        self.assertEqual(card.get_aditional_attrs()["tier"], "Wheelhouse")

    def test_parameter_counts(self) -> None:
        test_cases = [
            (60, HealthcheckStatus.WARNING, "Advanced", "60 tunable parameter(s)"),
            (250, HealthcheckStatus.FAIL, "Unsupported", "250 tunable parameter(s)"),
        ]

        for num_params, expected_status, expected_tier, expected_msg in test_cases:
            with self.subTest(num_params=num_params):
                params = [
                    RangeParameter(
                        name=f"x{i}",
                        parameter_type=ParameterType.FLOAT,
                        lower=0.0,
                        upper=1.0,
                    )
                    for i in range(num_params)
                ]
                experiment = get_branin_experiment()
                experiment._search_space = SearchSpace(parameters=params)
                summary = summarize_experiment(experiment, max_trials=100)
                card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)
                self.assertEqual(card.get_aditional_attrs()["tier"], expected_tier)

    def test_objectives_count(self) -> None:
        test_cases = [
            (3, HealthcheckStatus.WARNING, "Advanced", "3 objectives"),
            (5, HealthcheckStatus.FAIL, "Unsupported", "5 objectives"),
        ]

        for num_objs, expected_status, expected_tier, expected_msg in test_cases:
            with self.subTest(num_objectives=num_objs):
                experiment = get_branin_experiment()
                metrics = [Metric(name=f"m{i}") for i in range(num_objs)]
                experiment._optimization_config = MultiObjectiveOptimizationConfig(
                    objective=MultiObjective(
                        objectives=[
                            Objective(metric=m, minimize=False) for m in metrics
                        ]
                    )
                )
                summary = summarize_experiment(experiment, max_trials=100)
                card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_constraints(self) -> None:
        with self.subTest(constraint_type="outcome"):
            experiment = get_branin_experiment()
            metrics = [Metric(name=f"m{i}") for i in range(3)]
            experiment._optimization_config = OptimizationConfig(
                objective=Objective(metric=Metric(name="obj"), minimize=False),
                outcome_constraints=[
                    OutcomeConstraint(metric=m, op=ComparisonOp.LEQ, bound=1.0)
                    for m in metrics
                ],
            )
            summary = summarize_experiment(experiment, max_trials=100)
            card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("Advanced", card.subtitle)
            self.assertIn("3 outcome constraints", card.subtitle)

        with self.subTest(constraint_type="parameter"):
            params = [
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                )
                for i in range(5)
            ]
            parameter_constraints = [
                ParameterConstraint(constraint_dict={f"x{i}": 1.0}, bound=1.0)
                for i in range(3)
            ]
            experiment = get_branin_experiment()
            experiment._search_space = SearchSpace(
                parameters=params, parameter_constraints=parameter_constraints
            )
            summary = summarize_experiment(experiment, max_trials=100)
            card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("Advanced", card.subtitle)
            self.assertIn("3 parameter constraints", card.subtitle)

    def test_stopping_strategies(self) -> None:
        test_cases = [
            ("early_stopping", {"uses_early_stopping": True}, "Early stopping"),
            ("global_stopping", {"uses_global_stopping": True}, "Global stopping"),
        ]

        for name, kwargs, expected_msg in test_cases:
            with self.subTest(strategy=name):
                summary = summarize_experiment(
                    self.experiment, max_trials=100, **kwargs
                )
                card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
                self.assertIn("Advanced", card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_trial_counts(self) -> None:
        test_cases = [
            (300, HealthcheckStatus.WARNING, "Advanced", "300 total trials"),
            (600, HealthcheckStatus.FAIL, "Unsupported", "600 total trials"),
        ]

        for max_trials, expected_status, expected_tier, expected_msg in test_cases:
            with self.subTest(max_trials=max_trials):
                summary = summarize_experiment(self.experiment, max_trials=max_trials)
                card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_unsupported_configurations(self) -> None:
        test_cases = [
            (
                "not_using_configs",
                {"all_inputs_are_configs": False},
                "low-level Ax abstractions",
            ),
            (
                "high_failure_rate",
                {"tolerated_trial_failure_rate": 0.95},
                "0.95",
            ),
            (
                "invalid_failure_rate_check",
                {
                    "max_pending_trials": 10,
                    "min_failed_trials_for_failure_rate_check": 50,
                },
                "min_failed_trials_for_failure_rate_check",
            ),
            (
                "non_default_advanced_options",
                {"non_default_advanced_options": True},
                "Non-default advanced_options",
            ),
            (
                "merge_multiple_curves",
                {"uses_merge_multiple_curves": True},
                "merge_multiple_curves",
            ),
        ]

        for name, kwargs, expected_msg in test_cases:
            with self.subTest(config=name):
                # pyre-ignore[6]: Pyre has trouble inferring types in **kwargs unpacking
                summary = summarize_experiment(self.experiment, 100, **kwargs)
                card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
                self.assertIn("Unsupported", card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_unordered_choice_parameters(self) -> None:
        params = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            ),
            ChoiceParameter(
                name="cat1",
                parameter_type=ParameterType.STRING,
                values=["a", "b", "c", "d"],
                is_ordered=False,
            ),
        ]
        experiment = get_branin_experiment()
        experiment._search_space = SearchSpace(parameters=params)

        self.assertTrue(is_unordered_choice(params[1], min_choices=3, max_choices=5))

        summary = summarize_experiment(experiment, max_trials=100)
        card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("unordered choice parameter(s)", card.subtitle)

    def test_binary_parameters_count(self) -> None:
        params = [
            ChoiceParameter(
                name=f"binary{i}",
                parameter_type=ParameterType.BOOL,
                values=[True, False],
            )
            for i in range(60)
        ]
        experiment = get_branin_experiment()
        experiment._search_space = SearchSpace(parameters=params)

        for p in params:
            self.assertTrue(can_map_to_binary(p))

        summary = summarize_experiment(experiment, max_trials=100)
        card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 binary tunable parameter(s)", card.subtitle)

    def test_multiple_violations(self) -> None:
        params = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(60)
        ]
        experiment = get_branin_experiment()
        experiment._search_space = SearchSpace(parameters=params)
        summary = summarize_experiment(
            experiment, max_trials=300, uses_early_stopping=True
        )
        card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 tunable parameter(s)", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)

    def test_dataframe_summary(self) -> None:
        summary = summarize_experiment(self.experiment, max_trials=100)
        card = ComplexityRatingAnalysis(experiment_summary=summary).compute()

        df = card.df
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)

        metrics = df["Metric"].tolist()
        self.assertIn("Support Tier", metrics)
        self.assertIn("Total Parameters", metrics)
        self.assertIn("Objectives", metrics)
        self.assertIn("Outcome Constraints", metrics)
        self.assertIn("Parameter Constraints", metrics)

        tier_row = df[df["Metric"] == "Support Tier"]
        self.assertEqual(tier_row.iloc[0]["Value"], "Wheelhouse")

    def test_pre_computed_experiment_summary(self) -> None:
        custom_summary: OptimizationSummary = {
            "max_trials": 300,
            "num_params": 10,
            "num_binary": 5,
            "num_categorical_3_5": 1,
            "num_categorical_6_inf": 0,
            "num_parameter_constraints": 1,
            "num_objectives": 1,
            "num_outcome_constraints": 0,
            "uses_early_stopping": True,
            "uses_global_stopping": False,
            "tolerated_trial_failure_rate": None,
            "max_pending_trials": None,
            "min_failed_trials_for_failure_rate_check": None,
            "all_inputs_are_configs": True,
            "uses_merge_multiple_curves": False,
            "non_default_advanced_options": False,
        }

        card = ComplexityRatingAnalysis(experiment_summary=custom_summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)
