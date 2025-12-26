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
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_improvement_global_stopping_strategy,
    get_percentile_early_stopping_strategy,
)


class TestComplexityRatingAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()
        self.tier_metadata: dict[str, object] = {
            "user_supplied_max_trials": 100,
            "uses_standard_api": True,
        }

    def test_validate_applicable_state_requires_experiment(self) -> None:
        healthcheck = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata)
        result = healthcheck.validate_applicable_state(experiment=None)
        self.assertIsNotNone(result)
        self.assertIn("Experiment is required", result)

    def test_validate_applicable_state_passes_with_valid_inputs(self) -> None:
        healthcheck = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata)
        result = healthcheck.validate_applicable_state(experiment=self.experiment)
        self.assertIsNone(result)

    def test_standard_configuration(self) -> None:
        healthcheck = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata)
        card = healthcheck.compute(experiment=self.experiment)

        self.assertEqual(card.name, "ComplexityRatingAnalysis")
        self.assertEqual(card.title, "Complexity Rating Healthcheck")
        self.assertTrue(card.is_passing())
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertIn("Standard", card.subtitle)
        self.assertEqual(card.get_aditional_attrs()["tier"], "Standard")

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
                self.experiment._search_space = SearchSpace(parameters=params)
                card = ComplexityRatingAnalysis(
                    tier_metadata=self.tier_metadata
                ).compute(experiment=self.experiment)

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
                metrics = [Metric(name=f"m{i}") for i in range(num_objs)]
                self.experiment._optimization_config = MultiObjectiveOptimizationConfig(
                    objective=MultiObjective(
                        objectives=[
                            Objective(metric=m, minimize=False) for m in metrics
                        ]
                    )
                )
                card = ComplexityRatingAnalysis(
                    tier_metadata=self.tier_metadata
                ).compute(experiment=self.experiment)

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_constraints(self) -> None:
        with self.subTest(constraint_type="outcome"):
            metrics = [Metric(name=f"m{i}") for i in range(3)]
            self.experiment._optimization_config = OptimizationConfig(
                objective=Objective(metric=Metric(name="obj"), minimize=False),
                outcome_constraints=[
                    OutcomeConstraint(metric=m, op=ComparisonOp.LEQ, bound=1.0)
                    for m in metrics
                ],
            )
            card = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata).compute(
                experiment=self.experiment
            )

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
                ParameterConstraint(inequality=f"x{i} <= 1.0") for i in range(3)
            ]
            self.experiment._search_space = SearchSpace(
                parameters=params, parameter_constraints=parameter_constraints
            )
            card = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata).compute(
                experiment=self.experiment
            )

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("Advanced", card.subtitle)
            self.assertIn("3 parameter constraints", card.subtitle)

    def test_stopping_strategies(self) -> None:
        with self.subTest(strategy="early_stopping"):
            card = ComplexityRatingAnalysis(
                tier_metadata=self.tier_metadata,
                early_stopping_strategy=get_percentile_early_stopping_strategy(),
            ).compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("Advanced", card.subtitle)
            self.assertIn("Early stopping", card.subtitle)

        with self.subTest(strategy="global_stopping"):
            card = ComplexityRatingAnalysis(
                tier_metadata=self.tier_metadata,
                global_stopping_strategy=get_improvement_global_stopping_strategy(),
            ).compute(experiment=self.experiment)

            self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
            self.assertIn("Advanced", card.subtitle)
            self.assertIn("Global stopping", card.subtitle)

    def test_trial_counts(self) -> None:
        test_cases = [
            (300, HealthcheckStatus.WARNING, "Advanced", "300 total trials"),
            (600, HealthcheckStatus.FAIL, "Unsupported", "600 total trials"),
        ]

        for max_trials, expected_status, expected_tier, expected_msg in test_cases:
            with self.subTest(max_trials=max_trials):
                tier_metadata = {
                    "user_supplied_max_trials": max_trials,
                    "uses_standard_api": True,
                }
                card = ComplexityRatingAnalysis(tier_metadata=tier_metadata).compute(
                    experiment=self.experiment
                )

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_unsupported_configurations(self) -> None:
        with self.subTest(config="not_using_standard_api"):
            tier_metadata = {
                "user_supplied_max_trials": 100,
                "uses_standard_api": False,
            }
            card = ComplexityRatingAnalysis(tier_metadata=tier_metadata).compute(
                experiment=self.experiment
            )
            self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
            self.assertIn("Unsupported", card.subtitle)
            self.assertIn("uses_standard_api=False", card.subtitle)

        with self.subTest(config="high_failure_rate"):
            tier_metadata = {"user_supplied_max_trials": 100, "uses_standard_api": True}
            card = ComplexityRatingAnalysis(
                tier_metadata=tier_metadata,
                tolerated_trial_failure_rate=0.95,
            ).compute(experiment=self.experiment)
            self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
            self.assertIn("Unsupported", card.subtitle)
            self.assertIn("0.95", card.subtitle)

        with self.subTest(config="invalid_failure_rate_check"):
            tier_metadata = {"user_supplied_max_trials": 100, "uses_standard_api": True}
            card = ComplexityRatingAnalysis(
                tier_metadata=tier_metadata,
                max_pending_trials=10,
                min_failed_trials_for_failure_rate_check=50,
            ).compute(experiment=self.experiment)
            self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
            self.assertIn("Unsupported", card.subtitle)
            self.assertIn("min_failed_trials_for_failure_rate_check", card.subtitle)

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
        self.experiment._search_space = SearchSpace(parameters=params)

        self.assertTrue(is_unordered_choice(params[1], min_choices=3, max_choices=5))

        card = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata).compute(
            experiment=self.experiment
        )

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
        self.experiment._search_space = SearchSpace(parameters=params)

        for p in params:
            self.assertTrue(can_map_to_binary(p))

        card = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata).compute(
            experiment=self.experiment
        )

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
        experiment = self.experiment
        experiment._search_space = SearchSpace(parameters=params)
        tier_metadata = {"user_supplied_max_trials": 300, "uses_standard_api": True}
        card = ComplexityRatingAnalysis(
            tier_metadata=tier_metadata,
            early_stopping_strategy=get_percentile_early_stopping_strategy(),
        ).compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 tunable parameter(s)", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)

    def test_dataframe_summary(self) -> None:
        card = ComplexityRatingAnalysis(tier_metadata=self.tier_metadata).compute(
            experiment=self.experiment
        )

        df = card.df
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 5)

        metrics = df["Metric"].tolist()
        self.assertIn("Optimization Complexity Rating", metrics)
        self.assertIn("Total Parameters", metrics)
        self.assertIn("Objectives", metrics)
        self.assertIn("Outcome Constraints", metrics)
        self.assertIn("Parameter Constraints", metrics)

        tier_row = df[df["Metric"] == "Optimization Complexity Rating"]
        self.assertEqual(tier_row.iloc[0]["Value"], "Standard")
