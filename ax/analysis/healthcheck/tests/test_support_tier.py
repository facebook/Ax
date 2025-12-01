# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.support_tier import SupportTierAnalysis
from ax.core.experiment import Experiment
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
from ax.fb.adapter.utils import can_map_to_binary, is_unordered_choice
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


def summarize_experiment(
    experiment: Experiment,
    max_trials: int | None = None,
    uses_early_stopping: bool = False,
    uses_global_stopping: bool = False,
    tolerated_trial_failure_rate: float | None = None,
    max_pending_trials: int | None = None,
    min_failed_trials_for_failure_rate_check: int | None = None,
    all_inputs_are_configs: bool = True,
    uses_merge_multiple_curves: bool = False,
    non_default_advanced_options: bool = False,
) -> dict[str, object]:
    """Helper function to summarize an experiment for SupportTierAnalysis."""
    search_space = experiment.search_space
    optimization_config = experiment.optimization_config
    assert optimization_config is not None

    params = search_space.tunable_parameters.values()

    num_params = len(search_space.tunable_parameters)
    num_binary = sum(can_map_to_binary(p) for p in params)
    num_categorical_3_5 = sum(
        is_unordered_choice(p, min_choices=3, max_choices=5) for p in params
    )
    num_categorical_6_inf = sum(is_unordered_choice(p, min_choices=6) for p in params)
    num_parameter_constraints = len(search_space.parameter_constraints)
    num_objectives = (
        len(optimization_config.objective.objectives)
        if isinstance(optimization_config.objective, MultiObjective)
        else 1
    )
    num_outcome_constraints = len(optimization_config.outcome_constraints)

    return {
        "max_trials": max_trials,
        "num_params": num_params,
        "num_binary": num_binary,
        "num_categorical_3_5": num_categorical_3_5,
        "num_categorical_6_inf": num_categorical_6_inf,
        "num_parameter_constraints": num_parameter_constraints,
        "num_objectives": num_objectives,
        "num_outcome_constraints": num_outcome_constraints,
        "uses_early_stopping": uses_early_stopping,
        "uses_global_stopping": uses_global_stopping,
        "tolerated_trial_failure_rate": tolerated_trial_failure_rate,
        "max_pending_trials": max_pending_trials,
        "min_failed_trials_for_failure_rate_check": (
            min_failed_trials_for_failure_rate_check
        ),
        "all_inputs_are_configs": all_inputs_are_configs,
        "uses_merge_multiple_curves": uses_merge_multiple_curves,
        "non_default_advanced_options": non_default_advanced_options,
    }


class TestSupportTierAnalysis(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()

    def test_wheelhouse_configuration(self) -> None:
        summary = summarize_experiment(self.experiment, max_trials=100)
        healthcheck = SupportTierAnalysis(experiment_summary=summary)
        card = healthcheck.compute()

        self.assertEqual(card.name, "SupportTierAnalysis")
        self.assertEqual(card.title, "Support Tier Healthcheck")
        self.assertTrue(card.is_passing())
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertIn("Wheelhouse", card.subtitle)
        self.assertIn("fully supported", card.subtitle)
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
                card = SupportTierAnalysis(experiment_summary=summary).compute()

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
                card = SupportTierAnalysis(experiment_summary=summary).compute()

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
            card = SupportTierAnalysis(experiment_summary=summary).compute()

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
            card = SupportTierAnalysis(experiment_summary=summary).compute()

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
                card = SupportTierAnalysis(experiment_summary=summary).compute()

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
                card = SupportTierAnalysis(experiment_summary=summary).compute()

                self.assertEqual(card.get_status(), expected_status)
                self.assertIn(expected_tier, card.subtitle)
                self.assertIn(expected_msg, card.subtitle)

    def test_unsupported_configurations(self) -> None:
        test_cases = [
            (
                "not_using_configs",
                {"all_inputs_are_configs": False},
                "Using Ax abstractions",
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
                card = SupportTierAnalysis(experiment_summary=summary).compute()

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
        card = SupportTierAnalysis(experiment_summary=summary).compute()

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
        card = SupportTierAnalysis(experiment_summary=summary).compute()

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
        card = SupportTierAnalysis(experiment_summary=summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 tunable parameter(s)", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)

    def test_dataframe_summary(self) -> None:
        summary = summarize_experiment(self.experiment, max_trials=100)
        card = SupportTierAnalysis(experiment_summary=summary).compute()

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
        custom_summary = {
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

        card = SupportTierAnalysis(experiment_summary=custom_summary).compute()

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)
