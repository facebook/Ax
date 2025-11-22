# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.support_tier import SupportTierHealthcheck
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


class TestSupportTierHealthcheck(TestCase):
    def test_wheelhouse_configuration(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.name, "SupportTierHealthcheck")
        self.assertEqual(card.title, "Support Tier Healthcheck")
        self.assertTrue(card.is_passing())
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertIn("Wheelhouse", card.subtitle)
        self.assertIn("fully supported", card.subtitle)

        additional_attrs = card.get_aditional_attrs()
        self.assertEqual(additional_attrs["tier"], "Wheelhouse")

    def test_advanced_tier_too_many_parameters(self) -> None:
        params = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(60)
        ]
        search_space = SearchSpace(parameters=params)
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 tunable parameters", card.subtitle)
        self.assertIn("max in-wheelhouse is 50", card.subtitle)

        additional_attrs = card.get_aditional_attrs()
        self.assertEqual(additional_attrs["tier"], "Advanced")

    def test_unsupported_tier_too_many_parameters(self) -> None:
        params = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(250)
        ]
        search_space = SearchSpace(parameters=params)
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertFalse(card.is_passing())
        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("250 tunable parameters", card.subtitle)
        self.assertIn("max supported is 200", card.subtitle)

        additional_attrs = card.get_aditional_attrs()
        self.assertEqual(additional_attrs["tier"], "Unsupported")

    def test_advanced_tier_too_many_objectives(self) -> None:
        experiment = get_branin_experiment()
        metrics = [Metric(name=f"m{i}") for i in range(3)]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[Objective(metric=m, minimize=False) for m in metrics]
            )
        )
        experiment._optimization_config = optimization_config

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("3 objectives", card.subtitle)
        self.assertIn("max in-wheelhouse is 2", card.subtitle)

    def test_unsupported_tier_too_many_objectives(self) -> None:
        experiment = get_branin_experiment()
        metrics = [Metric(name=f"m{i}") for i in range(5)]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[Objective(metric=m, minimize=False) for m in metrics]
            )
        )
        experiment._optimization_config = optimization_config

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("5 objectives", card.subtitle)
        self.assertIn("max supported is 4", card.subtitle)

    def test_advanced_tier_too_many_outcome_constraints(self) -> None:
        experiment = get_branin_experiment()
        metrics = [Metric(name=f"m{i}") for i in range(3)]
        outcome_constraints = [
            OutcomeConstraint(metric=m, op=ComparisonOp.LEQ, bound=1.0) for m in metrics
        ]
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="obj"), minimize=False),
            outcome_constraints=outcome_constraints,
        )
        experiment._optimization_config = optimization_config

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("3 outcome constraints", card.subtitle)
        self.assertIn("max in-wheelhouse is 2", card.subtitle)

    def test_advanced_tier_too_many_parameter_constraints(self) -> None:
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
        search_space = SearchSpace(
            parameters=params, parameter_constraints=parameter_constraints
        )
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("3 parameter constraints", card.subtitle)
        self.assertIn("max in-wheelhouse is 2", card.subtitle)

    def test_advanced_tier_early_stopping_enabled(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=100, uses_early_stopping=True)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)

    def test_advanced_tier_global_stopping_enabled(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=100, uses_global_stopping=True)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("Global stopping is enabled", card.subtitle)

    def test_advanced_tier_too_many_trials(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=300)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("max in-wheelhouse is 200", card.subtitle)

    def test_unsupported_tier_too_many_trials(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=600)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("600 total trials", card.subtitle)
        self.assertIn("max supported is 500", card.subtitle)

    def test_unsupported_tier_not_using_configs(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(
            max_trials=100, all_inputs_are_configs=False
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("Using Ax abstractions", card.subtitle)
        self.assertIn("instead of simple config objects", card.subtitle)

    def test_unsupported_tier_high_failure_rate(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(
            max_trials=100, tolerated_trial_failure_rate=0.95
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("0.95", card.subtitle)
        self.assertIn("larger than 0.9", card.subtitle)

    def test_unsupported_tier_invalid_failure_rate_check(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(
            max_trials=100,
            max_pending_trials=10,
            min_failed_trials_for_failure_rate_check=50,
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("min_failed_trials_for_failure_rate_check", card.subtitle)
        self.assertIn("(50) exceeds", card.subtitle)

    def test_unsupported_tier_non_default_advanced_options(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(
            max_trials=100, non_default_advanced_options=True
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("Non-default advanced_options", card.subtitle)

    def test_unsupported_tier_merge_multiple_curves(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(
            max_trials=100, uses_merge_multiple_curves=True
        )
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.FAIL)
        self.assertIn("Unsupported", card.subtitle)
        self.assertIn("merge_multiple_curves", card.subtitle)

    def test_unordered_choice_parameters(self) -> None:
        from ax.fb.adapter.utils import is_unordered_choice

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
        search_space = SearchSpace(parameters=params)
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        self.assertTrue(is_unordered_choice(params[1], min_choices=3, max_choices=5))

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("unordered choice parameters", card.subtitle)

    def test_multiple_violations_shows_all_reasons(self) -> None:
        params = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(60)
        ]
        search_space = SearchSpace(parameters=params)
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        healthcheck = SupportTierHealthcheck(max_trials=300, uses_early_stopping=True)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 tunable parameters", card.subtitle)
        self.assertIn("300 total trials", card.subtitle)
        self.assertIn("Early stopping is enabled", card.subtitle)

    def test_dataframe_contains_summary_info(self) -> None:
        experiment = get_branin_experiment()
        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

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

    def test_binary_parameters_count(self) -> None:
        from ax.fb.adapter.utils import can_map_to_binary

        params = [
            ChoiceParameter(
                name=f"binary{i}",
                parameter_type=ParameterType.BOOL,
                values=[True, False],
            )
            for i in range(60)
        ]
        search_space = SearchSpace(parameters=params)
        experiment = get_branin_experiment()
        experiment._search_space = search_space

        for p in params:
            self.assertTrue(can_map_to_binary(p))

        healthcheck = SupportTierHealthcheck(max_trials=100)
        card = healthcheck.compute(experiment=experiment)

        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("Advanced", card.subtitle)
        self.assertIn("60 binary parameters", card.subtitle)
        self.assertIn("max in-wheelhouse is 50", card.subtitle)
