#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.adapter.registry import Generators
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    _NO_RISK_MEASURE,
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.risk_measures import RiskMeasure
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.preference_stubs import get_pbo_experiment
from botorch.acquisition import LearnedObjective
from pyre_extensions import assert_is_instance


OC_STR = (
    'OptimizationConfig(objective=Objective(metric_name="m1", minimize=False), '
    "outcome_constraints=[OutcomeConstraint(m3 >= -0.25%), "
    "OutcomeConstraint(m4 <= 0.25%), "
    "ScalarizedOutcomeConstraint(metric_names=['m3', 'm4'], "
    "weights=[0.5, 0.5], >= -0.25%)])"
)

MOOC_STR = (
    "MultiObjectiveOptimizationConfig(objective=MultiObjective(objectives="
    '[Objective(metric_name="m1", minimize=True), '
    'Objective(metric_name="m2", minimize=False)]), '
    "outcome_constraints=[OutcomeConstraint(m3 >= -0.25%), "
    "OutcomeConstraint(m3 <= 0.25%)], objective_thresholds=[])"
)


class OptimizationConfigTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2"),
            "m3": Metric(name="m3"),
            "m4": Metric(name="m4"),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.alt_objective = Objective(metric=self.metrics["m3"], minimize=False)
        self.multi_objective = MultiObjective(
            objectives=[self.objective, self.alt_objective],
        )
        self.m2_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )
        self.outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=-0.25
        )
        self.additional_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m4"], op=ComparisonOp.LEQ, bound=0.25
        )
        self.scalarized_outcome_constraint = ScalarizedOutcomeConstraint(
            metrics=[self.metrics["m3"], self.metrics["m4"]],
            weights=[0.5, 0.5],
            op=ComparisonOp.GEQ,
            bound=-0.25,
        )
        self.outcome_constraints = [
            self.outcome_constraint,
            self.additional_outcome_constraint,
            self.scalarized_outcome_constraint,
        ]
        self.single_output_risk_measure = RiskMeasure(
            risk_measure="Expectation",
            options={"n_w": 2},
        )
        self.multi_output_risk_measure = RiskMeasure(
            risk_measure="MultiOutputExpectation",
            options={"n_w": 2},
        )

    def test_Init(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(str(config1), OC_STR)
        with self.assertRaises(ValueError):
            config1.objective = self.alt_objective  # constrained Objective.
        # updating constraints is fine.
        config1.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(len(config1.metrics), 2)

        # objective without outcome_constraints is also supported
        config2 = OptimizationConfig(objective=self.objective)
        self.assertEqual(config2.outcome_constraints, [])

        # setting objective is fine too, if it's compatible with constraints..
        config2.objective = self.m2_objective
        # setting constraints on objectives is fine for MultiObjective components.

        config2.outcome_constraints = self.outcome_constraints
        self.assertEqual(config2.outcome_constraints, self.outcome_constraints)

        # Risk measure is correctly registered.
        self.assertIsNone(config2.risk_measure)
        config3 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        expected_str = (
            OC_STR[:-1] + ", risk_measure=RiskMeasure(risk_measure=Expectation, "
            "options={'n_w': 2}))"
        )
        self.assertEqual(str(config3), expected_str)
        self.assertIs(config3.risk_measure, self.single_output_risk_measure)

    def test_Eq(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        config2 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertEqual(config1, config2)

        new_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m2"], op=ComparisonOp.LEQ, bound=0.5
        )
        config3 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=[self.outcome_constraint, new_outcome_constraint],
        )
        self.assertNotEqual(config1, config3)

    def test_ConstraintValidation(self) -> None:
        # Can build OptimizationConfig with MultiObjective
        with self.assertRaises(ValueError):
            OptimizationConfig(objective=self.multi_objective)

        # Can't constrain on objective metric.
        objective_constraint = OutcomeConstraint(
            metric=self.objective.metric, op=ComparisonOp.GEQ, bound=0
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective, outcome_constraints=[objective_constraint]
            )
        # Using an outcome constraint for ScalarizedObjective should also raise
        with self.assertRaisesRegex(
            ValueError, "Cannot constrain on objective metric."
        ):
            OptimizationConfig(
                objective=self.m2_objective,
                outcome_constraints=[objective_constraint],
            )
        # Two outcome_constraints on the same metric with the same op
        # should raise.
        duplicate_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            op=self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=[self.outcome_constraint, duplicate_constraint],
            )

        # Three outcome_constraints on the same metric should raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=self.outcome_constraints + [opposing_constraint],
            )

        # Two outcome_constraints on the same metric with different ops and
        # flipped bounds (lower < upper) should raise.
        add_bound = 1 if self.outcome_constraint.op == ComparisonOp.LEQ else -1
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + add_bound,
        )
        with self.assertRaises(ValueError):
            OptimizationConfig(
                objective=self.objective,
                outcome_constraints=([self.outcome_constraint, opposing_constraint]),
            )

        # Two outcome_constraints on the same metric with different ops and
        # bounds should not raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        config = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=([self.outcome_constraint, opposing_constraint]),
        )
        self.assertEqual(
            config.outcome_constraints, [self.outcome_constraint, opposing_constraint]
        )

    def test_Clone(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertEqual(config1, config1.clone())

    def test_CloneWithArgs(self) -> None:
        config1 = OptimizationConfig(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.single_output_risk_measure,
        )
        config2 = OptimizationConfig(
            objective=self.objective,
        )
        config3 = OptimizationConfig(
            objective=self.objective, risk_measure=_NO_RISK_MEASURE.clone()
        )

        # Empty args produce exact clone
        self.assertEqual(
            config1.clone_with_args(),
            config1,
        )

        # None args not treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                risk_measure=None,
            ),
            config2,
        )

        # Arguments that has same value with default won't be treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                risk_measure=config3.risk_measure,
            ),
            config3,
        )


class MultiObjectiveOptimizationConfigTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = {
            "m1": Metric(name="m1", lower_is_better=True),
            "m2": Metric(name="m2", lower_is_better=False),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        self.objectives = {
            "o1": Objective(metric=self.metrics["m1"]),
            "o2": Objective(metric=self.metrics["m2"], minimize=False),
            "o3": Objective(metric=self.metrics["m3"], minimize=False),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=True)
        self.multi_objective = MultiObjective(
            objectives=[self.objectives["o1"], self.objectives["o2"]]
        )
        self.multi_objective_just_m2 = MultiObjective(
            objectives=[self.objectives["o2"]]
        )
        self.scalarized_objective = ScalarizedObjective(
            metrics=list(self.metrics.values()),
            weights=[-1.0, 1.0, 1.0],
            minimize=False,
        )
        self.outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=-0.25
        )
        self.additional_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.LEQ, bound=0.25
        )
        self.outcome_constraints = [
            self.outcome_constraint,
            self.additional_outcome_constraint,
        ]
        self.objective_thresholds = [
            ObjectiveThreshold(metric=self.metrics["m1"], bound=-1.0, relative=False),
            ObjectiveThreshold(metric=self.metrics["m2"], bound=-1.0, relative=False),
        ]
        self.relative_objective_thresholds = [
            ObjectiveThreshold(metric=self.metrics["m1"], bound=-1.0, relative=True),
            ObjectiveThreshold(
                metric=self.metrics["m2"],
                op=ComparisonOp.GEQ,
                bound=-1.0,
                relative=True,
            ),
        ]
        self.m1_constraint = OutcomeConstraint(
            metric=self.metrics["m1"], op=ComparisonOp.LEQ, bound=0.1, relative=True
        )
        self.m3_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.GEQ, bound=0.1, relative=True
        )
        self.single_output_risk_measure = RiskMeasure(
            risk_measure="Expectation",
            options={"n_w": 2},
        )
        self.multi_output_risk_measure = RiskMeasure(
            risk_measure="MultiOutputExpectation",
            options={"n_w": 2},
        )

    def test_Init(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(str(config1), MOOC_STR)
        with self.assertRaisesRegex(
            TypeError,
            "`MultiObjectiveOptimizationConfig` requires an objective of type "
            "`MultiObjective` or `ScalarizedObjective`.",
        ):
            # pyre-fixme [8]: Incompatible attribute type
            config1.objective = self.objective  # Wrong objective type
        # updating constraints is fine.
        config1.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(len(config1.metrics), 3)

        # objective without outcome_constraints is also supported
        config2 = MultiObjectiveOptimizationConfig(objective=self.multi_objective)

        # setting objective is fine too, if it's compatible with constraints.
        config2.objective = self.multi_objective

        # setting constraints on objectives is fine for MultiObjective components.
        config2.outcome_constraints = [self.outcome_constraint]
        self.assertEqual(config2.outcome_constraints, [self.outcome_constraint])

        # construct constraints with objective_thresholds:
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
        )
        self.assertEqual(config3.all_constraints, self.objective_thresholds)

        # objective_thresholds and outcome constraints together.
        config4 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
            outcome_constraints=[self.m3_constraint],
        )
        self.assertEqual(
            config4.all_constraints, [self.m3_constraint] + self.objective_thresholds
        )
        self.assertEqual(config4.outcome_constraints, [self.m3_constraint])
        self.assertEqual(config4.objective_thresholds, self.objective_thresholds)

        # verify relative_objective_thresholds works:
        config5 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.relative_objective_thresholds,
        )
        threshold = config5.objective_thresholds[0]
        self.assertTrue(threshold.relative)
        self.assertEqual(threshold.bound, -1.0)

        # ValueError on wrong direction constraints
        with self.assertRaises(UserInputError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                # pyre-fixme[6]: For 2nd param expected
                #  `Optional[List[ObjectiveThreshold]]` but got
                #  `List[OutcomeConstraint]`.
                objective_thresholds=[self.additional_outcome_constraint],
            )

        # Test with risk measures.
        self.assertIsNone(config5.risk_measure)
        config6 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.multi_output_risk_measure,
        )
        self.assertIs(config6.risk_measure, self.multi_output_risk_measure)
        expected_str = (
            MOOC_STR[:-1] + ", risk_measure=RiskMeasure(risk_measure="
            "MultiOutputExpectation, options={'n_w': 2}))"
        )
        self.assertEqual(str(config6), expected_str)
        # With scalarized objective.
        config7 = MultiObjectiveOptimizationConfig(
            objective=self.scalarized_objective,
            risk_measure=self.single_output_risk_measure,
        )
        self.assertIs(config7.risk_measure, self.single_output_risk_measure)

    def test_Eq(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(config1, config2)

        new_outcome_constraint = OutcomeConstraint(
            metric=self.metrics["m3"], op=ComparisonOp.LEQ, bound=0.5
        )
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=[self.outcome_constraint, new_outcome_constraint],
        )
        self.assertNotEqual(config1, config3)

    def test_ConstraintValidation(self) -> None:
        # Cannot build with non-MultiObjective
        with self.assertRaisesRegex(
            TypeError,
            "`MultiObjectiveOptimizationConfig` requires an objective of type "
            "`MultiObjective` or `ScalarizedObjective`.",
        ):
            # pyre-fixme [6]: Incompatible parameter type
            MultiObjectiveOptimizationConfig(objective=self.objective)

        # Using an outcome constraint for an objective should raise
        outcome_constraint_m1 = OutcomeConstraint(
            metric=self.metrics["m1"], op=ComparisonOp.LEQ, bound=1234, relative=False
        )
        with self.assertRaisesRegex(
            ValueError, "Cannot constrain on objective metric."
        ):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=[outcome_constraint_m1],
            )
        # Two outcome_constraints on the same metric with the same op
        # should raise.
        duplicate_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            op=self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=[self.outcome_constraint, duplicate_constraint],
            )

        # Three outcome_constraints on the same metric should raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=self.outcome_constraints + [opposing_constraint],
            )

        # Two outcome_constraints on the same metric with different ops and
        # flipped bounds (lower < upper) should raise.
        add_bound = 1 if self.outcome_constraint.op == ComparisonOp.LEQ else -1
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + add_bound,
        )
        with self.assertRaises(ValueError):
            MultiObjectiveOptimizationConfig(
                objective=self.multi_objective,
                outcome_constraints=([self.outcome_constraint, opposing_constraint]),
            )

        # Two outcome_constraints on the same metric with different ops and
        # bounds should not raise.
        opposing_constraint = OutcomeConstraint(
            metric=self.outcome_constraint.metric,
            # pyre-fixme[6]: For 2nd param expected `ComparisonOp` but got `bool`.
            op=not self.outcome_constraint.op,
            bound=self.outcome_constraint.bound + 1,
        )
        config = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            outcome_constraints=([self.outcome_constraint, opposing_constraint]),
        )
        self.assertEqual(
            config.outcome_constraints, [self.outcome_constraint, opposing_constraint]
        )

    def test_Clone(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, outcome_constraints=self.outcome_constraints
        )
        self.assertEqual(config1, config1.clone())

        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
        )
        self.assertEqual(config2, config2.clone())

    def test_CloneWithArgs(self) -> None:
        config1 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
            objective_thresholds=self.objective_thresholds,
            outcome_constraints=self.outcome_constraints,
            risk_measure=self.multi_output_risk_measure,
        )
        config2 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective,
        )
        config3 = MultiObjectiveOptimizationConfig(
            objective=self.multi_objective, risk_measure=_NO_RISK_MEASURE.clone()
        )

        # Empty args produce exact clone
        self.assertEqual(
            config1.clone_with_args(),
            config1,
        )

        # None args not treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                objective_thresholds=None,
                risk_measure=None,
            ),
            config2,
        )

        # Arguments that has same value with default won't be treated as default
        self.assertEqual(
            config1.clone_with_args(
                outcome_constraints=None,
                objective_thresholds=None,
                risk_measure=config3.risk_measure,
            ),
            config3,
        )


class PreferenceOptimizationConfigTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = {
            "metric1": Metric(name="metric1", lower_is_better=True),
            "metric2": Metric(name="metric2", lower_is_better=False),
            "metric3": Metric(name="metric3", lower_is_better=False),
        }
        self.objectives = {
            "o1": Objective(metric=self.metrics["metric1"], minimize=True),
            "o2": Objective(metric=self.metrics["metric2"], minimize=False),
            "o3": Objective(metric=self.metrics["metric3"], minimize=False),
        }
        self.multi_objective = MultiObjective(
            objectives=[self.objectives["o2"], self.objectives["o3"]]
        )
        self.preference_profile_name = "pe_exp"

    def test_Init(self) -> None:
        # Test basic initialization
        config = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name=self.preference_profile_name,
        )
        self.assertEqual(config.preference_profile_name, self.preference_profile_name)
        self.assertEqual(config.objective, self.multi_objective)
        self.assertEqual(config.outcome_constraints, [])
        self.assertIsNone(config.risk_measure)

        # Test that outcome_constraints are not supported
        with self.assertRaisesRegex(
            NotImplementedError, "Outcome constraints are not yet supported"
        ):
            PreferenceOptimizationConfig(
                objective=self.multi_objective,
                preference_profile_name=self.preference_profile_name,
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=self.metrics["metric1"],
                        op=ComparisonOp.LEQ,
                        bound=0.5,
                    )
                ],
            )

    def test_Eq(self) -> None:
        config1 = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name=self.preference_profile_name,
        )
        config2 = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name=self.preference_profile_name,
        )
        self.assertEqual(config1, config2)

        # Different preference_profile_name
        config3 = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name="different_profile",
        )
        self.assertNotEqual(config1, config3)

        # Different objective
        different_objective = MultiObjective(
            objectives=[self.objectives["o1"], self.objectives["o2"]]
        )
        config4 = PreferenceOptimizationConfig(
            objective=different_objective,
            preference_profile_name=self.preference_profile_name,
        )
        self.assertNotEqual(config1, config4)

    def test_Clone(self) -> None:
        config = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name=self.preference_profile_name,
        )
        cloned_config = assert_is_instance(config.clone(), PreferenceOptimizationConfig)
        self.assertEqual(config, cloned_config)
        self.assertIsNot(config, cloned_config)
        self.assertEqual(
            cloned_config.preference_profile_name, self.preference_profile_name
        )
        self.assertEqual(cloned_config.objective, self.multi_objective)

        config = PreferenceOptimizationConfig(
            objective=self.multi_objective,
            preference_profile_name=self.preference_profile_name,
        )

        # ======= Clone with args =======
        # Empty args produce exact clone
        cloned_config = config.clone_with_args()
        self.assertEqual(config, cloned_config)
        self.assertIsNot(config, cloned_config)

        # Clone with different objective
        different_objective = MultiObjective(
            objectives=[self.objectives["o1"], self.objectives["o3"]]
        )
        cloned_with_diff_objective = config.clone_with_args(
            objective=different_objective
        )
        self.assertEqual(cloned_with_diff_objective.objective, different_objective)
        self.assertEqual(
            cloned_with_diff_objective.preference_profile_name,
            self.preference_profile_name,
        )

        # Clone with different preference_profile_name
        different_profile = "different_profile"
        cloned_with_diff_profile = config.clone_with_args(
            preference_profile_name=different_profile
        )
        self.assertEqual(cloned_with_diff_profile.objective, self.multi_objective)
        self.assertEqual(
            cloned_with_diff_profile.preference_profile_name, different_profile
        )

    @mock_botorch_optimize
    def test_CandidateGenerationWithLearnedObjective(self) -> None:
        # Create fake experiments
        pref_metrics = ["metric2", "metric3"]
        metric_names = ["metric1", "metric2", "metric3"]

        # Create preference exploration experiment
        pe_exp = get_pbo_experiment(
            num_parameters=len(pref_metrics),
            num_experimental_metrics=0,
            parameter_names=pref_metrics,
            num_experimental_trials=0,
            num_preference_trials=3,
            num_preference_trials_w_repeated_arm=5,
            unbounded_search_space=True,
            experiment_name="pe_exp",
        )

        # Create main experiment
        exp = get_pbo_experiment(
            num_parameters=4,
            num_experimental_metrics=3,
            tracking_metric_names=metric_names,
            num_experimental_trials=4,
            num_preference_trials=0,
            num_preference_trials_w_repeated_arm=0,
            experiment_name="bo_exp",
        )

        # Create PreferenceOptimizationConfig
        pref_opt_config = PreferenceOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=exp.metrics[pref_m], minimize=False)
                    for pref_m in pref_metrics
                ]
            ),
            preference_profile_name=pe_exp.name,
        )

        # Add auxiliary experiment to main experiment
        exp.add_auxiliary_experiment(
            purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            auxiliary_experiment=AuxiliaryExperiment(experiment=pe_exp),
        )
        exp._optimization_config = pref_opt_config

        # Generate candidates
        m = Generators.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.lookup_data(),
            optimization_config=pref_opt_config,
        )

        # Generate candidates and make sure LearnedObjective is constructed
        with patch(
            "ax.generators.torch.utils.LearnedObjective",
            wraps=LearnedObjective,
        ) as MockLearnedObjective:
            m.gen(n=2)
            MockLearnedObjective.assert_called_once()
