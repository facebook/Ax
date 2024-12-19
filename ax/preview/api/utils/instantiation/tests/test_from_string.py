# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.exceptions.core import UserInputError
from ax.preview.api.utils.instantiation.from_string import (
    _sanitize_dot,
    optimization_config_from_string,
    parse_objective,
    parse_outcome_constraint,
    parse_parameter_constraint,
)
from ax.utils.common.testutils import TestCase


class TestFromString(TestCase):
    def test_optimization_config_from_string(self) -> None:
        only_objective = optimization_config_from_string(objective_str="ne")
        self.assertEqual(
            only_objective,
            OptimizationConfig(
                objective=Objective(metric=MapMetric(name="ne"), minimize=False),
            ),
        )

        with_constraints = optimization_config_from_string(
            objective_str="ne", outcome_constraint_strs=["qps >= 0"]
        )
        self.assertEqual(
            with_constraints,
            OptimizationConfig(
                objective=Objective(metric=MapMetric(name="ne"), minimize=False),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=MapMetric(name="qps"),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=False,
                    )
                ],
            ),
        )

        with_constraints_and_objective_threshold = optimization_config_from_string(
            objective_str="-ne, qps",
            outcome_constraint_strs=["qps >= 1000", "flops <= 1000000"],
        )
        self.assertEqual(
            with_constraints_and_objective_threshold,
            MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[
                        Objective(metric=MapMetric(name="ne"), minimize=True),
                        Objective(metric=MapMetric(name="qps"), minimize=False),
                    ]
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=MapMetric(name="flops"),
                        op=ComparisonOp.LEQ,
                        bound=1000000.0,
                        relative=False,
                    )
                ],
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=MapMetric(name="qps"),
                        op=ComparisonOp.GEQ,
                        bound=1000.0,
                        relative=False,
                    )
                ],
            ),
        )

    def test_parse_paramter_constraint(self) -> None:
        constraint = parse_parameter_constraint(constraint_str="x1 + x2 <= 1")
        self.assertEqual(
            constraint,
            ParameterConstraint(constraint_dict={"x1": 1, "x2": 1}, bound=1.0),
        )

        with_coefficients = parse_parameter_constraint(
            constraint_str="2 * x1 + 3 * x2 <= 1"
        )
        self.assertEqual(
            with_coefficients,
            ParameterConstraint(constraint_dict={"x1": 2, "x2": 3}, bound=1.0),
        )

        flipped_sign = parse_parameter_constraint(constraint_str="x1 + x2 >= 1")
        self.assertEqual(
            flipped_sign,
            ParameterConstraint(constraint_dict={"x1": -1, "x2": -1}, bound=-1.0),
        )

        weird = parse_parameter_constraint(constraint_str="x1 + x2 <= 1.5 * x3 + 2")
        self.assertEqual(
            weird,
            ParameterConstraint(
                constraint_dict={"x1": 1, "x2": 1, "x3": -1.5}, bound=2.0
            ),
        )

        with self.assertRaisesRegex(UserInputError, "Only linear"):
            parse_parameter_constraint(constraint_str="x1 * x2 <= 1")

    def test_parse_objective(self) -> None:
        single_objective = parse_objective(objective_str="ne")
        self.assertEqual(
            single_objective, Objective(metric=MapMetric(name="ne"), minimize=False)
        )

        maximize_single_objective = parse_objective(objective_str="-qps")
        self.assertEqual(
            maximize_single_objective,
            Objective(metric=MapMetric(name="qps"), minimize=True),
        )

        scalarized_objective = parse_objective(
            objective_str="0.5 * ne1 + 0.3 * ne2 + 0.2 * ne3"
        )
        self.assertEqual(
            scalarized_objective,
            ScalarizedObjective(
                metrics=[
                    MapMetric(name="ne1"),
                    MapMetric(name="ne2"),
                    MapMetric(name="ne3"),
                ],
                weights=[0.5, 0.3, 0.2],
                minimize=False,
            ),
        )

        multiobjective = parse_objective(objective_str="ne, -qps")
        self.assertEqual(
            multiobjective,
            MultiObjective(
                objectives=[
                    Objective(metric=MapMetric(name="ne"), minimize=False),
                    Objective(metric=MapMetric(name="qps"), minimize=True),
                ]
            ),
        )

        with self.assertRaisesRegex(UserInputError, "Only linear"):
            parse_objective(objective_str="ne * qps")

    def test_parse_outcome_constraint(self) -> None:
        constraint = parse_outcome_constraint(constraint_str="flops <= 1000000")
        self.assertEqual(
            constraint,
            OutcomeConstraint(
                metric=MapMetric(name="flops"),
                op=ComparisonOp.LEQ,
                bound=1000000.0,
                relative=False,
            ),
        )

        flipped_sign = parse_outcome_constraint(constraint_str="flops >= 1000000.0")
        self.assertEqual(
            flipped_sign,
            OutcomeConstraint(
                metric=MapMetric(name="flops"),
                op=ComparisonOp.GEQ,
                bound=1000000.0,
                relative=False,
            ),
        )

        relative = parse_outcome_constraint(constraint_str="flops <= 105 * baseline")
        self.assertEqual(
            relative,
            OutcomeConstraint(
                metric=MapMetric(name="flops"),
                op=ComparisonOp.LEQ,
                bound=105.0,
                relative=True,
            ),
        )

        scalarized = parse_outcome_constraint(
            constraint_str="0.5 * flops1 + 0.3 * flops2 <= 1000000"
        )
        self.assertEqual(
            scalarized,
            ScalarizedOutcomeConstraint(
                metrics=[MapMetric(name="flops1"), MapMetric(name="flops2")],
                weights=[0.5, 0.3],
                op=ComparisonOp.LEQ,
                bound=1000000.0,
                relative=False,
            ),
        )

        with self.assertRaisesRegex(UserInputError, "Expected an inequality"):
            parse_outcome_constraint(constraint_str="flops == 1000000")

        with self.assertRaisesRegex(UserInputError, "Only linear"):
            parse_outcome_constraint(constraint_str="flops * flops <= 1000000")

    def test_sanitize_dot(self) -> None:
        self.assertEqual(_sanitize_dot("foo.bar.baz"), "foo__dot__bar__dot__baz")

        constraint = parse_parameter_constraint(constraint_str="foo.bar + foo.baz <= 1")
        self.assertEqual(
            constraint,
            ParameterConstraint(
                constraint_dict={"foo.bar": 1, "foo.baz": 1}, bound=1.0
            ),
        )
