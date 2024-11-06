# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from ax.analysis.plotly.utils import get_constraint_violated_probabilities
from ax.core.metric import Metric
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestUtils(TestCase):
    def test_no_constraints_violates_none(self) -> None:
        constraint_violated_probabilities = get_constraint_violated_probabilities(
            # predictions for 2 observations on metrics a and b
            predictions=[
                (
                    {"a": 1.0, "b": 2.0},
                    {"a": 0.1, "b": 0.2},
                ),
                (
                    {"a": 1.1, "b": 2.1},
                    {"a": 0.1, "b": 0.2},
                ),
            ],
            outcome_constraints=[],
        )
        self.assertEqual(
            constraint_violated_probabilities, {"any_constraint_violated": [0.0, 0.0]}
        )

    def test_relative_constraints_are_not_accepted(self) -> None:
        with self.assertRaisesRegex(
            UserInputError,
            "does not support relative outcome constraints",
        ):
            get_constraint_violated_probabilities(
                predictions=[],
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric("a"),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=True,
                    )
                ],
            )

    def test_it_gives_a_result_per_constraint_plus_overall(self) -> None:
        constraint_violated_probabilities = get_constraint_violated_probabilities(
            # predictions for 2 observations on metrics a and b
            predictions=[
                (
                    {"a": 1.0, "b": 2.0},
                    {"a": 0.1, "b": 0.2},
                ),
                (
                    {"a": 1.1, "b": 2.1},
                    {"a": 0.1, "b": 0.2},
                ),
            ],
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric("a"),
                    op=ComparisonOp.GEQ,
                    bound=0.9,
                    relative=False,
                ),
                OutcomeConstraint(
                    metric=Metric("b"),
                    op=ComparisonOp.LEQ,
                    bound=2.2,
                    relative=False,
                ),
            ],
        )
        self.assertEqual(
            len(constraint_violated_probabilities.keys()),
            3,
        )
        self.assertIn("any_constraint_violated", constraint_violated_probabilities)
        self.assertIn("a", constraint_violated_probabilities)
        self.assertIn("b", constraint_violated_probabilities)
        self.assertAlmostEqual(
            constraint_violated_probabilities["any_constraint_violated"][0],
            0.292,
            places=2,
        )
        self.assertAlmostEqual(
            constraint_violated_probabilities["any_constraint_violated"][1],
            0.324,
            places=2,
        )
        self.assertAlmostEqual(
            constraint_violated_probabilities["a"][0], 0.158, places=2
        )
        self.assertAlmostEqual(
            constraint_violated_probabilities["a"][1], 0.022, places=2
        )
        self.assertAlmostEqual(
            constraint_violated_probabilities["b"][0], 0.158, places=2
        )
        self.assertAlmostEqual(
            constraint_violated_probabilities["b"][1], 0.308, places=2
        )
