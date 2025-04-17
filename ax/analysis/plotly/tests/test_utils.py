# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.plotly.utils import (
    get_arm_tooltip,
    get_constraint_violated_probabilities,
    trial_status_to_plotly_color,
)
from ax.core.metric import Metric
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from plotly import express as px


class TestUtils(TestCase):
    def test_trial_status_to_plotly_color(self) -> None:
        plotly_blue = px.colors.hex_to_rgb(px.colors.qualitative.Plotly[0])
        solid_completed_blue = trial_status_to_plotly_color(
            trial_status="COMPLETED", ci_transparency=False
        )
        self.assertEqual(
            solid_completed_blue,
            f"rgba({plotly_blue[0]}, {plotly_blue[1]}, {plotly_blue[2]}, 1)",
        )

        ci_completed_blue = trial_status_to_plotly_color(
            trial_status="COMPLETED", ci_transparency=True
        )
        self.assertEqual(
            ci_completed_blue,
            f"rgba({plotly_blue[0]}, {plotly_blue[1]}, {plotly_blue[2]}, 0.5)",
        )

        # Ensure we do not get a KeyError on a trial status name that is not
        # expected.
        trial_status_to_plotly_color(trial_status="StrangeStatus")

    def test_get_arm_tooltip(self) -> None:
        row = pd.Series(
            {
                "trial_index": 5,
                "trial_status": "COMPLETED",
                "arm_name": "5_0",
                "foo_mean": 0.450508,
                "bar_mean": -0.222466,
                "foo_sem": 0.15232,
                "bar_sem": 0.638425,
                "generation_node": "MBM",
                "p_feasible": 0.1,
            }
        )

        foo_tooltip = get_arm_tooltip(row=row, metric_names=["foo"])
        self.assertEqual(
            foo_tooltip,
            "Trial: 5<br />Arm: 5_0<br />Status: COMPLETED<br />"
            "Generation Node: MBM<br />foo: 0.45051±0.29855<br />",
        )

        bar_tooltip = get_arm_tooltip(row=row, metric_names=["bar"])
        self.assertEqual(
            bar_tooltip,
            "Trial: 5<br />Arm: 5_0<br />Status: COMPLETED<br />"
            "Generation Node: MBM<br />bar: -0.22247±1.25131<br />",
        )

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
