# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
from ax.analysis.healthcheck.search_space_analysis import (
    search_space_boundary_proportions,
    SearchSpaceAnalysis,
)
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestSearchSpaceAnalysis(TestCase):
    def test_search_space_analysis(self) -> None:
        experiment = get_branin_experiment(with_trial=False)
        arms = [
            Arm(name="1_0", parameters={"x1": 0.0, "x2": 1.0}),
            Arm(name="1_1", parameters={"x1": -5.0, "x2": 0.0}),
            Arm(name="1_2", parameters={"x1": -5.0, "x2": 1.0}),
        ]
        experiment.new_batch_trial(generator_run=GeneratorRun(arms=arms))
        ssa = SearchSpaceAnalysis(trial_index=0)
        card = ssa.compute(experiment=experiment)

        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Search space boundary check")
        expected_subtitle = (
            "More than 50.00% of Ax suggested values are at the bound for the "
            "parameters and constraints below. This may indicate that the "
            "optimal values are outside of these bounds. Consider relaxing the "
            "following bounds.\n\n"
            "| Boundary   | Bound       | "
            "Proportion of suggested values on boundary   |\n"
            "|:-----------|:------------|"
            ":---------------------------------------------|\n"
            "| x1 = -5.0  | Lower bound |"
            " 66.67%                                       |"
        )
        self.assertEqual(card.subtitle, expected_subtitle)

        arms = [
            Arm(name="2_0", parameters={"x1": 5.0, "x2": 1.0}),
            Arm(name="2_1", parameters={"x1": 5.0, "x2": 0.0}),
            Arm(name="2_2", parameters={"x1": -5.0, "x2": 2.0}),
        ]
        experiment.new_batch_trial(generator_run=GeneratorRun(arms=arms))
        ssa = SearchSpaceAnalysis(trial_index=1)
        card = ssa.compute(experiment=experiment)
        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Search space boundary check")
        self.assertEqual(card.subtitle, "Search space does not need to be updated.")

        arms = [
            Arm(name="2_0", parameters={"x1": 5.0, "x2": 1.0}),
            Arm(name="3_1", parameters={"x1": -5.0, "x2": 0.5}),
            Arm(name="2_2", parameters={"x1": -5.0, "x2": 2.0}),
        ]
        experiment.new_batch_trial(generator_run=GeneratorRun(arms=arms))
        ssa = SearchSpaceAnalysis(trial_index=2)
        card = ssa.compute(experiment=experiment)
        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Search space boundary check")
        self.assertTrue(card.subtitle is not None and "x1" in card.subtitle)

    def test_search_space_boundary_proportions(self) -> None:
        ss = SearchSpace(
            parameters=[
                RangeParameter(
                    name="float_range_1",
                    parameter_type=ParameterType.FLOAT,
                    lower=1.0,
                    upper=6.0,
                ),
                RangeParameter(
                    name="float_range_2",
                    parameter_type=ParameterType.FLOAT,
                    lower=1.0,
                    upper=3.0,
                ),
                ChoiceParameter(
                    name="choice_ordered",
                    parameter_type=ParameterType.INT,
                    values=[1, 2, 3],
                    is_ordered=True,
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(inequality="float_range_1 + float_range_2 <= 4")
            ],
        )

        parameterizations: list[dict[str, None | bool | float | int | str]] = [
            {
                "float_range_1": 1.0,
                "float_range_2": 1.0,
                "choice_ordered": 1,
            },
            {
                "float_range_1": 2.0,
                "float_range_2": 2.0,
                "choice_ordered": 1,
            },
            {"float_range_1": 1.0, "float_range_2": 3.0, "choice_ordered": 3},
        ]

        df = search_space_boundary_proportions(
            search_space=ss, parameterizations=parameterizations
        )

        dff = pd.DataFrame(
            {
                "Boundary": [
                    "float_range_1 = 1.0",
                    "float_range_1 = 6.0",
                    "float_range_2 = 1.0",
                    "float_range_2 = 3.0",
                    "choice_ordered = 1",
                    "choice_ordered = 3",
                    "(1.0*float_range_1) + (1.0*float_range_2) = 4.0",
                ],
                "Bound": [
                    "Lower bound",
                    "Upper bound",
                    "Lower bound",
                    "Upper bound",
                    "Lower bound",
                    "Upper bound",
                    "Parameter constraint",
                ],
                "Proportion of suggested values on boundary": [
                    2.0 / 3,
                    0.0,
                    1.0 / 3,
                    1.0 / 3,
                    2.0 / 3,
                    1.0 / 3,
                    2.0 / 3,
                ],
            }
        )
        self.assertTrue(dff.equals(df))

        # test with no parameterizations
        df = search_space_boundary_proportions(search_space=ss, parameterizations=[])
        self.assertTrue(
            np.all(
                np.equal(
                    df["Proportion of suggested values on boundary"].values, np.zeros(7)
                )
            )
        )
        # test with no parameterizations in SS
        df = search_space_boundary_proportions(
            search_space=ss,
            parameterizations=[
                {
                    "float_range_1": 0.0,
                    "float_range_2": 1.0,
                    "choice_ordered": 1,
                },
            ],
        )
        self.assertTrue(
            np.all(
                np.equal(
                    df["Proportion of suggested values on boundary"].values, np.zeros(7)
                )
            )
        )
