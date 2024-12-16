# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Union

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.search_space_analysis import (
    boundary_proportions_message,
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

        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Ax Search Space Analysis Warning")
        print(card.subtitle)
        subtitle = (
            "\n - Parameter x1 values are at their lower bound in 66.67% of all "
            "suggested parameters, which exceeds the threshold of 50.00%. "
            "Consider decreasing this lower bound of the search space and "
            "re-generating the candidates inside the expanded search space. "
        )
        self.assertEqual(card.subtitle, subtitle)

        arms = [
            Arm(name="2_0", parameters={"x1": 5.0, "x2": 1.0}),
            Arm(name="2_1", parameters={"x1": 5.0, "x2": 0.0}),
            Arm(name="2_2", parameters={"x1": -5.0, "x2": 2.0}),
        ]
        experiment.new_batch_trial(generator_run=GeneratorRun(arms=arms))
        ssa = SearchSpaceAnalysis(trial_index=1)
        card = ssa.compute(experiment=experiment)
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Ax Search Space Analysis Success")
        self.assertEqual(card.subtitle, "Search space does not need to be updated.")

        arms = [
            Arm(name="2_0", parameters={"x1": 5.0, "x2": 1.0}),
            Arm(name="3_1", parameters={"x1": -5.0, "x2": 0.5}),
            Arm(name="2_2", parameters={"x1": -5.0, "x2": 2.0}),
        ]
        experiment.new_batch_trial(generator_run=GeneratorRun(arms=arms))
        ssa = SearchSpaceAnalysis(trial_index=2)
        card = ssa.compute(experiment=experiment)
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.name, "SearchSpaceAnalysis")
        self.assertEqual(card.title, "Ax Search Space Analysis Warning")
        self.assertTrue("x1" in card.subtitle)

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
                ParameterConstraint(
                    constraint_dict={"float_range_1": 1.0, "float_range_2": 1.0},
                    bound=4.0,
                )
            ],
        )

        parametrizations: List[dict[str, Union[None, bool, float, int, str]]] = [
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
            search_space=ss, parametrizations=parametrizations
        )

        dff = pd.DataFrame(
            {
                "parameter_or_constraint": [
                    ss.parameters["float_range_1"],
                    ss.parameters["float_range_1"],
                    ss.parameters["float_range_2"],
                    ss.parameters["float_range_2"],
                    ss.parameters["choice_ordered"],
                    ss.parameters["choice_ordered"],
                    ss.parameter_constraints[0],
                ],
                "boundary": [
                    "float_range_1 = 1.0",
                    "float_range_1 = 6.0",
                    "float_range_2 = 1.0",
                    "float_range_2 = 3.0",
                    "choice_ordered = 1",
                    "choice_ordered = 3",
                    "1.0*float_range_1 + 1.0*float_range_2 = 4.0",
                ],
                "proportion": [
                    2.0 / 3,
                    0.0,
                    1.0 / 3,
                    1.0 / 3,
                    2.0 / 3,
                    1.0 / 3,
                    2.0 / 3,
                ],
                "bound": [
                    "lower",
                    "upper",
                    "lower",
                    "upper",
                    "lower",
                    "upper",
                    "upper",
                ],
            }
        )
        self.assertTrue(dff.equals(df))

    def test_boundary_proportions_message(self) -> None:
        float_range_1 = RangeParameter(
            name="float_range_1",
            parameter_type=ParameterType.FLOAT,
            lower=1.0,
            upper=6.0,
        )
        float_range_2 = RangeParameter(
            name="float_range_2",
            parameter_type=ParameterType.FLOAT,
            lower=1.0,
            upper=3.0,
        )
        pc = ParameterConstraint(
            constraint_dict={"float_range_1": 1.0, "float_range_2": 1.0}, bound=4.0
        )
        df = pd.DataFrame(
            {
                "parameter_or_constraint": [
                    float_range_1,
                    float_range_2,
                    pc,
                ],
                "boundary": [
                    "float_range_1 = 1.0",
                    "float_range_2 = 5.0",
                    "1.0*float_range_1 + 1.0*float_range_2 = 4.0",
                ],
                "proportion": [2.0 / 3, 0.0, 2.0 / 3],
                "bound": ["lower", "upper", "upper"],
            }
        )

        msg = boundary_proportions_message(boundary_proportions_df=df)
        self.assertTrue(
            "float_range_1" in msg
            and "ParameterConstraint(1.0*float_range_1 + 1.0*float_range_2 <= 4.0)"
            in msg
        )
