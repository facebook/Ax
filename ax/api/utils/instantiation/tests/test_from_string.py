# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from ax.api.utils.instantiation.from_string import optimization_config_from_string
from ax.core.objective import Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestFromString(TestCase):
    def test_optimization_config_from_string(self) -> None:
        only_objective = optimization_config_from_string(objective_str="ne")
        self.assertEqual(
            only_objective,
            OptimizationConfig(
                objective=Objective(expression="ne"),
            ),
        )

        with_constraints = optimization_config_from_string(
            objective_str="ne", outcome_constraint_strs=["qps >= 0"]
        )
        self.assertEqual(
            with_constraints,
            OptimizationConfig(
                objective=Objective(expression="ne"),
                outcome_constraints=[
                    OutcomeConstraint(expression="qps >= 0"),
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
                objective=Objective(expression="-ne, qps"),
                outcome_constraints=[
                    OutcomeConstraint(expression="flops <= 1000000"),
                ],
                objective_thresholds=[
                    OutcomeConstraint(expression="qps >= 1000"),
                ],
            ),
        )

    def test_objective_constraint_on_single_objective_raises(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Outcome constraints may not be placed"
        ):
            optimization_config_from_string(
                objective_str="ne", outcome_constraint_strs=["ne >= 0"]
            )
