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
                objective=Objective(
                    expression="ne", metric_name_to_signature={"ne": "ne"}
                ),
            ),
        )

        with_constraints = optimization_config_from_string(
            objective_str="ne", outcome_constraint_strs=["qps >= 0"]
        )
        self.assertEqual(
            with_constraints,
            OptimizationConfig(
                objective=Objective(
                    expression="ne", metric_name_to_signature={"ne": "ne"}
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        expression="qps >= 0", metric_name_to_signature={"qps": "qps"}
                    ),
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
                objective=Objective(
                    expression="-ne, qps",
                    metric_name_to_signature={"ne": "ne", "qps": "qps"},
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        expression="flops <= 1000000",
                        metric_name_to_signature={"flops": "flops"},
                    ),
                ],
                objective_thresholds=[
                    OutcomeConstraint(
                        expression="qps >= 1000",
                        metric_name_to_signature={"qps": "qps"},
                    ),
                ],
            ),
        )

    def test_constraint_against_optimization_direction_on_objective(self) -> None:
        # A constraint that bounds a minimized objective from below (against
        # its optimization direction) cannot be an objective threshold, so it
        # must be kept as a true outcome constraint. The aligned upper bound
        # becomes an objective threshold.
        config = optimization_config_from_string(
            objective_str="-flops, -ne",
            outcome_constraint_strs=[
                "flops >= 42.50",
                "flops <= 94.38",
                "ne <= 0.62938",
            ],
        )
        self.assertEqual(
            config,
            MultiObjectiveOptimizationConfig(
                objective=Objective(
                    expression="-flops, -ne",
                    metric_name_to_signature={"flops": "flops", "ne": "ne"},
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        expression="flops >= 42.50",
                        metric_name_to_signature={"flops": "flops"},
                    ),
                ],
                objective_thresholds=[
                    OutcomeConstraint(
                        expression="flops <= 94.38",
                        metric_name_to_signature={"flops": "flops"},
                    ),
                    OutcomeConstraint(
                        expression="ne <= 0.62938",
                        metric_name_to_signature={"ne": "ne"},
                    ),
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
