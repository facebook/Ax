# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import (
    Ackley,
    ConstrainedGramacy,
    ConstrainedHartmann,
)
from hypothesis import given, strategies as st


class TestBenchmarkProblem(TestCase):
    def test_single_objective_from_botorch(self) -> None:
        for botorch_test_problem in [Ackley(), ConstrainedHartmann(dim=6)]:
            test_problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=botorch_test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
            )

            # Test search space
            self.assertEqual(
                len(test_problem.search_space.parameters), botorch_test_problem.dim
            )
            self.assertEqual(
                len(test_problem.search_space.parameters),
                len(test_problem.search_space.range_parameters),
            )
            self.assertTrue(
                all(
                    # pyre-fixme[16]: `Parameter` has no attribute `lower`.
                    test_problem.search_space.range_parameters[f"x{i}"].lower
                    == botorch_test_problem._bounds[i][0]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' lower bounds must all match Botorch problem's bounds.",
            )
            self.assertTrue(
                all(
                    # pyre-fixme[16]: `Parameter` has no attribute `upper`.
                    test_problem.search_space.range_parameters[f"x{i}"].upper
                    == botorch_test_problem._bounds[i][1]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' upper bounds must all match Botorch problem's bounds.",
            )

            # Test optimum
            self.assertEqual(
                test_problem.optimal_value, botorch_test_problem._optimal_value
            )
            # test optimization config
            self.assertEqual(
                test_problem.optimization_config.objective.metric.name,
                test_problem.name,
            )
            self.assertTrue(test_problem.optimization_config.objective.minimize)
            # test repr method
            if isinstance(botorch_test_problem, Ackley):
                self.assertEqual(
                    test_problem.optimization_config.outcome_constraints, []
                )
                expected_repr = (
                    "SingleObjectiveBenchmarkProblem(name=Ackley, "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="Ackley", '
                    "minimize=True), outcome_constraints=[]), "
                    "num_trials=1, "
                    "infer_noise=True, "
                    "tracking_metrics=[])"
                )
            else:
                outcome_constraint = (
                    test_problem.optimization_config.outcome_constraints[0]
                )
                self.assertEqual(outcome_constraint.metric.name, "constraint_slack_0")
                self.assertEqual(outcome_constraint.op, ComparisonOp.GEQ)
                self.assertFalse(outcome_constraint.relative)
                self.assertEqual(outcome_constraint.bound, 0.0)
                expected_repr = (
                    "SingleObjectiveBenchmarkProblem(name=ConstrainedHartmann, "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="ConstrainedHartmann", minimize=True), '
                    "outcome_constraints=[OutcomeConstraint(constraint_slack_0"
                    " >= 0.0)]), num_trials=1, infer_noise=True, "
                    "tracking_metrics=[])"
                )

            self.assertEqual(repr(test_problem), expected_repr)

    @given(
        st.booleans(),
        st.one_of(st.none(), st.just(0.1)),
        st.one_of(st.none(), st.just(0.2), st.just([0.3, 0.4])),
    )
    def test_constrained_from_botorch(
        self,
        infer_noise: bool,
        objective_noise_std: Optional[float],
        constraint_noise_std: Optional[Union[float, List[float]]],
    ) -> None:
        ax_problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem_class=ConstrainedGramacy,
            test_problem_kwargs={
                "noise_std": objective_noise_std,
                "constraint_noise_std": constraint_noise_std,
            },
            num_trials=1,
            infer_noise=infer_noise,
        )
        self.assertTrue(ax_problem.runner._is_constrained)
        botorch_problem = checked_cast(
            ConstrainedGramacy, ax_problem.runner.test_problem
        )
        self.assertEqual(botorch_problem.noise_std, objective_noise_std)
        self.assertEqual(botorch_problem.constraint_noise_std, constraint_noise_std)
        opt_config = ax_problem.optimization_config
        outcome_constraints = opt_config.outcome_constraints
        self.assertEqual(
            [constraint.metric.name for constraint in outcome_constraints],
            [f"constraint_slack_{i}" for i in range(botorch_problem.num_constraints)],
        )
        if infer_noise:
            expected_opt_noise_sd = None
        elif objective_noise_std is None:
            expected_opt_noise_sd = 0.0
        else:
            expected_opt_noise_sd = objective_noise_std

        self.assertEqual(opt_config.objective.metric.noise_sd, expected_opt_noise_sd)

        if infer_noise:
            expected_constraint_noise_sd = [None for _ in range(2)]
        elif constraint_noise_std is None:
            expected_constraint_noise_sd = [0.0 for _ in range(2)]
        elif isinstance(constraint_noise_std, float):
            expected_constraint_noise_sd = [constraint_noise_std for _ in range(2)]
        else:
            expected_constraint_noise_sd = constraint_noise_std

        self.assertEqual(
            [constraint.metric.noise_sd for constraint in outcome_constraints],
            expected_constraint_noise_sd,
        )

    def test_moo_from_botorch(self) -> None:
        test_problem = BraninCurrin()
        branin_currin_problem = (
            MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
            )
        )

        # Test search space
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters), test_problem.dim
        )
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters),
            len(branin_currin_problem.search_space.range_parameters),
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `lower`.
                branin_currin_problem.search_space.range_parameters[f"x{i}"].lower
                == test_problem._bounds[i][0]
                for i in range(test_problem.dim)
            ),
            "Parameters' lower bounds must all match Botorch problem's bounds.",
        )
        self.assertTrue(
            all(
                # pyre-fixme[16]: `Parameter` has no attribute `upper`.
                branin_currin_problem.search_space.range_parameters[f"x{i}"].upper
                == test_problem._bounds[i][1]
                for i in range(test_problem.dim)
            ),
            "Parameters' upper bounds must all match Botorch problem's bounds.",
        )

        # Test hypervolume
        self.assertEqual(
            branin_currin_problem.maximum_hypervolume, test_problem._max_hv
        )
        self.assertEqual(branin_currin_problem.reference_point, test_problem._ref_point)
