# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Optional, Union

import torch

from ax.benchmark.benchmark_metric import BenchmarkMetric

from ax.benchmark.benchmark_problem import create_problem_from_botorch
from ax.benchmark.runners.botorch_test import BotorchTestProblemRunner
from ax.core.arm import Arm
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.multi_fidelity import AugmentedBranin
from botorch.test_functions.multi_objective import BraninCurrin, ConstrainedBraninCurrin
from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    ConstrainedGramacy,
    ConstrainedHartmann,
    Cosine8,
)
from hypothesis import given, strategies as st
from pyre_extensions import assert_is_instance


class TestBenchmarkProblem(TestCase):
    def setUp(self) -> None:
        # Print full output, so that any differences in 'repr' output are shown
        self.maxDiff = None
        super().setUp()

    def _test_multi_fidelity_or_multi_task(self, fidelity_or_task: str) -> None:
        """
        Args:
            fidelity_or_task: "fidelity" or "task"
        """
        parameters = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(2)
        ] + [
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                values=[0, 1],
                is_fidelity=fidelity_or_task == "fidelity",
                is_task=fidelity_or_task == "task",
                target_value=1,
            )
        ]
        problem = create_problem_from_botorch(
            test_problem_class=AugmentedBranin,
            test_problem_kwargs={},
            # pyre-fixme: Incompatible parameter type [6]: In call
            # `SearchSpace.__init__`, for 1st positional argument, expected
            # `List[Parameter]` but got `List[RangeParameter]`.
            search_space=SearchSpace(parameters),
            num_trials=3,
        )
        arm = Arm(parameters={"x0": 1.0, "x1": 0.0, "x2": 0.0})
        at_target = assert_is_instance(
            Branin()
            .evaluate_true(torch.tensor([1.0, 0.0], dtype=torch.double).unsqueeze(0))
            .item(),
            float,
        )
        self.assertAlmostEqual(
            problem.runner.evaluate_oracle(arm=arm).item(), at_target
        )
        # first term: (-(b - 0.1) * (1 - x3)  + c - r)^2
        # low-fidelity: (-b - 0.1 + c - r)^2
        # high-fidelity: (-b + c - r)^2
        t = -5.1 / (4 * math.pi**2) + 5 / math.pi - 6
        expected_change = (t + 0.1) ** 2 - t**2
        self.assertAlmostEqual(
            problem.runner.get_Y_true(arm=arm).item(),
            at_target + expected_change,
        )

    def test_multi_fidelity_or_multi_task(self) -> None:
        self._test_multi_fidelity_or_multi_task(fidelity_or_task="fidelity")
        self._test_multi_fidelity_or_multi_task(fidelity_or_task="task")

    def test_single_objective_from_botorch(self) -> None:
        for botorch_test_problem in [Ackley(), ConstrainedHartmann(dim=6)]:
            test_problem = create_problem_from_botorch(
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
                    test_problem.search_space.range_parameters[f"x{i}"].lower
                    == botorch_test_problem._bounds[i][0]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' lower bounds must all match Botorch problem's bounds.",
            )
            self.assertTrue(
                all(
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
                    "BenchmarkProblem(name='Ackley', "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="Ackley", '
                    "minimize=True), outcome_constraints=[]), "
                    "num_trials=1, "
                    "observe_noise_stds=False, "
                    "optimal_value=0.0)"
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
                    "BenchmarkProblem(name='ConstrainedHartmann', "
                    "optimization_config=OptimizationConfig(objective=Objective("
                    'metric_name="ConstrainedHartmann", minimize=True), '
                    "outcome_constraints=[OutcomeConstraint(constraint_slack_0"
                    " >= 0.0)]), "
                    "num_trials=1, "
                    "observe_noise_stds=False, "
                    "optimal_value=-3.32237)"
                )

            self.assertEqual(repr(test_problem), expected_repr)

    def _test_constrained_from_botorch(
        self,
        observe_noise_sd: bool,
        objective_noise_std: Optional[float],
        constraint_noise_std: Optional[Union[float, list[float]]],
        test_problem_class: type[ConstrainedBaseTestProblem],
    ) -> None:
        ax_problem = create_problem_from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs={
                "noise_std": objective_noise_std,
                "constraint_noise_std": constraint_noise_std,
            },
            lower_is_better=True,
            num_trials=1,
            observe_noise_sd=observe_noise_sd,
        )
        runner = checked_cast(BotorchTestProblemRunner, ax_problem.runner)
        self.assertTrue(runner._is_constrained)
        botorch_problem = checked_cast(ConstrainedBaseTestProblem, runner.test_problem)
        self.assertEqual(botorch_problem.noise_std, objective_noise_std)
        self.assertEqual(botorch_problem.constraint_noise_std, constraint_noise_std)
        opt_config = ax_problem.optimization_config
        outcome_constraints = opt_config.outcome_constraints
        self.assertEqual(
            [constraint.metric.name for constraint in outcome_constraints],
            [f"constraint_slack_{i}" for i in range(botorch_problem.num_constraints)],
        )
        objective = opt_config.objective
        metric = (
            objective.metrics[0]
            if isinstance(objective, MultiObjective)
            else objective.metric
        )

        self.assertEqual(
            checked_cast(BenchmarkMetric, metric).observe_noise_sd,
            observe_noise_sd,
        )

        # TODO: Support observing noise variance only for some outputs
        for constraint in outcome_constraints:
            self.assertEqual(
                checked_cast(BenchmarkMetric, constraint.metric).observe_noise_sd,
                observe_noise_sd,
            )

    # pyre-fixme[56]: Invalid decoration. Pyre was not able to infer the type of
    # argument `hypothesis.strategies.booleans()` to decorator factory
    # `hypothesis.given`.
    @given(
        st.booleans(),
        st.one_of(st.none(), st.just(0.1)),
        st.one_of(st.none(), st.just(0.2), st.just([0.3, 0.4])),
    )
    def test_constrained_soo_from_botorch(
        self,
        observe_noise_sd: bool,
        objective_noise_std: Optional[float],
        constraint_noise_std: Optional[Union[float, list[float]]],
    ) -> None:
        self._test_constrained_from_botorch(
            observe_noise_sd=observe_noise_sd,
            objective_noise_std=objective_noise_std,
            constraint_noise_std=constraint_noise_std,
            test_problem_class=ConstrainedGramacy,
        )

    def test_constrained_moo_from_botorch(self) -> None:
        self._test_constrained_from_botorch(
            observe_noise_sd=False,
            objective_noise_std=None,
            constraint_noise_std=None,
            test_problem_class=ConstrainedBraninCurrin,
        )

    def _test_moo_from_botorch(self, lower_is_better: bool) -> None:
        test_problem = BraninCurrin()
        branin_currin_problem = create_problem_from_botorch(
            test_problem_class=test_problem.__class__,
            test_problem_kwargs={},
            num_trials=1,
            lower_is_better=lower_is_better,
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
                branin_currin_problem.search_space.range_parameters[f"x{i}"].lower
                == test_problem._bounds[i][0]
                for i in range(test_problem.dim)
            ),
            "Parameters' lower bounds must all match Botorch problem's bounds.",
        )
        self.assertTrue(
            all(
                branin_currin_problem.search_space.range_parameters[f"x{i}"].upper
                == test_problem._bounds[i][1]
                for i in range(test_problem.dim)
            ),
            "Parameters' upper bounds must all match Botorch problem's bounds.",
        )

        # Test hypervolume
        self.assertEqual(branin_currin_problem.optimal_value, test_problem._max_hv)
        opt_config = assert_is_instance(
            branin_currin_problem.optimization_config, MultiObjectiveOptimizationConfig
        )
        reference_point = [
            threshold.bound for threshold in opt_config.objective_thresholds
        ]
        self.assertEqual(reference_point, test_problem._ref_point)

        self.assertTrue(
            all(
                t.op is (ComparisonOp.LEQ if lower_is_better else ComparisonOp.GEQ)
                for t in opt_config.objective_thresholds
            )
        )
        self.assertTrue(
            all(
                metric.lower_is_better is lower_is_better
                for metric in opt_config.metrics.values()
            )
        )

    def test_moo_from_botorch(self) -> None:
        self._test_moo_from_botorch(lower_is_better=True)
        self._test_moo_from_botorch(lower_is_better=False)

    def test_maximization_problem(self) -> None:
        test_problem = create_problem_from_botorch(
            test_problem_class=Cosine8,
            lower_is_better=False,
            num_trials=1,
            test_problem_kwargs={},
        )
        self.assertFalse(test_problem.optimization_config.objective.minimize)
