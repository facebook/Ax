# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product

from ax.benchmark.benchmark_metric import BenchmarkMetric

from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    create_problem_from_botorch,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.multi_objective import BraninCurrin, ConstrainedBraninCurrin
from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    ConstrainedGramacy,
    ConstrainedHartmann,
    Cosine8,
)
from pyre_extensions import assert_is_instance


class TestBenchmarkProblem(TestCase):
    def setUp(self) -> None:
        # Print full output, so that any differences in 'repr' output are shown
        self.maxDiff = None
        super().setUp()

    def test_inference_value_not_implemented(self) -> None:
        objectives = [
            Objective(metric=BenchmarkMetric(name, lower_is_better=True))
            for name in ["Branin", "Currin"]
        ]
        optimization_config = OptimizationConfig(objective=objectives[0])
        test_function = BoTorchTestFunction(
            botorch_problem=Branin(), outcome_names=["Branin"]
        )
        with self.assertRaisesRegex(NotImplementedError, "Only `n_best_points=1`"):
            BenchmarkProblem(
                name="foo",
                optimization_config=optimization_config,
                num_trials=1,
                optimal_value=0.0,
                baseline_value=1.0,
                search_space=SearchSpace(parameters=[]),
                test_function=test_function,
                n_best_points=2,
            )

        with self.assertRaisesRegex(
            NotImplementedError, "Inference trace is not supported for MOO"
        ):
            BenchmarkProblem(
                name="foo",
                optimization_config=MultiObjectiveOptimizationConfig(
                    objective=MultiObjective(objectives)
                ),
                num_trials=1,
                optimal_value=0.0,
                search_space=SearchSpace(parameters=[]),
                baseline_value=1.0,
                test_function=test_function,
                n_best_points=1,
                report_inference_value_as_trace=True,
            )

    def test_mismatch_of_names_on_test_function_and_opt_config_raises(self) -> None:
        objectives = [
            Objective(metric=BenchmarkMetric(name, lower_is_better=True))
            for name in ["Branin", "Currin"]
        ]
        test_function = BoTorchTestFunction(
            botorch_problem=Branin(), outcome_names=["Branin"]
        )
        opt_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives)
        )
        with self.assertRaisesRegex(
            ValueError,
            "The following objectives are defined on "
            "`optimization_config` but not included in "
            "`runner.test_function.outcome_names`: {'Currin'}.",
        ):
            BenchmarkProblem(
                name="foo",
                optimization_config=opt_config,
                num_trials=1,
                optimal_value=1.0,
                search_space=SearchSpace(parameters=[]),
                test_function=test_function,
                baseline_value=0.0,
            )

        opt_config = OptimizationConfig(
            objective=objectives[0],
            outcome_constraints=[
                OutcomeConstraint(
                    BenchmarkMetric("c", lower_is_better=False),
                    ComparisonOp.LEQ,
                    0.0,
                )
            ],
        )
        with self.assertRaisesRegex(
            ValueError,
            "The following constraints are defined on "
            "`optimization_config` but not included in "
            "`runner.test_function.outcome_names`: {'c'}.",
        ):
            BenchmarkProblem(
                name="foo",
                optimization_config=opt_config,
                num_trials=1,
                optimal_value=0.0,
                search_space=SearchSpace(parameters=[]),
                test_function=test_function,
                baseline_value=1.0,
            )

    def test_single_objective_from_botorch(self) -> None:
        for botorch_test_problem in [Ackley(), ConstrainedHartmann(dim=6)]:
            test_problem = create_problem_from_botorch(
                test_problem_class=botorch_test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
                baseline_value=100.0,
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
            metric_name = test_problem.optimization_config.objective.metric.name
            self.assertEqual(metric_name, test_problem.name)
            self.assertTrue(test_problem.optimization_config.objective.minimize)
            # test repr method
            if isinstance(botorch_test_problem, Ackley):
                self.assertEqual(
                    test_problem.optimization_config.outcome_constraints, []
                )
            else:
                outcome_constraint = (
                    test_problem.optimization_config.outcome_constraints[0]
                )
                self.assertEqual(outcome_constraint.metric.name, "constraint_slack_0")
                self.assertEqual(outcome_constraint.op, ComparisonOp.GEQ)
                self.assertFalse(outcome_constraint.relative)
                self.assertEqual(outcome_constraint.bound, 0.0)

    def _test_constrained_from_botorch(
        self,
        observe_noise_sd: bool,
        noise_std: float | list[float],
        test_problem_class: type[ConstrainedBaseTestProblem],
    ) -> None:
        ax_problem = create_problem_from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs={},
            lower_is_better=True,
            num_trials=1,
            observe_noise_sd=observe_noise_sd,
            noise_std=noise_std,
        )
        test_problem = assert_is_instance(ax_problem.test_function, BoTorchTestFunction)
        botorch_problem = assert_is_instance(
            test_problem.botorch_problem, ConstrainedBaseTestProblem
        )
        self.assertEqual(ax_problem.noise_std, noise_std)
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
            assert_is_instance(metric, BenchmarkMetric).observe_noise_sd,
            observe_noise_sd,
        )

        # TODO: Support observing noise variance only for some outputs
        for constraint in outcome_constraints:
            self.assertEqual(
                assert_is_instance(constraint.metric, BenchmarkMetric).observe_noise_sd,
                observe_noise_sd,
            )

    def test_constrained_soo_from_botorch(self) -> None:
        for observe_noise_sd, noise_std in product(
            [False, True],
            [0.0, 0.1, [0.1, 0.3, 0.4]],
        ):
            with self.subTest(observe_noise_sd=observe_noise_sd, noise_std=noise_std):
                self._test_constrained_from_botorch(
                    observe_noise_sd=observe_noise_sd,
                    noise_std=noise_std,
                    test_problem_class=ConstrainedGramacy,
                )

    def test_constrained_moo_from_botorch(self) -> None:
        self._test_constrained_from_botorch(
            observe_noise_sd=False,
            noise_std=0.0,
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
            baseline_value=-8,
        )
        self.assertFalse(test_problem.optimization_config.objective.minimize)

    def test_sq_out_of_search_space(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Status quo parameters are not in the search space."
        ):
            create_problem_from_botorch(
                test_problem_class=Branin,
                lower_is_better=True,
                num_trials=1,
                test_problem_kwargs={},
                status_quo_params={"x0": 20.0, "x1": 20.0},
            )

    def test_get_soo_opt_config(self) -> None:
        opt_config = get_soo_opt_config(
            outcome_names=["foo", "bar"],
            lower_is_better=False,
            observe_noise_sd=True,
        )
        self.assertIsInstance(opt_config.objective, Objective)
        self.assertEqual(len(opt_config.outcome_constraints), 1)
        objective_metric = assert_is_instance(
            opt_config.objective.metric, BenchmarkMetric
        )
        self.assertEqual(objective_metric.name, "foo")
        self.assertEqual(objective_metric.observe_noise_sd, True)
        self.assertEqual(objective_metric.lower_is_better, False)
        constraint_metric = assert_is_instance(
            opt_config.outcome_constraints[0].metric, BenchmarkMetric
        )
        self.assertEqual(constraint_metric.name, "bar")
        self.assertEqual(constraint_metric.observe_noise_sd, True)
        self.assertEqual(constraint_metric.lower_is_better, False)

    def test_get_moo_opt_config(self) -> None:
        opt_config = get_moo_opt_config(
            outcome_names=["foo", "bar", "baz", "pony"],
            ref_point=[3.0, 4.0],
            lower_is_better=False,
            observe_noise_sd=True,
            num_constraints=2,
        )
        self.assertEqual(len(opt_config.objective.metrics), 2)
        self.assertEqual(len(opt_config.outcome_constraints), 2)
        self.assertEqual(opt_config.objective_thresholds[0].bound, 3.0)
        objective_metrics = [
            assert_is_instance(metric, BenchmarkMetric)
            for metric in opt_config.objective.metrics
        ]
        self.assertEqual(len(objective_metrics), 2)
        self.assertEqual(objective_metrics[0].name, "foo")
        self.assertEqual(objective_metrics[0].observe_noise_sd, True)
        self.assertEqual(objective_metrics[0].lower_is_better, False)
        self.assertEqual(objective_metrics[1].name, "bar")
        # Check constraints
        self.assertEqual(len(opt_config.outcome_constraints), 2)
        constraint_metrics = [
            assert_is_instance(constraint.metric, BenchmarkMetric)
            for constraint in opt_config.outcome_constraints
        ]
        self.assertEqual(constraint_metrics[0].name, "baz")
        self.assertEqual(constraint_metrics[0].observe_noise_sd, True)
        self.assertEqual(constraint_metrics[0].lower_is_better, False)
