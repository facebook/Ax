# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    get_continuous_search_space,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.core.objective import Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Branin


class TestBenchmarkProblem(TestCase):
    def setUp(self) -> None:
        # Print full output, so that any differences in 'repr' output are shown
        self.maxDiff = None
        super().setUp()

    def test_mismatch_of_names_on_test_function_and_opt_config_raises(self) -> None:
        test_function = BoTorchTestFunction(
            botorch_problem=Branin(), outcome_names=["Branin"]
        )
        opt_config = MultiObjectiveOptimizationConfig(
            objective=Objective(expression="-Branin, -Currin"),
            objective_thresholds=[
                OutcomeConstraint(expression="Branin <= 0.0"),
                OutcomeConstraint(expression="Currin <= 0.0"),
            ],
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

        opt_config2 = OptimizationConfig(
            objective=Objective(expression="-Branin"),
            outcome_constraints=[
                OutcomeConstraint(expression="c <= 0.0"),
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
                optimization_config=opt_config2,
                num_trials=1,
                optimal_value=0.0,
                search_space=SearchSpace(parameters=[]),
                test_function=test_function,
                baseline_value=1.0,
                worst_feasible_value=2.0,
            )

    def test_missing_names_on_test_function_with_scalarized_objective(self) -> None:
        objective = Objective(
            expression="-BraninCurrin_0 - BraninCurrin_missing",
        )
        test_function = BoTorchTestFunction(
            botorch_problem=BraninCurrin(),
            outcome_names=["BraninCurrin_0", "BraninCurrin_1"],
        )
        opt_config = OptimizationConfig(objective=objective)
        with self.assertRaisesRegex(
            ValueError,
            "The following objectives are defined on "
            "`optimization_config` but not included in "
            "`runner.test_function.outcome_names`: {'BraninCurrin_missing'}.",
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

    def test_get_soo_opt_config(self) -> None:
        opt_config, metrics = get_soo_opt_config(
            outcome_names=["foo", "bar"],
            lower_is_better=False,
            observe_noise_sd=True,
        )
        self.assertIsInstance(opt_config.objective, Objective)
        self.assertEqual(len(opt_config.outcome_constraints), 1)
        self.assertEqual(opt_config.objective.metric_names, ["foo"])
        self.assertFalse(opt_config.objective.minimize)
        self.assertEqual(opt_config.outcome_constraints[0].metric_names, ["bar"])
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].name, "foo")
        self.assertEqual(metrics[1].name, "bar")

    def test_get_moo_opt_config(self) -> None:
        opt_config, metrics = get_moo_opt_config(
            outcome_names=["foo", "bar", "baz", "pony"],
            ref_point=[3.0, 4.0],
            lower_is_better=False,
            observe_noise_sd=True,
            num_constraints=2,
        )
        self.assertEqual(set(opt_config.objective.metric_names), {"foo", "bar"})
        self.assertEqual(len(opt_config.outcome_constraints), 2)
        self.assertEqual(opt_config.objective_thresholds[0].bound, 3.0)
        # Check objective metric names
        objective_metric_names = opt_config.objective.metric_names
        self.assertEqual(len(objective_metric_names), 2)
        self.assertEqual(objective_metric_names[0], "foo")
        self.assertEqual(objective_metric_names[1], "bar")
        # Check constraints
        self.assertEqual(len(opt_config.outcome_constraints), 2)
        constraint_metric_names = [
            c.metric_names[0] for c in opt_config.outcome_constraints
        ]
        self.assertEqual(constraint_metric_names[0], "baz")
        # Check metrics
        self.assertEqual(len(metrics), 4)
        self.assertEqual(metrics[0].name, "foo")
        self.assertEqual(metrics[1].name, "bar")
        self.assertEqual(metrics[2].name, "baz")
        self.assertEqual(metrics[3].name, "pony")

    def test_get_continuous_search_space(self) -> None:
        bounds = [(0.0, 1.0), (2.0, 3.0)]
        with self.subTest("Dummy parameters not specified"):
            search_space = get_continuous_search_space(bounds=bounds)
            self.assertEqual(len(search_space.parameters), 2)
            self.assertEqual(
                len(search_space.parameters), len(search_space.range_parameters)
            )
            self.assertEqual({"x0", "x1"}, search_space.parameters.keys())

        with self.subTest("Dummy parameters specified"):
            search_space = get_continuous_search_space(
                bounds=bounds, n_dummy_dimensions=2
            )
            self.assertEqual(len(search_space.parameters), 4)
            self.assertEqual(
                len(search_space.parameters), len(search_space.range_parameters)
            )
            self.assertEqual(
                {"x0", "x1", "embedding_dummy_0", "embedding_dummy_1"},
                set(search_space.parameters.keys()),
            )
