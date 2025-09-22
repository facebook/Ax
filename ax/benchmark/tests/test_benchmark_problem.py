# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    get_continuous_search_space,
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
from ax.utils.common.testutils import TestCase
from botorch.test_functions.synthetic import Branin
from pyre_extensions import assert_is_instance


class TestBenchmarkProblem(TestCase):
    def setUp(self) -> None:
        # Print full output, so that any differences in 'repr' output are shown
        self.maxDiff = None
        super().setUp()

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
        self.assertEqual(objective_metric.signature, "foo")
        self.assertEqual(objective_metric.observe_noise_sd, True)
        self.assertEqual(objective_metric.lower_is_better, False)
        constraint_metric = assert_is_instance(
            opt_config.outcome_constraints[0].metric, BenchmarkMetric
        )
        self.assertEqual(constraint_metric.name, "bar")
        self.assertEqual(constraint_metric.signature, "bar")
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
