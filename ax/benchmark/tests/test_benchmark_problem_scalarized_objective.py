# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem, get_continuous_search_space
from ax.benchmark.testing.benchmark_stubs import get_multi_objective_benchmark_problem
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from pyre_extensions import assert_is_instance, none_throws


class TestBenchmarkProblemWithScalarizedObjective(TestCase):
    def setUp(self) -> None:
        """Setup common test fixtures for BenchmarkProblem with ScalarizedObjective."""
        self.search_space = get_continuous_search_space(bounds=[(0.0, 1.0), (0.0, 1.0)])
        self.base_problem = get_multi_objective_benchmark_problem(
            observe_noise_sd=False,
            num_trials=4,
            test_problem_class=BraninCurrin,
        )
        self.test_function = self.base_problem.test_function
        self.metrics = list(self.base_problem.optimization_config.metrics.values())
        self.weights = [0.5, 0.5]
        self.optimization_config = OptimizationConfig(
            objective=ScalarizedObjective(
                metrics=self.metrics,
                weights=self.weights,
                minimize=True,
            )
        )

    def test_benchmark_problem_with_scalarized_objective_basic_structure(
        self,
    ) -> None:
        # Setup: Create BenchmarkProblem with ScalarizedObjective
        problem = BenchmarkProblem(
            name="test_scalarized_objective",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify problem structure
        self.assertEqual(problem.name, "test_scalarized_objective")
        self.assertEqual(problem.num_trials, 10)
        self.assertEqual(problem.optimal_value, 0.0)
        self.assertEqual(problem.baseline_value, 100.0)
        self.assertIsInstance(
            problem.optimization_config.objective, ScalarizedObjective
        )

    def test_benchmark_problem_scalarized_objective_has_correct_metrics(
        self,
    ) -> None:
        # Setup: Create BenchmarkProblem with ScalarizedObjective
        problem = BenchmarkProblem(
            name="test_scalarized_metrics",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify metrics are correctly configured
        objective = problem.optimization_config.objective
        scalarized_obj = assert_is_instance(objective, ScalarizedObjective)
        self.assertEqual(len(scalarized_obj.metrics), len(self.metrics))

        # Verify each metric
        for metric, expected_metric in zip(scalarized_obj.metrics, self.metrics):
            self.assertIsInstance(metric, BenchmarkMetric)
            self.assertEqual(metric.name, expected_metric.name)
            self.assertEqual(metric.lower_is_better, expected_metric.lower_is_better)

    def test_benchmark_problem_scalarized_objective_has_correct_weights(
        self,
    ) -> None:
        # Setup: Create BenchmarkProblem with ScalarizedObjective
        problem = BenchmarkProblem(
            name="test_scalarized_weights",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify weights are correctly configured
        objective = problem.optimization_config.objective
        scalarized_obj = assert_is_instance(objective, ScalarizedObjective)
        self.assertEqual(len(scalarized_obj.weights), len(self.weights))

        for weight, expected_weight in zip(scalarized_obj.weights, self.weights):
            self.assertEqual(weight, expected_weight)

    def test_benchmark_problem_scalarized_objective_minimize_flag(self) -> None:
        # Setup: Create problems with minimize=True and minimize=False
        for minimize in (True, False):
            baseline_value = 100.0 * minimize
            for m in self.metrics:
                m.lower_is_better = minimize
            problem = BenchmarkProblem(
                name="test_minimize_false",
                search_space=self.search_space,
                optimization_config=OptimizationConfig(
                    objective=ScalarizedObjective(
                        metrics=self.metrics,
                        weights=self.weights,
                        minimize=minimize,
                    )
                ),
                test_function=self.test_function,
                num_trials=10,
                optimal_value=100 - baseline_value,
                baseline_value=baseline_value,
            )

            # Assert: Verify minimize flag
            objective = problem.optimization_config.objective
            scalarized_obj = assert_is_instance(objective, ScalarizedObjective)
            self.assertEqual(scalarized_obj.minimize, minimize)

    def test_benchmark_problem_scalarized_objective_with_different_weights(
        self,
    ) -> None:
        # Setup: Create ScalarizedObjective with unequal weights
        unequal_weights = [0.7, 0.3]
        problem = BenchmarkProblem(
            name="test_unequal_weights",
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=ScalarizedObjective(
                    metrics=self.metrics,
                    weights=unequal_weights,
                    minimize=True,
                )
            ),
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify weights are correctly set
        objective = problem.optimization_config.objective
        scalarized_obj = assert_is_instance(objective, ScalarizedObjective)
        self.assertEqual(scalarized_obj.weights, unequal_weights)

    def test_benchmark_problem_scalarized_objective_with_observe_noise_sd(
        self,
    ) -> None:
        # Setup: Create metrics with observe_noise_sd=True
        metrics_with_noise = [
            BenchmarkMetric(
                name="BraninCurrin_0", lower_is_better=True, observe_noise_sd=True
            ),
            BenchmarkMetric(
                name="BraninCurrin_1", lower_is_better=True, observe_noise_sd=True
            ),
        ]

        problem = BenchmarkProblem(
            name="test_observe_noise",
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=ScalarizedObjective(
                    # pyre-fixme [6]: Incompatible parameter type [6]: In call
                    # `ScalarizedObjective.__init__`, for argument `metrics`,
                    # expected `List[Metric]` but got `List[BenchmarkMetric]`.
                    metrics=metrics_with_noise,
                    weights=[0.5, 0.5],
                    minimize=True,
                )
            ),
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify observe_noise_sd is set correctly for all metrics
        objective = problem.optimization_config.objective
        scalarized_obj = assert_is_instance(objective, ScalarizedObjective)

        for metric in scalarized_obj.metrics:
            self.assertIsInstance(metric, BenchmarkMetric)
            self.assertTrue(metric.observe_noise_sd)  # pyre-ignore[16]

    def test_benchmark_problem_scalarized_objective_with_mixed_lower_is_better(
        self,
    ) -> None:
        # Setup: Create metrics with mixed lower_is_better values
        mixed_metrics = [
            BenchmarkMetric(
                name="BraninCurrin_0", lower_is_better=True, observe_noise_sd=False
            ),
            BenchmarkMetric(
                name="BraninCurrin_1", lower_is_better=False, observe_noise_sd=False
            ),
        ]

        problem = BenchmarkProblem(
            name="test_mixed_directions",
            search_space=self.search_space,
            optimization_config=OptimizationConfig(
                objective=ScalarizedObjective(
                    # pyre-fixme[6]: Incompatible parameter type [6]: In call
                    # `ScalarizedObjective.__init__`, for argument `metrics`,
                    # expected `List[Metric]` but got `List[BenchmarkMetric]`.
                    metrics=mixed_metrics,
                    weights=[0.6, -0.4],
                    minimize=True,
                )
            ),
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
        )

        # Assert: Verify lower_is_better is set correctly for each metric
        objective = problem.optimization_config.objective
        scalarized_obj = assert_is_instance(objective, ScalarizedObjective)

        self.assertTrue(scalarized_obj.metrics[0].lower_is_better)
        self.assertFalse(scalarized_obj.metrics[1].lower_is_better)

    def test_benchmark_problem_scalarized_objective_with_noise_std(self) -> None:
        # Setup: Create problem with noise_std
        noise_std = 0.1
        problem = BenchmarkProblem(
            name="test_with_noise_std",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
            noise_std=noise_std,
        )

        # Assert: Verify noise_std is set correctly
        self.assertEqual(problem.noise_std, noise_std)

    def test_benchmark_problem_scalarized_objective_with_dict_noise_std(self) -> None:
        # Setup: Create problem with per-metric noise_std
        noise_std_dict = {
            "branin_metric_1": 0.1,
            "branin_metric_2": 0.2,
        }
        problem = BenchmarkProblem(
            name="test_with_dict_noise_std",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
            noise_std=noise_std_dict,
        )

        # Assert: Verify noise_std dict is set correctly
        self.assertEqual(problem.noise_std, noise_std_dict)

    def test_benchmark_problem_scalarized_objective_with_tracking_metrics(
        self,
    ) -> None:
        # Setup: Create problem with tracking metrics
        tracking_metric = BenchmarkMetric(
            name="branin",
            lower_is_better=True,
            observe_noise_sd=False,
        )

        problem = BenchmarkProblem(
            name="test_tracking_metrics",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            test_function=self.test_function,
            num_trials=10,
            optimal_value=0.0,
            baseline_value=100.0,
            tracking_metrics=[tracking_metric],
        )

        # Assert: Verify tracking metrics are set
        tracking_metrics = none_throws(problem.tracking_metrics)
        self.assertEqual(len(tracking_metrics), 1)
        self.assertEqual(tracking_metrics[0].name, "branin")
