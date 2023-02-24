# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.benchmark.benchmark import (
    benchmark_full_run,
    benchmark_replication,
    benchmark_test,
)
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_sobol_benchmark_method,
    get_sobol_gpei_benchmark_method,
)
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.test_functions.synthetic import Branin


class TestBenchmark(TestCase):
    def test_replication_synthetic(self) -> None:
        problem = get_single_objective_benchmark_problem()

        res = benchmark_replication(
            problem=problem, method=get_sobol_benchmark_method(), seed=0
        )

        self.assertEqual(
            problem.num_trials,
            len(res.experiment.trials),
        )

        self.assertTrue(np.all(res.score_trace <= 100))

    def test_replication_moo(self) -> None:
        problem = get_multi_objective_benchmark_problem()

        res = benchmark_replication(
            problem=problem, method=get_sobol_benchmark_method(), seed=0
        )

        self.assertEqual(
            problem.num_trials,
            len(res.experiment.trials),
        )
        self.assertEqual(
            problem.num_trials * 2,
            len(res.experiment.fetch_data().df),
        )

        self.assertTrue(np.all(res.score_trace <= 100))

    def test_test(self) -> None:
        problem = get_single_objective_benchmark_problem()
        agg = benchmark_test(
            problem=problem,
            method=get_sobol_benchmark_method(),
            seeds=(0, 1),
        )

        self.assertEqual(len(agg.results), 2)
        self.assertTrue(
            all(
                len(result.experiment.trials) == problem.num_trials
                for result in agg.results
            ),
            "All experiments must have 4 trials",
        )

        for col in ["mean", "P10", "P25", "P50", "P75", "P90"]:
            self.assertTrue((agg.score_trace[col] <= 100).all())

    @fast_botorch_optimize
    def test_full_run(self) -> None:
        aggs = benchmark_full_run(
            problems=[get_single_objective_benchmark_problem()],
            methods=[get_sobol_benchmark_method(), get_sobol_gpei_benchmark_method()],
            seeds=(0, 1),
        )

        self.assertEqual(len(aggs), 2)

        for agg in aggs:
            for col in ["mean", "P10", "P25", "P50", "P75", "P90"]:
                self.assertTrue((agg.score_trace[col] <= 100).all())

    def test_timeout(self) -> None:
        problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem_class=Branin,
            test_problem_kwargs={},
            num_trials=1000,  # Unachievable num_trials
        )

        generation_strategy = GenerationStrategy(
            name="SOBOL+GPEI::default",
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ],
        )

        method = BenchmarkMethod(
            name=generation_strategy.name,
            generation_strategy=generation_strategy,
            scheduler_options=SchedulerOptions(
                max_pending_trials=1,
                init_seconds_between_polls=0,
                min_seconds_before_poll=0,
                timeout_hours=0.001,  # Strict timeout of 3.6 seconds
            ),
        )

        # Each replication will have a different number of trials
        result = benchmark_test(problem=problem, method=method, seeds=(0, 1, 2, 3))

        # Test the traces get composited correctly. The AggregatedResult's traces
        # should be the length of the shortest trace in the BenchmarkResults
        min_num_trials = min(len(res.optimization_trace) for res in result.results)
        self.assertEqual(len(result.optimization_trace), min_num_trials)
        self.assertEqual(len(result.score_trace), min_num_trials)
