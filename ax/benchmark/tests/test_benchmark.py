# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark import (
    benchmark_scored_full_run,
    get_sobol_baseline,
    benchmark_replication,
    benchmark_test,
    benchmark_full_run,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    get_sobol_gpei_benchmark_method,
    get_single_objective_benchmark_problem,
    get_multi_objective_benchmark_problem,
    get_sobol_benchmark_method,
)
from ax.utils.testing.mock import fast_botorch_optimize


class TestBenchmark(TestCase):
    def test_replication_synthetic(self):
        method = get_sobol_benchmark_method()
        res = benchmark_replication(
            problem=get_single_objective_benchmark_problem(), method=method
        )

        self.assertEqual(
            method.scheduler_options.total_trials,
            len(res.experiment.trials),
        )

        # Assert optimization trace is monotonic
        for i in range(1, len(res.optimization_trace)):
            self.assertLessEqual(
                res.optimization_trace[i], res.optimization_trace[i - 1]
            )

    def test_replication_moo(self):
        method = get_sobol_benchmark_method()

        res = benchmark_replication(
            problem=get_multi_objective_benchmark_problem(), method=method
        )

        self.assertEqual(
            method.scheduler_options.total_trials,
            len(res.experiment.trials),
        )
        self.assertEqual(
            method.scheduler_options.total_trials * 2,
            len(res.experiment.fetch_data().df),
        )

        # Assert optimization trace is monotonic (hypervolume should always increase)
        for i in range(1, len(res.optimization_trace)):
            self.assertGreaterEqual(
                res.optimization_trace[i], res.optimization_trace[i - 1]
            )

    def test_test(self):
        agg = benchmark_test(
            problem=get_single_objective_benchmark_problem(),
            method=get_sobol_benchmark_method(),
            num_replications=2,
        )

        self.assertEqual(len(agg.experiments), 2)
        self.assertTrue(
            all(len(experiment.trials) == 4 for experiment in agg.experiments),
            "All experiments must have 4 trials",
        )

        # Assert optimization trace is monotonic
        for i in range(1, len(agg.optimization_trace)):
            self.assertLessEqual(
                agg.optimization_trace["mean"][i], agg.optimization_trace["mean"][i - 1]
            )

    @fast_botorch_optimize
    def test_full_run(self):
        aggs = benchmark_full_run(
            problems=[get_single_objective_benchmark_problem()],
            methods=[get_sobol_benchmark_method(), get_sobol_gpei_benchmark_method()],
            num_replications=2,
        )

        self.assertEqual(len(aggs), 2)

        # Assert optimization traces are monotonic
        for agg in aggs:
            for i in range(1, len(agg.optimization_trace)):
                self.assertLessEqual(
                    agg.optimization_trace["mean"][i],
                    agg.optimization_trace["mean"][i - 1],
                )

    @fast_botorch_optimize
    def test_scored(self):
        problem = get_single_objective_benchmark_problem()
        baseline = get_sobol_baseline(
            problem=problem, num_replications=1, total_trials=5
        )

        scored_results = benchmark_scored_full_run(
            problems_baseline_results=[(problem, baseline)],
            methods=[get_sobol_gpei_benchmark_method()],
            num_replications=1,
        )

        self.assertEqual(len(scored_results), 1)
        self.assertTrue(
            (scored_results[0].score < 100).all()
        )  # Score should never be over 100
