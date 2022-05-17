# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.scored_benchmark import (
    scored_benchmark_full_run,
    scored_benchmark_replication,
    scored_benchmark_test,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    get_aggregated_benchmark_result,
    get_single_objective_benchmark_problem,
    get_sobol_gpei_benchmark_method,
)
from ax.utils.testing.mock import fast_botorch_optimize


class TestProblems(TestCase):
    @fast_botorch_optimize
    def test_scored_benchmark_replication(self) -> None:
        scored_result = scored_benchmark_replication(
            problem=get_single_objective_benchmark_problem(),
            method=get_sobol_gpei_benchmark_method(),
        )

        self.assertEqual(len(scored_result.score_trace), 4)
        self.assertTrue(
            (scored_result.score_trace < 100).all()
        )  # Score should never be over 100

    @fast_botorch_optimize
    def test_scored_benchmark_test(self) -> None:
        aggregated_scored_result = scored_benchmark_test(
            problem=get_single_objective_benchmark_problem(),
            method=get_sobol_gpei_benchmark_method(),
            num_replications=2,
        )

        self.assertEqual(len(aggregated_scored_result.score_trace), 4)
        self.assertTrue(
            (aggregated_scored_result.score_trace["mean"] < 100).all()
        )  # Score should never be over 100
        self.assertTrue(
            (aggregated_scored_result.score_trace["median"] < 100).all()
        )  # Score should never be over 100

    @fast_botorch_optimize
    def test_scored_benchmark_full_run(self) -> None:
        aggregated_scored_results = scored_benchmark_full_run(
            problems_baseline_results=[
                (
                    get_single_objective_benchmark_problem(),
                    get_aggregated_benchmark_result(),
                )
            ],
            methods=[get_sobol_gpei_benchmark_method()],
            num_replications=2,
        )

        self.assertEqual(len(aggregated_scored_results), 1)
