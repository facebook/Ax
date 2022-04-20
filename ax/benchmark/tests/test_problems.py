# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.benchmark.problems.registry import (
    get_problem_and_baseline,
    BENCHMARK_PROBLEM_REGISTRY,
)
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_baselines(self):

        # Make sure the json parsing suceeded
        for name in BENCHMARK_PROBLEM_REGISTRY.keys():
            if "MNIST" in name:
                continue  # Skip these as they cause the test to take a long time

            problem, baseline = get_problem_and_baseline(problem_name=name)

            self.assertTrue(isinstance(baseline, AggregatedBenchmarkResult))
            self.assertIn(problem.name, baseline.name)
