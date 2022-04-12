# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark2.benchmark_result import AggregatedBenchmarkResult
from ax.benchmark2.problems.synthetic import get_problem_and_baseline_from_botorch
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_baselines(self):

        # Make sure the json parsing suceeded
        for _, baseline in [
            get_problem_and_baseline_from_botorch("ackley"),
            get_problem_and_baseline_from_botorch("branin"),
        ]:
            self.assertTrue(isinstance(baseline, AggregatedBenchmarkResult))
