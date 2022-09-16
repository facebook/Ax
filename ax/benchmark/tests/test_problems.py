# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.problems.registry import BENCHMARK_PROBLEM_REGISTRY, get_problem
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_problems(self) -> None:

        # Make sure problem construction succeeds
        for name in BENCHMARK_PROBLEM_REGISTRY.keys():
            if "MNIST" in name:
                continue  # Skip these as they cause the test to take a long time

            get_problem(problem_name=name)
