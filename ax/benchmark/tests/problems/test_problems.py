# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.problems.registry import BENCHMARK_PROBLEM_REGISTRY, get_problem
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_problems(self) -> None:

        # Make sure problem construction succeeds
        for name in BENCHMARK_PROBLEM_REGISTRY.keys():
            if "MNIST" in name:
                continue  # Skip these as they cause the test to take a long time

            get_problem(problem_name=name)

    def test_name(self) -> None:
        expected_names = [
            ("branin", "Branin"),
            ("hartmann3", "Hartmann_3d"),
            ("hartmann6", "Hartmann_6d"),
            ("hartmann30", "Hartmann_30d"),
            ("branin_currin_observed_noise", "BraninCurrin_observed_noise"),
            ("branin_currin30_observed_noise", "BraninCurrin_observed_noise_30d"),
            ("levy4", "Levy_4d"),
        ]
        for registry_key, problem_name in expected_names:
            problem = get_problem(problem_name=registry_key)
            self.assertEqual(problem.name, problem_name)

    def test_no_duplicates(self) -> None:
        keys = [elt for elt in BENCHMARK_PROBLEM_REGISTRY.keys() if "MNIST" not in elt]
        names = {get_problem(problem_name=key).name for key in keys}
        self.assertEqual(len(keys), len(names))
