# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.problems.synthetic.bandit import get_bandit_problem, get_baseline
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_get_baseline(self) -> None:
        num_choices = 5
        baseline = get_baseline(num_choices=num_choices, n_sims=100)
        self.assertGreater(baseline, 0)
        # Worst = num_choices - 1; random guessing = (num_choices - 1) / 2
        self.assertLess(baseline, (num_choices - 1) / 2)

    def test_get_bandit_problem(self) -> None:
        problem = get_bandit_problem()
        self.assertEqual(problem.name, "Bandit")
        self.assertEqual(problem.num_trials, 3)
        self.assertTrue(problem.report_inference_value_as_trace)

        problem = get_bandit_problem(num_choices=26, num_trials=4)
        self.assertEqual(problem.num_trials, 4)
        self.assertEqual(problem.status_quo_params, {"x0": 26 // 2})

    def test_baseline_exception(self) -> None:
        with self.assertWarnsRegex(
            Warning, expected_regex="Baseline value is not available for num_choices=17"
        ):
            problem = get_bandit_problem(num_choices=17)

        self.assertEqual(problem.baseline_value, get_bandit_problem().baseline_value)
