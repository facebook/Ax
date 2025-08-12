# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.problems.synthetic.from_botorch import (
    get_augmented_branin_problem,
    get_augmented_branin_search_space,
)
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


class TestBoTorchProblems(TestCase):
    def test_get_augmented_branin_search_space(self) -> None:
        with self.subTest("fidelity"):
            search_space = get_augmented_branin_search_space(
                fidelity_or_task="fidelity"
            )
            param = assert_is_instance(search_space.parameters["x2"], RangeParameter)
            self.assertEqual(param.target_value, 1.0)
            self.assertTrue(param.is_fidelity)

        with self.subTest("task"):
            problem = get_augmented_branin_problem(fidelity_or_task="task")
            param = assert_is_instance(
                problem.search_space.parameters["x2"], ChoiceParameter
            )
            self.assertEqual(param.target_value, 1.0)
            self.assertTrue(param.is_task)
            self.assertFalse(param.is_fidelity)

    def test_get_augmented_branin_problem(self) -> None:
        with self.subTest("inference value as trace"):
            problem = get_augmented_branin_problem(
                fidelity_or_task="fidelity", report_inference_value_as_trace=True
            )
            self.assertTrue(problem.report_inference_value_as_trace)
            self.assertEqual(problem.name, "AugmentedBranin")

        with self.subTest("Do not report inference value as trace"):
            problem = get_augmented_branin_problem(
                fidelity_or_task="fidelity", report_inference_value_as_trace=False
            )
            self.assertFalse(problem.report_inference_value_as_trace)
