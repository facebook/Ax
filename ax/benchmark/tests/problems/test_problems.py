# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.registry import BENCHMARK_PROBLEM_REGISTRY, get_problem
from ax.benchmark.problems.runtime_funcs import int_from_params
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_problems(self) -> None:
        # Make sure problem construction succeeds
        for name in BENCHMARK_PROBLEM_REGISTRY.keys():
            if "MNIST" in name:
                continue  # Skip these as they cause the test to take a long time

            problem = get_problem(problem_key=name)
            self.assertIsInstance(problem, BenchmarkProblem, msg=name)

    def test_name(self) -> None:
        expected_names = [
            ("Bandit", "Bandit"),
            ("branin", "Branin"),
            ("hartmann3", "Hartmann_3d"),
            ("hartmann6", "Hartmann_6d"),
            ("hartmann30", "Hartmann_30d"),
            ("branin_currin_observed_noise", "BraninCurrin_observed_noise"),
            ("branin_currin30_observed_noise", "BraninCurrin_observed_noise_30d"),
            ("levy4", "Levy_4d"),
        ] + [
            (name, name)
            for name in ["Discrete Ackley", "Discrete Hartmann", "Discrete Rosenbrock"]
        ]
        for registry_key, problem_name in expected_names:
            problem = get_problem(problem_key=registry_key)
            self.assertEqual(problem.name, problem_name)

    def test_no_duplicates(self) -> None:
        keys = [elt for elt in BENCHMARK_PROBLEM_REGISTRY.keys() if "MNIST" not in elt]
        names = {get_problem(problem_key=key).name for key in keys}
        self.assertEqual(len(keys), len(names))

    def test_external_registry(self) -> None:
        registry = {
            "test_problem": BENCHMARK_PROBLEM_REGISTRY["branin"],
        }
        problem = get_problem(problem_key="test_problem", registry=registry)
        self.assertEqual(problem.name, "Branin")
        with self.assertRaises(KeyError):
            get_problem(problem_key="branin", registry=registry)

    def test_registry_kwargs_not_mutated(self) -> None:
        entry = BENCHMARK_PROBLEM_REGISTRY["jenatton"]
        expected_kws = {"num_trials": 50, "observe_noise_sd": False}
        self.assertEqual(entry.factory_kwargs, expected_kws)
        problem = get_problem(problem_key="jenatton", num_trials=5)
        self.assertEqual(problem.num_trials, 5)
        self.assertEqual(
            BENCHMARK_PROBLEM_REGISTRY["jenatton"].factory_kwargs, expected_kws
        )
        problem = get_problem(problem_key="jenatton")
        self.assertEqual(problem.num_trials, 50)

    def test_runtime_funcs(self) -> None:
        parameters = {"x0": 0.5, "x1": -3, "x2": "-4", "x3": False, "x4": None}
        result = int_from_params(params=parameters)
        expected = 1
        self.assertEqual(result, expected)
