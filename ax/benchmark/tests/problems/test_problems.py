# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest.mock import patch

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.registry import (
    BENCHMARK_PROBLEM_REGISTRY,
    get_benchmark_problem,
)
from ax.benchmark.problems.runtime_funcs import int_from_params
from ax.benchmark.testing.benchmark_stubs import get_mock_lcbench_data
from ax.utils.common.testutils import TestCase
from sklearn.pipeline import Pipeline


class TestProblems(TestCase):
    def test_load_problems(self) -> None:
        # Make sure problem construction succeeds
        for name in BENCHMARK_PROBLEM_REGISTRY.keys():
            if "MNIST" in name:
                continue  # Skip these as they cause the test to take a long time
            if name == "botorch_synthetic":
                # Skip this generic one since it requires the mandatory argument
                # `test_problem_class`. This is tested separately below.
                continue

            # Avoid downloading data from the internet
            with patch(
                "ax.benchmark.problems.surrogate."
                "lcbench.early_stopping.load_lcbench_data",
                return_value=get_mock_lcbench_data(),
            ), patch.object(Pipeline, "fit"):
                problem = get_benchmark_problem(problem_key=name)
            self.assertIsInstance(problem, BenchmarkProblem, msg=name)

        generic_synthetic_problem = get_benchmark_problem(
            problem_key="botorch_synthetic", test_problem_class="Branin"
        )
        self.assertIsInstance(
            generic_synthetic_problem, BenchmarkProblem, msg="botorch_synthetic"
        )

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
            problem = get_benchmark_problem(problem_key=registry_key)
            self.assertEqual(problem.name, problem_name)

    def test_no_duplicates(self) -> None:
        problem_names = set()
        problem_keys = []
        # Avoid downloading data from the internet
        with patch(
            "ax.benchmark.problems.surrogate."
            "lcbench.early_stopping.load_lcbench_data",
            return_value=get_mock_lcbench_data(),
        ), patch.object(Pipeline, "fit"):
            for problem_key in BENCHMARK_PROBLEM_REGISTRY.keys():
                if "MNIST" in problem_key:
                    continue  # Skip these as they cause the test to take a long time
                if problem_key == "botorch_synthetic":
                    # Skip this generic one since it requires the mandatory argument
                    # `test_problem_class`. This is added separately below.
                    continue

                problem_keys.append(problem_key)
                problem_names.add(get_benchmark_problem(problem_key=problem_key).name)

        # Handle the botorch_synthetic special case separately. Note that StyblinskiTang
        # works here only because it is not registered as its own benchmark problem
        # in the registry.
        problem_keys.append("botorch_synthetic")
        problem_names.add(
            get_benchmark_problem(
                problem_key="botorch_synthetic",
                test_problem_class="StyblinskiTang",
                test_problem_kwargs={"dim": 2},
            ).name
        )

        self.assertEqual(len(problem_names), len(problem_keys))

    def test_external_registry(self) -> None:
        registry = {
            "test_problem": BENCHMARK_PROBLEM_REGISTRY["branin"],
        }
        problem = get_benchmark_problem(problem_key="test_problem", registry=registry)
        self.assertEqual(problem.name, "Branin")
        with self.assertRaises(KeyError):
            get_benchmark_problem(problem_key="branin", registry=registry)

    def test_registry_kwargs_not_mutated(self) -> None:
        entry = BENCHMARK_PROBLEM_REGISTRY["jenatton"]
        expected_kws = {"num_trials": 50, "observe_noise_sd": False}
        self.assertEqual(entry.factory_kwargs, expected_kws)
        problem = get_benchmark_problem(problem_key="jenatton", num_trials=5)
        self.assertEqual(problem.num_trials, 5)
        self.assertEqual(
            BENCHMARK_PROBLEM_REGISTRY["jenatton"].factory_kwargs, expected_kws
        )
        problem = get_benchmark_problem(problem_key="jenatton")
        self.assertEqual(problem.num_trials, 50)

    def test_runtime_funcs(self) -> None:
        parameters = {"x0": 0.5, "x1": -3, "x2": "-4", "x3": False, "x4": None}
        result = int_from_params(params=parameters)
        expected = 1
        self.assertEqual(result, expected)
