# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from random import choice
from unittest.mock import patch

from ax.benchmark.benchmark_metric import BenchmarkMetric

from ax.benchmark.benchmark_problem import BenchmarkProblem

from ax.benchmark.problems.hpo.torchvision import (
    CNN,
    PyTorchCNNTorchvisionBenchmarkTestFunction,
)
from ax.benchmark.problems.registry import get_problem
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import TestDataset
from pyre_extensions import assert_is_instance


class TestPyTorchCNNTorchvision(TestCase):
    def setUp(self) -> None:
        self.parameters = {
            "lr": 1e-1,
            "momentum": 0.5,
            "weight_decay": 0.5,
            "step_size": 10,
            "gamma": 0.5,
        }
        super().setUp()

    def test_problem_properties(self) -> None:
        num_trials = 173

        with patch.dict(
            "ax.benchmark.problems.hpo.torchvision._REGISTRY",
            {"MNIST": TestDataset, "FashionMNIST": TestDataset},
        ):
            self.assertEqual(
                get_problem(problem_key="hpo_pytorch_cnn_MNIST").name,
                "HPO_PyTorchCNN_Torchvision::MNIST",
            )
            problem = get_problem(
                problem_key="hpo_pytorch_cnn_FashionMNIST", num_trials=num_trials
            )

        self.assertEqual(problem.name, "HPO_PyTorchCNN_Torchvision::FashionMNIST")
        self.assertIsInstance(problem, BenchmarkProblem)
        self.assertEqual(problem.optimal_value, 1.0)
        self.assertSetEqual(
            set(problem.search_space.parameters.keys()),
            {"lr", "momentum", "weight_decay", "step_size", "gamma"},
        )
        self.assertFalse(problem.optimization_config.objective.minimize)
        self.assertEqual(problem.num_trials, num_trials)
        metric = assert_is_instance(
            problem.optimization_config.objective.metric, BenchmarkMetric
        )
        self.assertFalse(metric.observe_noise_sd)

    def test_deterministic(self) -> None:
        problem_name = choice(["MNIST", "FashionMNIST"])
        with patch.dict(
            "ax.benchmark.problems.hpo.torchvision._REGISTRY",
            {problem_name: TestDataset},
        ):
            problem = get_problem(problem_key=f"hpo_pytorch_cnn_{problem_name}")

        test_function = problem.test_function

        self.assertEqual(test_function.outcome_names, ["accuracy"])
        expected = 0.21875
        result = test_function.evaluate_true(params=self.parameters)
        self.assertEqual(result.item(), expected)
        self.assertEqual(problem.noise_std, 0.0)

        with self.subTest("test caching"):
            with patch(
                "ax.benchmark.problems.hpo.torchvision.CNN",
                wraps=CNN,
            ) as mock_CNN:
                test_function.evaluate_true(params=self.parameters)
            mock_CNN.assert_not_called()

            other_params = {**self.parameters, "lr": 0.9}
            with patch(
                "ax.benchmark.problems.hpo.torchvision.CNN", wraps=CNN
            ) as mock_CNN:
                test_function.evaluate_true(params=other_params)
            mock_CNN.assert_called_once()

    def test_load_wrong_name(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Unrecognized torchvision dataset 'pencil'"
        ):
            PyTorchCNNTorchvisionBenchmarkTestFunction(name="pencil")
