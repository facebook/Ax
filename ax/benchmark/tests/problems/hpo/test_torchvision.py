# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from random import choice
from unittest.mock import MagicMock, patch

from ax.benchmark.benchmark_problem import BenchmarkProblem

from ax.benchmark.problems.hpo.torchvision import CNN
from ax.benchmark.problems.registry import get_problem
from ax.core.arm import Arm
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import TestDataset


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
                get_problem(problem_name="hpo_pytorch_cnn_MNIST").name,
                "HPO_PyTorchCNN_Torchvision::MNIST",
            )
            problem = get_problem(
                problem_name="hpo_pytorch_cnn_FashionMNIST", num_trials=num_trials
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
        self.assertFalse(problem.observe_noise_stds)

    def test_deterministic(self) -> None:
        problem_name = choice(["MNIST", "FashionMNIST"])
        with patch.dict(
            "ax.benchmark.problems.hpo.torchvision._REGISTRY",
            {problem_name: TestDataset},
        ):
            problem = get_problem(problem_name=f"hpo_pytorch_cnn_{problem_name}")
        # pyre-fixme[6]: complaining that the annotation for Arm.parameters is
        # too broad because it's not immutable
        arm = Arm(parameters=self.parameters, name="0")
        trial = Trial(experiment=MagicMock()).add_arm(arm=arm)

        result = problem.runner.run(trial=trial)
        expected = 0.21875
        self.assertEqual(
            result,
            {
                "Ys": {"0": [expected]},
                "Ystds": {"0": [0.0]},
                "outcome_names": ["accuracy"],
            },
        )

        with self.subTest("test caching"):
            with patch(
                "ax.benchmark.problems.hpo.torchvision.CNN",
                wraps=CNN,
            ) as mock_CNN:
                problem.runner.run(trial=trial)
            mock_CNN.assert_not_called()

            other_trial = Trial(experiment=MagicMock()).add_arm(
                arm=Arm(parameters={**self.parameters, "lr": 0.9}, name="1")
            )
            with patch(
                "ax.benchmark.problems.hpo.torchvision.CNN", wraps=CNN
            ) as mock_CNN:
                problem.runner.run(trial=other_trial)
            mock_CNN.assert_called_once()
