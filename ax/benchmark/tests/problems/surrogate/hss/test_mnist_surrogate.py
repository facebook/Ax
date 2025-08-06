# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.problems.surrogate.hss.mnist_surrogate import (
    get_mnist_surrogate_benchmark,
)
from ax.utils.common.testutils import TestCase


class TestMNISTSurrogate(TestCase):
    def test_mnist_surrogate(self) -> None:
        cases = [
            (
                {"lr": 0.001, "use_dropout": False, "use_weight_decay": False},
                0.9820157289505005,
            ),
            (
                {
                    "lr": 0.01,
                    "use_dropout": False,
                    "use_weight_decay": True,
                    "weight_decay": 0.01,
                },
                0.9627381563186646,
            ),
            (
                {
                    "lr": 0.1,
                    "use_dropout": True,
                    "dropout": 0.1,
                    "use_weight_decay": False,
                },
                0.103008933365345,
            ),
            (
                {
                    "lr": 0.01,
                    "use_dropout": True,
                    "dropout": 0.5,
                    "use_weight_decay": True,
                    "weight_decay": 0.01,
                },
                0.9476024508476257,
            ),
        ]

        benchmark = get_mnist_surrogate_benchmark(num_trials=1)

        for params, target_value in cases:
            self.assertAlmostEqual(
                benchmark.test_function.evaluate_true(params).item(),
                target_value,
            )

    def test_benchmark_creation(self) -> None:
        benchmark = get_mnist_surrogate_benchmark(num_trials=1)

        objective = benchmark.optimization_config.objective
        metric = objective.metric

        self.assertEqual(metric.name, "MNIST Test Accuracy")
        self.assertFalse(objective.minimize)
        self.assertFalse(metric.lower_is_better)
