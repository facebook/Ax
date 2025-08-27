# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.benchmark.problems.surrogate.hss.cifar10_surrogate import (
    get_cifar10_surrogate_benchmark,
)
from ax.utils.common.testutils import TestCase


class TestCIFAR10Surrogate(TestCase):
    def test_benchmark_creation(self) -> None:
        benchmark = get_cifar10_surrogate_benchmark(num_trials=1)
        objective = benchmark.optimization_config.objective

        self.assertFalse(objective.minimize)
        self.assertFalse(objective.metric.lower_is_better)
        self.assertEqual(objective.metric.name, "CIFAR10 Test Accuracy")

    def test_cifar10_surrogate(self) -> None:
        cases = [
            (
                {
                    "lr": 0.001,
                    "two-more-blocks": False,
                    "use_softplus_activation": False,
                    "use_weight_decay": False,
                },
                0.7585159540176392,
            ),
            (
                {
                    "lr": 0.01,
                    "two-more-blocks": False,
                    "use_softplus_activation": False,
                    "use_weight_decay": True,
                    "weight_decay": 0.0001,
                },
                0.7298734784126282,
            ),
            (
                {
                    "lr": 1e-05,
                    "two-more-blocks": True,
                    "use_softplus_activation": True,
                    "softplus_beta": 2.0,
                    "use_weight_decay": False,
                },
                0.713783860206604,
            ),
            (
                {
                    "lr": 0.0001,
                    "two-more-blocks": True,
                    "use_softplus_activation": True,
                    "softplus_beta": 2.0,
                    "use_weight_decay": True,
                    "weight_decay": 1e-08,
                },
                0.7651293277740479,
            ),
        ]

        benchmark = get_cifar10_surrogate_benchmark(num_trials=1)

        for params, target_value in cases:
            self.assertAlmostEqual(
                benchmark.test_function.evaluate_true(params).item(),
                target_value,
            )
