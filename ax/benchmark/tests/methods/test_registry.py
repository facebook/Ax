# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.methods.registry import get_benchmark_method
from ax.utils.common.testutils import TestCase


class TestMethodsRegistry(TestCase):
    def test_sobol_method_registry(self) -> None:
        method = get_benchmark_method("Sobol")
        self.assertEqual(method.batch_size, 1)
        self.assertFalse(method.distribute_replications)
        self.assertEqual(method.generation_strategy.name, "Sobol")
        custom_method = get_benchmark_method("Sobol", batch_size=5)
        self.assertEqual(custom_method.batch_size, 5)
