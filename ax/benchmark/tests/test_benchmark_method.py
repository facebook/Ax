# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.methods.sobol import get_sobol_generation_strategy
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class TestBenchmarkMethod(TestCase):
    def test_benchmark_method(self) -> None:
        gs = get_sobol_generation_strategy()
        method = BenchmarkMethod(name="Sobol10", generation_strategy=gs)
        self.assertEqual(method.name, "Sobol10")

        # test that `fit_tracking_metrics` has been correctly set to False
        for step in method.generation_strategy._steps:
            self.assertFalse(
                none_throws(step.generator_kwargs).get("fit_tracking_metrics")
            )

        method = BenchmarkMethod(generation_strategy=gs)
        self.assertEqual(method.name, method.generation_strategy.name)

        # test that instantiation works with node-based strategies
        method = BenchmarkMethod(name="Sobol10", generation_strategy=gs)
        for node in method.generation_strategy._nodes:
            self.assertFalse(
                none_throws(node.generator_spec_to_gen_from.generator_kwargs).get(
                    "fit_tracking_metrics"
                )
            )
