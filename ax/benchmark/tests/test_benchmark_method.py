# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.early_stopping.strategies.threshold import ThresholdEarlyStoppingStrategy
from ax.modelbridge.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class TestBenchmarkMethod(TestCase):
    def setUp(self) -> None:
        super().setUp()
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL, model_kwargs={}, model_gen_kwargs={}
        )
        self.gs = GenerationStrategy(
            nodes=[GenerationNode(node_name="sobol", model_specs=[sobol_model_spec])]
        )

    def test_benchmark_method(self) -> None:
        gs = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=10)],
            name="SOBOL",
        )
        method = BenchmarkMethod(name="Sobol10", generation_strategy=gs)

        # test that `fit_tracking_metrics` has been correctly set to False
        for step in method.generation_strategy._steps:
            self.assertFalse(none_throws(step.model_kwargs).get("fit_tracking_metrics"))

        options = method.scheduler_options
        self.assertEqual(options.max_pending_trials, 1)
        self.assertEqual(options.init_seconds_between_polls, 0)
        self.assertEqual(options.min_seconds_before_poll, 0)
        self.assertEqual(method.timeout_hours, 4)

        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=gs, timeout_hours=10
        )
        self.assertEqual(method.timeout_hours, 10)

        # test that instantiation works with node-based strategies
        method = BenchmarkMethod(name="Sobol10", generation_strategy=self.gs)
        for node in method.generation_strategy._nodes:
            self.assertFalse(
                none_throws(node.model_spec_to_gen_from.model_kwargs).get(
                    "fit_tracking_metrics"
                )
            )

    def test_raises_when_ess_polls_with_delay(self) -> None:
        ess = ThresholdEarlyStoppingStrategy(seconds_between_polls=10)
        with self.assertWarnsRegex(Warning, "seconds_between_polls"):
            BenchmarkMethod(
                generation_strategy=self.gs,
                early_stopping_strategy=ess,
            )
