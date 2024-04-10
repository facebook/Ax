# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_benchmark_scheduler_options,
)
from ax.modelbridge.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none


class TestBenchmarkMethod(TestCase):
    def test_benchmark_method(self) -> None:
        gs = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=10)],
            name="SOBOL",
        )
        options = get_benchmark_scheduler_options()
        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=gs, scheduler_options=options
        )

        # test that `fit_tracking_metrics` has been correctly set to False
        for step in method.generation_strategy._steps:
            self.assertFalse(not_none(step.model_kwargs).get("fit_tracking_metrics"))

        self.assertEqual(method.scheduler_options, options)
        self.assertEqual(options.max_pending_trials, 1)
        self.assertEqual(options.init_seconds_between_polls, 0)
        self.assertEqual(options.min_seconds_before_poll, 0)
        self.assertEqual(options.timeout_hours, 4)

        options = get_benchmark_scheduler_options(timeout_hours=10)
        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=gs, scheduler_options=options
        )
        self.assertEqual(method.scheduler_options.timeout_hours, 10)

        # test that instantiation works with node-based strategies
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL, model_kwargs={}, model_gen_kwargs={}
        )
        node_gs = GenerationStrategy(
            nodes=[GenerationNode(node_name="sobol", model_specs=[sobol_model_spec])]
        )

        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=node_gs, scheduler_options=options
        )
        for node in method.generation_strategy._nodes:
            self.assertFalse(
                not_none(node.model_spec_to_gen_from.model_kwargs).get(
                    "fit_tracking_metrics"
                )
            )
