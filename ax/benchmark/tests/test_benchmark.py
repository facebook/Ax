#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark import full_benchmark_run
from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import branin
from ax.utils.testing.core_stubs import (
    get_branin_optimization_config,
    get_branin_search_space,
    get_optimization_config,
)


class TestBenchmark(TestCase):
    def test_basic(self):
        """Run through the benchmarking loop."""
        results = full_benchmark_run(
            problems=[
                SimpleBenchmarkProblem(branin, noise_sd=0.4),
                BenchmarkProblem(
                    name="Branin",
                    search_space=get_branin_search_space(),
                    optimization_config=get_branin_optimization_config(),
                ),
                BenchmarkProblem(
                    search_space=get_branin_search_space(),
                    optimization_config=get_optimization_config(),
                ),
            ],
            methods=[
                GenerationStrategy(
                    steps=[GenerationStep(model=Models.SOBOL, num_arms=-1)]
                )
            ],
            num_replications=3,
            num_trials=5,
            # Just to have it be more telling if something is broken
            raise_all_exceptions=True,
            batch_size=[[1], [3], [1]],
        )
        self.assertEqual(len(results["Branin"]["Sobol"]), 3)

    def test_raise_all_exceptions(self):
        """Checks that an exception nested in the benchmarking stack is raised
        when `raise_all_exceptions` is True.
        """

        def broken_benchmark_replication(*args, **kwargs) -> Experiment:
            raise ValueError("Oh, exception!")

        with self.assertRaisesRegex(ValueError, "Oh, exception!"):
            full_benchmark_run(
                problems=[SimpleBenchmarkProblem(branin, noise_sd=0.4)],
                methods=[
                    GenerationStrategy(
                        steps=[GenerationStep(model=Models.SOBOL, num_arms=-1)]
                    )
                ],
                num_replications=3,
                num_trials=5,
                raise_all_exceptions=True,
                benchmark_replication=broken_benchmark_replication,
            )
