# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark2.benchmark import benchmark_replication, benchmark_test
from ax.benchmark2.benchmark_method import BenchmarkMethod
from ax.benchmark2.benchmark_problem import (
    SingleObjectiveBenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions
from ax.utils.common.testutils import TestCase
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Ackley


class TestBenchmark(TestCase):
    def setUp(self):
        self.ackley = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem=Ackley()
        )
        self.branin_currin = (
            MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem=BraninCurrin()
            )
        )

        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=4,
                )
            ],
            name="SOBOL",
        )
        options = SchedulerOptions(total_trials=4)
        self.sobol4 = BenchmarkMethod(
            name="SOBOL", generation_strategy=gs, scheduler_options=options
        )

    def test_replication_synthetic(self):
        res = benchmark_replication(problem=self.ackley, method=self.sobol4)

        self.assertEqual(
            self.sobol4.scheduler_options.total_trials,
            len(res.experiment.trials),
        )

        # Assert optimization trace is monotonic
        for i in range(1, len(res.optimization_trace)):
            self.assertLessEqual(
                res.optimization_trace[i], res.optimization_trace[i - 1]
            )

    def test_replication_moo(self):
        res = benchmark_replication(problem=self.branin_currin, method=self.sobol4)

        self.assertEqual(
            self.sobol4.scheduler_options.total_trials,
            len(res.experiment.trials),
        )
        self.assertEqual(
            self.sobol4.scheduler_options.total_trials * 2,
            len(res.experiment.fetch_data().df),
        )

        # Assert optimization trace is monotonic (hypervolume should always increase)
        for i in range(1, len(res.optimization_trace)):
            self.assertGreaterEqual(
                res.optimization_trace[i], res.optimization_trace[i - 1]
            )

    def test_test(self):
        agg = benchmark_test(
            problem=self.ackley, method=self.sobol4, num_replications=2
        )

        self.assertEqual(len(agg.experiments), 2)
        self.assertTrue(
            all(len(experiment.trials) == 4 for experiment in agg.experiments),
            "All experiments must have 4 trials",
        )

        # Assert optimization trace is monotonic
        for i in range(1, len(agg.optimization_trace)):
            self.assertLessEqual(
                agg.optimization_trace["mean"][i], agg.optimization_trace["mean"][i - 1]
            )
