# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark2.benchmark import (
    benchmark_replication,
    benchmark_test,
    benchmark_full_run,
)
from ax.benchmark2.benchmark_method import BenchmarkMethod
from ax.benchmark2.benchmark_problem import (
    SingleObjectiveBenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.scheduler import SchedulerOptions
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_modeling
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Branin, Ackley


class TestBenchmark(TestCase):
    def setUp(self):
        self.ackley = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem=Ackley()
        )
        self.branin = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem=Branin()
        )
        self.branin_currin = (
            MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem=BraninCurrin()
            )
        )

        options = SchedulerOptions(total_trials=4, init_seconds_between_polls=0)

        self.sobol4 = BenchmarkMethod(
            name="SOBOL",
            generation_strategy=GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
                name="SOBOL",
            ),
            scheduler_options=options,
        )

        self.mbo_sobol_gpei = BenchmarkMethod(
            name="MBO_SOBOL_GPEI",
            generation_strategy=GenerationStrategy(
                name="Modular::Sobol+GPEI",
                steps=[
                    GenerationStep(
                        model=Models.SOBOL, num_trials=3, min_trials_observed=3
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        model_kwargs={
                            "surrogate": Surrogate(FixedNoiseGP),
                            "botorch_acqf_class": qNoisyExpectedImprovement,
                        },
                        model_gen_kwargs={
                            "model_gen_options": {
                                Keys.OPTIMIZER_KWARGS: {
                                    "num_restarts": 50,
                                    "raw_samples": 1024,
                                },
                                Keys.ACQF_KWARGS: {
                                    "prune_baseline": True,
                                    "qmc": True,
                                    "mc_samples": 512,
                                },
                            }
                        },
                    ),
                ],
            ),
            scheduler_options=options,
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

    @fast_modeling
    def test_full_run(self):
        aggs = benchmark_full_run(
            problems=[self.ackley],
            methods=[self.sobol4, self.mbo_sobol_gpei],
            num_replications=2,
        )

        self.assertEqual(len(aggs), 2)

        # Assert optimization traces are monotonic
        for agg in aggs:
            for i in range(1, len(agg.optimization_trace)):
                self.assertLessEqual(
                    agg.optimization_trace["mean"][i],
                    agg.optimization_trace["mean"][i - 1],
                )
