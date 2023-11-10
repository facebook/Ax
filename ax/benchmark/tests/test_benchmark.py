# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import numpy as np
from ax.benchmark.benchmark import (
    benchmark_multiple_problems_methods,
    benchmark_one_method_problem,
    benchmark_replication,
)
from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_sequential_optimization_scheduler_options,
)
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.benchmark_stubs import (
    get_moo_surrogate,
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_sobol_benchmark_method,
    get_soo_surrogate,
)
from ax.utils.testing.core_stubs import get_dataset, get_experiment
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.test_functions.synthetic import Branin


class TestBenchmark(TestCase):
    def test_storage(self) -> None:
        problem = get_single_objective_benchmark_problem()
        res = benchmark_replication(
            problem=problem, method=get_sobol_benchmark_method(), seed=0
        )
        # Experiment is not in storage yet
        self.assertTrue(res.experiment is not None)
        self.assertEqual(res.experiment_storage_id, None)
        experiment = res.experiment

        # test saving to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".json") as f:
            save_experiment(not_none(res.experiment), f.name)
            res.experiment_storage_id = f.name
            res.experiment = None
            self.assertIsNone(res.experiment)
            self.assertEqual(res.experiment_storage_id, f.name)

            # load it back
            experiment = load_experiment(f.name)
            self.assertEqual(experiment, experiment)

    def test_benchmark_result_invalid_inputs(self) -> None:
        """
        Test that a BenchmarkResult cannot be specified with both an `experiment`
        and an `experiment_storage_id`.
        """
        with self.assertRaisesRegex(ValueError, "Cannot specify both an `experiment` "):
            BenchmarkResult(
                name="name",
                seed=0,
                optimization_trace=np.array([]),
                score_trace=np.array([]),
                fit_time=0.0,
                gen_time=0.0,
                experiment=get_experiment(),
                experiment_storage_id="experiment_storage_id",
            )

        with self.assertRaisesRegex(
            ValueError, "Must provide an `experiment` or `experiment_storage_id`"
        ):
            BenchmarkResult(
                name="name",
                seed=0,
                optimization_trace=np.array([]),
                score_trace=np.array([]),
                fit_time=0.0,
                gen_time=0.0,
            )

    def test_replication_sobol_synthetic(self) -> None:
        method = get_sobol_benchmark_method()
        problem = get_single_objective_benchmark_problem()
        res = benchmark_replication(problem=problem, method=method, seed=0)

        self.assertEqual(
            min(problem.num_trials, not_none(method.scheduler_options.total_trials)),
            len(not_none(res.experiment).trials),
        )

        self.assertTrue(np.isfinite(res.score_trace).all())
        self.assertTrue(np.all(res.score_trace <= 100))

    def test_replication_sobol_surrogate(self) -> None:
        method = get_sobol_benchmark_method()

        for name, problem in [
            ("soo", get_soo_surrogate()),
            ("moo", get_moo_surrogate()),
        ]:

            with self.subTest(name, problem=problem):
                surrogate, datasets = not_none(problem.get_surrogate_and_datasets)()
                datasets = [get_dataset()]
                surrogate.fit(
                    datasets,
                    metric_names=datasets[0].outcome_names,
                    search_space_digest=extract_search_space_digest(
                        problem.search_space,
                        param_names=[*problem.search_space.parameters.keys()],
                    ),
                )
                res = benchmark_replication(problem=problem, method=method, seed=0)

                self.assertEqual(
                    min(
                        problem.num_trials,
                        not_none(method.scheduler_options.total_trials),
                    ),
                    len(not_none(res.experiment).trials),
                )

                self.assertTrue(np.isfinite(res.score_trace).all())
                self.assertTrue(np.all(res.score_trace <= 100))

    @fast_botorch_optimize
    def test_replication_mbm(self) -> None:
        for method, problem, expected_name in [
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    scheduler_options=get_sequential_optimization_scheduler_options(),
                ),
                get_single_objective_benchmark_problem(infer_noise=False, num_trials=6),
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=FixedNoiseGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                ),
                get_single_objective_benchmark_problem(infer_noise=False, num_trials=6),
                "MBM::FixedNoiseGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=FixedNoiseGP,
                    acquisition_cls=qNoisyExpectedHypervolumeImprovement,
                ),
                get_multi_objective_benchmark_problem(infer_noise=False, num_trials=6),
                "MBM::FixedNoiseGP_qNEHVI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SaasFullyBayesianSingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                ),
                get_multi_objective_benchmark_problem(num_trials=6),
                "MBM::SAAS_qLogNEI",
            ),
        ]:
            with self.subTest(method=method, problem=problem):
                res = benchmark_replication(problem=problem, method=method, seed=0)
                self.assertEqual(
                    problem.num_trials,
                    len(not_none(res.experiment).trials),
                )
                self.assertTrue(np.all(res.score_trace <= 100))
                self.assertEqual(method.name, method.generation_strategy.name)
                self.assertEqual(method.name, expected_name)

    def test_replication_moo_sobol(self) -> None:
        problem = get_multi_objective_benchmark_problem()

        res = benchmark_replication(
            problem=problem, method=get_sobol_benchmark_method(), seed=0
        )

        self.assertEqual(
            problem.num_trials,
            len(not_none(res.experiment).trials),
        )
        self.assertEqual(
            problem.num_trials * 2,
            len(not_none(res.experiment).fetch_data().df),
        )

        self.assertTrue(np.all(res.score_trace <= 100))

    def test_benchmark_one_method_problem(self) -> None:
        problem = get_single_objective_benchmark_problem()
        agg = benchmark_one_method_problem(
            problem=problem,
            method=get_sobol_benchmark_method(),
            seeds=(0, 1),
        )

        self.assertEqual(len(agg.results), 2)
        self.assertTrue(
            all(
                len(not_none(result.experiment).trials) == problem.num_trials
                for result in agg.results
            ),
            "All experiments must have 4 trials",
        )

        for col in ["mean", "P25", "P50", "P75"]:
            self.assertTrue((agg.score_trace[col] <= 100).all())

    @fast_botorch_optimize
    def test_benchmark_multiple_problems_methods(self) -> None:
        aggs = benchmark_multiple_problems_methods(
            problems=[get_single_objective_benchmark_problem(num_trials=6)],
            methods=[
                get_sobol_benchmark_method(),
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP, acquisition_cls=qLogNoisyExpectedImprovement
                ),
            ],
            seeds=(0, 1),
        )

        self.assertEqual(len(aggs), 2)
        for agg in aggs:
            for col in ["mean", "P25", "P50", "P75"]:
                self.assertTrue((agg.score_trace[col] <= 100).all())

    def test_timeout(self) -> None:
        problem = SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
            test_problem_class=Branin,
            test_problem_kwargs={},
            num_trials=1000,  # Unachievable num_trials
        )

        generation_strategy = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP, acquisition_cls=qLogNoisyExpectedImprovement
        ).generation_strategy

        method = BenchmarkMethod(
            name=generation_strategy.name,
            generation_strategy=generation_strategy,
            scheduler_options=SchedulerOptions(
                max_pending_trials=1,
                init_seconds_between_polls=0,
                min_seconds_before_poll=0,
                timeout_hours=0.001,  # Strict timeout of 3.6 seconds
            ),
        )

        # Each replication will have a different number of trials
        result = benchmark_one_method_problem(
            problem=problem, method=method, seeds=(0, 1, 2, 3)
        )

        # Test the traces get composited correctly. The AggregatedResult's traces
        # should be the length of the shortest trace in the BenchmarkResults
        min_num_trials = min(len(res.optimization_trace) for res in result.results)
        self.assertEqual(len(result.optimization_trace), min_num_trials)
        self.assertEqual(len(result.score_trace), min_num_trials)
