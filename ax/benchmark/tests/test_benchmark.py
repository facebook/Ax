# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
from unittest.mock import patch

import numpy as np
from ax.benchmark.benchmark import (
    _create_benchmark_experiment,
    benchmark_multiple_problems_methods,
    benchmark_one_method_problem,
    benchmark_replication,
    make_ground_truth_metrics,
    make_ground_truth_optimization_config,
)
from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_benchmark_scheduler_options,
)
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.metrics.base import GroundTruthMetricMixin
from ax.benchmark.metrics.benchmark import BenchmarkMetric, GroundTruthBenchmarkMetric
from ax.benchmark.problems.registry import get_problem
from ax.modelbridge.generation_strategy import GenerationNode, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from ax.modelbridge.registry import Models
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.benchmark_stubs import (
    get_constrained_multi_objective_benchmark_problem,
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
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.test_functions.synthetic import Branin


class TestBenchmark(TestCase):
    @fast_botorch_optimize
    def test_batch(self) -> None:
        batch_size = 5

        problem = get_problem("ackley4", num_trials=2)
        batch_options = get_benchmark_scheduler_options(batch_size=batch_size)
        for sequential in [False, True]:
            with self.subTest(sequential=sequential):
                batch_method_joint = get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    scheduler_options=batch_options,
                    distribute_replications=False,
                    model_gen_kwargs={
                        "model_gen_options": {
                            "optimizer_kwargs": {"sequential": sequential}
                        }
                    },
                    num_sobol_trials=1,
                )
                # this is generating more calls to optimize_acqf than expected
                with patch(
                    "ax.models.torch.botorch_modular.acquisition.optimize_acqf",
                    wraps=optimize_acqf,
                ) as mock_optimize_acqf:
                    benchmark_one_method_problem(
                        problem=problem, method=batch_method_joint, seeds=[0]
                    )
                mock_optimize_acqf.assert_called_once()
                self.assertEqual(
                    mock_optimize_acqf.call_args.kwargs["sequential"], sequential
                )
                self.assertEqual(mock_optimize_acqf.call_args.kwargs["q"], batch_size)

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

    def test_make_ground_truth_metrics(self) -> None:
        problem = get_single_objective_benchmark_problem(observe_noise_sd=False)
        metric = problem.optimization_config.objective.metric

        # basic setup
        gt_metrics = make_ground_truth_metrics(problem=problem)
        self.assertEqual(len(gt_metrics), 1)
        gt_metric = checked_cast(GroundTruthBenchmarkMetric, gt_metrics[metric.name])
        self.assertIs(gt_metric.original_metric, metric)

        # add a tracking metric
        tracking_metric = BenchmarkMetric(name="test_track", lower_is_better=True)
        problem.tracking_metrics = [tracking_metric]
        gt_metrics = make_ground_truth_metrics(problem=problem)
        self.assertEqual(len(gt_metrics), 2)
        gt_tracking_metric = checked_cast(
            GroundTruthBenchmarkMetric, gt_metrics["test_track"]
        )
        self.assertIs(gt_tracking_metric.original_metric, tracking_metric)

        # set include_tracking_metrics=False
        gt_metrics = make_ground_truth_metrics(
            problem=problem, include_tracking_metrics=False
        )
        self.assertEqual(len(gt_metrics), 1)

        # error out if the problem does not have ground truth
        problem.has_ground_truth = False
        with self.assertRaisesRegex(ValueError, "do not have a ground truth"):
            make_ground_truth_metrics(problem=problem)

    def test_make_ground_truth_optimization_config(self) -> None:
        problem = get_single_objective_benchmark_problem(observe_noise_sd=False)
        metric = problem.optimization_config.objective.metric
        experiment = _create_benchmark_experiment(
            problem=problem, method_name="test_method"
        )

        # A vanilla experiment w/o ground truth metrics attached should error
        with self.assertRaisesRegex(
            ValueError, f"Ground truth metric for metric {metric.name} not found!"
        ):
            make_ground_truth_optimization_config(experiment)

        # Add the ground truth metric and check basic behavior
        gt_metric = make_ground_truth_metrics(problem)[metric.name]
        experiment.add_tracking_metric(gt_metric)
        gt_opt_cfg = make_ground_truth_optimization_config(experiment)
        self.assertIs(gt_opt_cfg.objective.metric, gt_metric)

        # Test behavior with MOO problem and outcome constraints
        problem = get_constrained_multi_objective_benchmark_problem(
            observe_noise_sd=False
        )
        experiment = _create_benchmark_experiment(
            problem=problem, method_name="test_method"
        )
        gt_metrics = make_ground_truth_metrics(problem)
        for metric in problem.optimization_config.objective.metrics:
            experiment.add_tracking_metric(gt_metrics[metric.name])
        gt_opt_cfg = make_ground_truth_optimization_config(experiment)

        for metric in gt_opt_cfg.objective.metrics:
            gt_name = metric.name
            metric = checked_cast(GroundTruthMetricMixin, metric)
            self.assertIs(metric, gt_metrics[metric.get_original_name(gt_name)])

        for metric in gt_opt_cfg.outcome_constraints:
            gt_name = metric.metric.name
            metric = checked_cast(GroundTruthMetricMixin, metric.metric)
            self.assertIs(metric, gt_metrics[metric.get_original_name(gt_name)])

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

    def test_create_benchmark_experiment(self) -> None:

        with self.subTest("noiseless"):
            problem = get_single_objective_benchmark_problem(observe_noise_sd=False)
            experiment = _create_benchmark_experiment(
                problem=problem, method_name="test_method"
            )
            self.assertTrue("|test_method_" in experiment.name)
            self.assertFalse("_observed_noise" in experiment.name)
            self.assertEqual(experiment.search_space, problem.search_space)
            self.assertEqual(
                experiment.optimization_config, problem.optimization_config
            )
            self.assertEqual(len(experiment.tracking_metrics), 0)
            self.assertEqual(experiment.runner, problem.runner)

        with self.subTest("noisy, unobserved noise std"):
            problem = get_single_objective_benchmark_problem(
                observe_noise_sd=False, test_problem_kwargs={"noise_std": 0.1}
            )
            experiment = _create_benchmark_experiment(
                problem=problem, method_name="test_method"
            )
            self.assertTrue("|test_method_" in experiment.name)
            self.assertFalse("_observed_noise" in experiment.name)
            self.assertEqual(experiment.search_space, problem.search_space)
            self.assertEqual(
                experiment.optimization_config, problem.optimization_config
            )
            self.assertEqual(len(experiment.tracking_metrics), 1)
            gt_metric = checked_cast(
                GroundTruthBenchmarkMetric, experiment.tracking_metrics[0]
            )
            self.assertIs(
                gt_metric.original_metric,
                problem.optimization_config.objective.metric,
            )
            self.assertEqual(experiment.runner, problem.runner)

        with self.subTest("noisy, observed noise std"):
            problem = get_single_objective_benchmark_problem(
                observe_noise_sd=True, test_problem_kwargs={"noise_std": 0.1}
            )
            experiment = _create_benchmark_experiment(
                problem=problem, method_name="test_method"
            )
            self.assertTrue("|test_method_" in experiment.name)
            self.assertTrue("_observed_noise" in experiment.name)
            self.assertEqual(experiment.search_space, problem.search_space)
            self.assertEqual(
                experiment.optimization_config, problem.optimization_config
            )
            self.assertEqual(len(experiment.tracking_metrics), 1)
            gt_metric = checked_cast(
                GroundTruthBenchmarkMetric, experiment.tracking_metrics[0]
            )
            self.assertIs(
                gt_metric.original_metric,
                problem.optimization_config.objective.metric,
            )
            self.assertEqual(experiment.runner, problem.runner)

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

        # This is kind of a weird setup - these are "surrogates" that use a Branin
        # synthetic function. The idea here is to test the machinery around the
        # surrogate benchmarks without having to actually load a surrogate model
        # of potentially non-neglible size.
        for name, problem in [
            ("soo", get_soo_surrogate()),
            ("moo", get_moo_surrogate()),
        ]:
            with self.subTest(name, problem=problem):
                surrogate, datasets = not_none(problem.get_surrogate_and_datasets)()
                surrogate.fit(
                    [get_dataset()],
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
                    distribute_replications=True,
                ),
                get_problem("constrained_gramacy_observed_noise", num_trials=6),
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    scheduler_options=get_benchmark_scheduler_options(),
                    distribute_replications=False,
                ),
                get_single_objective_benchmark_problem(
                    observe_noise_sd=True, num_trials=6
                ),
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    distribute_replications=False,
                ),
                get_single_objective_benchmark_problem(
                    observe_noise_sd=True, num_trials=6
                ),
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qNoisyExpectedHypervolumeImprovement,
                    distribute_replications=False,
                ),
                get_multi_objective_benchmark_problem(
                    observe_noise_sd=True, num_trials=6
                ),
                "MBM::SingleTaskGP_qNEHVI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SaasFullyBayesianSingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    distribute_replications=False,
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
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    distribute_replications=False,
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
            lower_is_better=True,
            num_trials=1000,  # Unachievable num_trials
        )

        generation_strategy = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogNoisyExpectedImprovement,
            distribute_replications=False,
            num_sobol_trials=1000,  # Ensures we don't use BO
        ).generation_strategy

        method = BenchmarkMethod(
            name=generation_strategy.name,
            generation_strategy=generation_strategy,
            scheduler_options=SchedulerOptions(
                max_pending_trials=1,
                init_seconds_between_polls=0,
                min_seconds_before_poll=0,
                timeout_hours=0.0001,  # Strict timeout of 0.36 seconds
            ),
        )

        # Each replication will have a different number of trials
        result = benchmark_one_method_problem(
            problem=problem, method=method, seeds=(0, 1)
        )

        # Test the traces get composited correctly. The AggregatedResult's traces
        # should be the length of the shortest trace in the BenchmarkResults
        min_num_trials = min(len(res.optimization_trace) for res in result.results)
        self.assertEqual(len(result.optimization_trace), min_num_trials)
        self.assertEqual(len(result.score_trace), min_num_trials)

    def test_replication_with_generation_node(self) -> None:
        method = BenchmarkMethod(
            name="Sobol Generation Node",
            generation_strategy=GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="Sobol",
                        model_specs=[
                            ModelSpec(Models.SOBOL, model_kwargs={"deduplicate": True})
                        ],
                    )
                ]
            ),
            scheduler_options=SchedulerOptions(),
        )
        problem = get_single_objective_benchmark_problem()
        res = benchmark_replication(problem=problem, method=method, seed=0)

        self.assertEqual(problem.num_trials, len(not_none(res.experiment).trials))
        self.assertTrue(np.isnan(res.score_trace).all())
