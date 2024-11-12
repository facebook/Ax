# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
from itertools import product
from time import monotonic
from unittest.mock import patch

import numpy as np
from ax.benchmark.benchmark import (
    benchmark_multiple_problems_methods,
    benchmark_one_method_problem,
    benchmark_replication,
)
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem, create_problem_from_botorch
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.benchmark.problems.registry import get_problem
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.external_generation_node import ExternalGenerationNode
from ax.modelbridge.generation_strategy import GenerationNode, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.utils.common.mock import mock_patch_method_original
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    DeterministicGenerationNode,
    get_discrete_search_space,
    get_moo_surrogate,
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_soo_surrogate,
    IdentityTestFunction,
    TestDataset,
)
from ax.utils.testing.core_stubs import get_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.test_functions.synthetic import Branin
from pyre_extensions import assert_is_instance, none_throws


class TestBenchmark(TestCase):
    @mock_botorch_optimize
    def test_batch(self) -> None:
        batch_size = 5

        problem = get_problem("ackley4", num_trials=2)
        for sequential in [False, True]:
            with self.subTest(sequential=sequential):
                batch_method_joint = get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    batch_size=batch_size,
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
            problem=problem,
            method=get_sobol_benchmark_method(distribute_replications=False),
            seed=0,
        )
        # Experiment is not in storage yet
        self.assertTrue(res.experiment is not None)
        self.assertEqual(res.experiment_storage_id, None)
        experiment = res.experiment

        # test saving to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".json") as f:
            save_experiment(none_throws(res.experiment), f.name)
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
                inference_trace=np.array([]),
                oracle_trace=np.array([]),
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
                inference_trace=np.array([]),
                oracle_trace=np.array([]),
                optimization_trace=np.array([]),
                score_trace=np.array([]),
                fit_time=0.0,
                gen_time=0.0,
            )

    def test_replication_sobol_synthetic(self) -> None:
        method = get_sobol_benchmark_method(distribute_replications=False)
        problems = [
            get_single_objective_benchmark_problem(),
            get_problem("jenatton", num_trials=6),
        ]
        for problem in problems:
            res = benchmark_replication(problem=problem, method=method, seed=0)

            self.assertEqual(
                problem.num_trials, len(none_throws(res.experiment).trials)
            )
            self.assertTrue(np.isfinite(res.score_trace).all())
            self.assertTrue(np.all(res.score_trace <= 100))
            experiment = none_throws(res.experiment)
            self.assertIn(f"{problem.name}|Sobol", experiment.name)
            self.assertEqual(experiment.search_space, problem.search_space)
            self.assertEqual(
                experiment.optimization_config, problem.optimization_config
            )

    def test_replication_sobol_surrogate(self) -> None:
        method = get_sobol_benchmark_method(distribute_replications=False)

        # This is kind of a weird setup - these are "surrogates" that use a Branin
        # synthetic function. The idea here is to test the machinery around the
        # surrogate benchmarks without having to actually load a surrogate model
        # of potentially non-neglible size.
        for name, problem in [
            ("soo", get_soo_surrogate()),
            ("moo", get_moo_surrogate()),
        ]:
            with self.subTest(name, problem=problem):
                res = benchmark_replication(problem=problem, method=method, seed=0)

                self.assertEqual(
                    problem.num_trials,
                    len(none_throws(res.experiment).trials),
                )

                self.assertTrue(np.isfinite(res.score_trace).all())
                self.assertTrue(np.all(res.score_trace <= 100))

    def test_replication_async(self) -> None:
        """
        The test function is the identity function, higher is better, observed
        to be noiseless. And the generation strategy deterministically produces
        candidates with values 0, 1, 2, .... So if the trials complete in order,
        the optimization trace should be 0, 1, 2, .... If the trials complete
        out of order, the traces should track the argmax of the completion
        order.
        """
        search_space = get_discrete_search_space()
        method = BenchmarkMethod(
            generation_strategy=GenerationStrategy(
                nodes=[DeterministicGenerationNode(search_space=search_space)]
            ),
            distribute_replications=False,
            max_pending_trials=2,
            batch_size=1,
        )
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=BenchmarkMetric(
                    name="objective",
                    observe_noise_sd=True,
                    lower_is_better=False,
                ),
                minimize=False,
            )
        )

        complete_out_of_order_runtimes = {
            0: 2,
            1: 1,
            2: 3,
            3: 1,
        }
        trial_runtime_funcs = {
            "All complete at different times": lambda trial: trial.index * 3,
            "Trials complete immediately": lambda trial: 0,
            "Trials complete at same time": lambda trial: 1,
            "Complete out of order": lambda trial: complete_out_of_order_runtimes[
                trial.index
            ],
        }

        # First case:
        # Time   | trial 0 | trial 1 | trial 2 | trial 3
        #  t=0   |   .     |   .     |         |
        #  t=1-2 |         |   .     |   .     |
        #  t=3-6 |         |         |   .     |    .
        #  t=7-12|         |         |         |    .

        # Second case:
        # Time   | trial 0 | trial 1 | trial 2 | trial 3
        #  t=0   |   .     |   .     |         |
        #  t=1   |   .     |         |   .     |
        #  t=2   |         |         |   .     |    .
        #  t=3   |         |         |   .     |
        expected_start_times = {
            "All complete at different times": [0, 0, 1, 3],
            "Trials complete immediately": [0, 0, 1, 1],
            # Completing after 0 seconds has the same effect as completing after
            # 1 second, because a new trial can't start until the next time
            # increment.
            "Trials complete at same time": [0, 0, 1, 1],
            "Complete out of order": [0, 0, 1, 2],
        }
        expected_pending_in_each_gen = {
            "All complete at different times": [[], [0], [1], [2]],
            "Trials complete immediately": [[], [0], [], [2]],
            "Trials complete at same time": [[], [0], [], [2]],
            "Complete out of order": [[], [0], [0], [2]],
        }
        # When two trials complete at the same time, the inference trace uses
        # data from both to get the best point, and repeats it.
        # The oracle trace is the same.
        expected_inference_traces = {
            "All complete at different times": [0, 1, 2, 3],
            # 0 and 1 complete at the same time, as do 2 and 3
            "Trials complete immediately": [1, 1, 3, 3],
            "Trials complete at same time": [1, 1, 3, 3],
            "Complete out of order": [1, 1, 3, 3],
        }
        test_function = IdentityTestFunction()

        for case_name, trial_runtime_func in trial_runtime_funcs.items():
            with self.subTest(case_name, trial_runtime_func=trial_runtime_func):
                problem = BenchmarkProblem(
                    name="test",
                    search_space=search_space,
                    optimization_config=optimization_config,
                    test_function=test_function,
                    num_trials=4,
                    optimal_value=19.0,
                    trial_runtime_func=trial_runtime_func,
                )

                with mock_patch_method_original(
                    mock_path=(
                        "ax.utils.testing.benchmark_stubs.ExternalGenerationNode._gen"
                    ),
                    original_method=ExternalGenerationNode._gen,
                ) as mock_gen:
                    result = benchmark_replication(
                        problem=problem,
                        method=method,
                        seed=0,
                        strip_runner_before_saving=False,
                    )
                pending_in_each_gen = [
                    [
                        elt[0].trial_index
                        for elt in call_kwargs.get("pending_observations").values()
                    ]
                    for _, call_kwargs in mock_gen.call_args_list
                ]
                self.assertEqual(
                    pending_in_each_gen,
                    expected_pending_in_each_gen[case_name],
                    case_name,
                )

                experiment = none_throws(result.experiment)
                runner = assert_is_instance(experiment.runner, BenchmarkRunner)
                backend_simulator = none_throws(
                    runner.simulated_backend_runner
                ).simulator
                completed_trials = backend_simulator.state().completed
                self.assertEqual(len(completed_trials), 4)
                for trial_index, expected_start_time in enumerate(
                    expected_start_times[case_name]
                ):
                    trial = experiment.trials[trial_index]
                    self.assertEqual(trial.index, trial.arms[0].parameters["x0"])
                    expected_runtime = trial_runtime_func(trial)
                    self.assertEqual(
                        backend_simulator.get_sim_trial_by_index(
                            trial_index=trial_index
                        ).__dict__,
                        {
                            "trial_index": trial_index,
                            "sim_runtime": expected_runtime,
                            "sim_start_time": expected_start_time,
                            "sim_queued_time": expected_start_time,
                            "sim_completed_time": expected_start_time
                            + expected_runtime,
                        },
                        f"Failure for trial {trial_index} with {case_name}",
                    )
                self.assertFalse(np.isnan(result.inference_trace).any())
                self.assertEqual(
                    result.inference_trace.tolist(),
                    expected_inference_traces[case_name],
                )

    @mock_botorch_optimize
    def _test_replication_with_inference_value(
        self,
        batch_size: int,
        use_model_predictions: bool,
        report_inference_value_as_trace: bool,
    ) -> None:
        seed = 1
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogNoisyExpectedImprovement,
            distribute_replications=False,
            use_model_predictions_for_best_point=use_model_predictions,
            num_sobol_trials=3,
            batch_size=batch_size,
        )

        num_trials = 4
        problem = get_single_objective_benchmark_problem(
            num_trials=num_trials,
            report_inference_value_as_trace=report_inference_value_as_trace,
            noise_std=100.0,
        )
        res = benchmark_replication(problem=problem, method=method, seed=seed)
        # The inference trace could coincide with the oracle trace, but it won't
        # happen in this example with high noise and a seed
        self.assertEqual(
            np.equal(res.inference_trace, res.optimization_trace).all(),
            report_inference_value_as_trace,
        )
        self.assertEqual(
            np.equal(res.oracle_trace, res.optimization_trace).all(),
            not report_inference_value_as_trace,
        )

        self.assertEqual(res.optimization_trace.shape, (problem.num_trials,))
        self.assertTrue((res.inference_trace >= res.oracle_trace).all())
        self.assertTrue((res.score_trace >= 0).all())
        self.assertTrue((res.score_trace <= 100).all())

    def test_replication_with_inference_value(self) -> None:
        for (
            use_model_predictions,
            batch_size,
            report_inference_value_as_trace,
        ) in product(
            [False, True],
            [1, 2],
            [False, True],
        ):
            with self.subTest(
                batch_size=batch_size,
                use_model_predictions=use_model_predictions,
                report_inference_value_as_trace=report_inference_value_as_trace,
            ):
                self._test_replication_with_inference_value(
                    batch_size=batch_size,
                    use_model_predictions=use_model_predictions,
                    report_inference_value_as_trace=report_inference_value_as_trace,
                )

        with self.assertRaisesRegex(
            NotImplementedError,
            "Inference trace is not supported for MOO",
        ):
            get_multi_objective_benchmark_problem(report_inference_value_as_trace=True)

    @mock_botorch_optimize
    def test_replication_mbm(self) -> None:
        with patch.dict(
            "ax.benchmark.problems.hpo.torchvision._REGISTRY",
            {"MNIST": TestDataset},
        ):
            mnist_problem = get_problem(
                problem_key="hpo_pytorch_cnn_MNIST", name="MNIST", num_trials=6
            )
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
                    acquisition_cls=qLogNoisyExpectedHypervolumeImprovement,
                    distribute_replications=False,
                ),
                get_multi_objective_benchmark_problem(
                    observe_noise_sd=True, num_trials=6
                ),
                "MBM::SingleTaskGP_qLogNEHVI",
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
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    distribute_replications=False,
                ),
                mnist_problem,
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qKnowledgeGradient,
                    distribute_replications=False,
                ),
                get_single_objective_benchmark_problem(
                    observe_noise_sd=False, num_trials=6
                ),
                "MBM::SingleTaskGP_qKnowledgeGradient",
            ),
        ]:
            with self.subTest(method=method, problem=problem):
                res = benchmark_replication(problem=problem, method=method, seed=0)
                self.assertEqual(
                    problem.num_trials,
                    len(none_throws(res.experiment).trials),
                )
                self.assertTrue(np.all(res.score_trace <= 100))
                self.assertEqual(method.name, method.generation_strategy.name)
                self.assertEqual(method.name, expected_name)

    def test_replication_moo_sobol(self) -> None:
        problem = get_multi_objective_benchmark_problem()

        res = benchmark_replication(
            problem=problem,
            method=get_sobol_benchmark_method(distribute_replications=False),
            seed=0,
        )

        self.assertEqual(
            problem.num_trials,
            len(none_throws(res.experiment).trials),
        )
        self.assertEqual(
            problem.num_trials * 2,
            len(none_throws(res.experiment).fetch_data().df),
        )

        self.assertTrue(np.all(res.score_trace <= 100))

    def test_benchmark_one_method_problem(self) -> None:
        problem = get_single_objective_benchmark_problem()
        agg = benchmark_one_method_problem(
            problem=problem,
            method=get_sobol_benchmark_method(distribute_replications=False),
            seeds=(0, 1),
        )

        self.assertEqual(len(agg.results), 2)
        self.assertTrue(
            all(
                len(none_throws(result.experiment).trials) == problem.num_trials
                for result in agg.results
            ),
            "All experiments must have 4 trials",
        )

        for col in ["mean", "P25", "P50", "P75"]:
            self.assertTrue((agg.score_trace[col] <= 100).all())

    @mock_botorch_optimize
    def test_benchmark_multiple_problems_methods(self) -> None:
        aggs = benchmark_multiple_problems_methods(
            problems=[get_single_objective_benchmark_problem(num_trials=6)],
            methods=[
                get_sobol_benchmark_method(distribute_replications=False),
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
        problem = create_problem_from_botorch(
            test_problem_class=Branin,
            test_problem_kwargs={},
            num_trials=1000,  # Unachievable num_trials
        )

        generation_strategy = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogNoisyExpectedImprovement,
            distribute_replications=False,
            num_sobol_trials=1000,  # Ensures we don't use BO
        ).generation_strategy

        timeout_seconds = 2.0
        method = BenchmarkMethod(
            name=generation_strategy.name,
            generation_strategy=generation_strategy,
            timeout_hours=timeout_seconds / 3600,
        )

        # Each replication will have a different number of trials

        start = monotonic()
        with self.assertLogs("ax.benchmark.benchmark", level="WARNING") as cm:
            result = benchmark_one_method_problem(
                problem=problem, method=method, seeds=(0, 1)
            )
        elapsed = monotonic() - start
        self.assertGreater(elapsed, timeout_seconds)
        self.assertIn(
            "WARNING:ax.benchmark.benchmark:The optimization loop timed out.", cm.output
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
        )
        problem = get_single_objective_benchmark_problem()
        res = benchmark_replication(problem=problem, method=method, seed=0)

        self.assertEqual(problem.num_trials, len(none_throws(res.experiment).trials))
        self.assertTrue(np.isnan(res.score_trace).all())
