# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import tempfile
from itertools import product
from logging import WARNING
from math import pi
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from ax.benchmark.benchmark import (
    benchmark_multiple_problems_methods,
    benchmark_one_method_problem,
    benchmark_replication,
    compute_baseline_value_from_sobol,
    compute_score_trace,
    get_benchmark_scheduler_options,
    get_oracle_experiment_from_experiment,
    get_oracle_experiment_from_params,
)
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    create_problem_from_botorch,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.benchmark.benchmark_test_functions.synthetic import IdentityTestFunction
from ax.benchmark.methods.modular_botorch import (
    get_sobol_botorch_modular_acquisition,
    get_sobol_mbm_generation_strategy,
)
from ax.benchmark.methods.sobol import (
    get_sobol_benchmark_method,
    get_sobol_generation_strategy,
)
from ax.benchmark.problems.registry import get_problem
from ax.core.map_data import MapData
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.early_stopping.strategies.threshold import ThresholdEarlyStoppingStrategy
from ax.modelbridge.external_generation_node import ExternalGenerationNode
from ax.modelbridge.generation_strategy import GenerationNode, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.service.utils.scheduler_options import TrialType
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.utils.common.logger import get_logger
from ax.utils.common.mock import mock_patch_method_original
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    get_async_benchmark_method,
    get_async_benchmark_problem,
    get_discrete_search_space,
    get_moo_surrogate,
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_soo_surrogate,
    TestDataset,
)

from ax.utils.testing.core_stubs import get_branin_experiment, get_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf

from botorch.test_functions.multi_fidelity import AugmentedBranin
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
                with patch(
                    "ax.models.torch.botorch_modular.acquisition.optimize_acqf",
                    wraps=optimize_acqf,
                ) as mock_optimize_acqf:
                    benchmark_one_method_problem(
                        problem=problem,
                        method=batch_method_joint,
                        seeds=[0],
                        scheduler_logging_level=WARNING,
                    )
                mock_optimize_acqf.assert_called_once()
                self.assertEqual(
                    mock_optimize_acqf.call_args.kwargs["sequential"], sequential
                )
                self.assertEqual(mock_optimize_acqf.call_args.kwargs["q"], batch_size)

    def _test_storage(self, map_data: bool) -> None:
        problem = get_async_benchmark_problem(map_data=map_data)
        method = get_async_benchmark_method()
        res = benchmark_replication(
            problem=problem, method=method, seed=0, scheduler_logging_level=WARNING
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

    def test_storage(self) -> None:
        self._test_storage(map_data=False)
        self._test_storage(map_data=True)

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
            res = benchmark_replication(
                problem=problem, method=method, seed=0, scheduler_logging_level=WARNING
            )

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

    def test_compute_score_trace(self) -> None:
        opt_trace = np.array([1, 0, -1, 2, float("nan"), 4])

        with self.subTest("Higher is better"):
            optimal_value = 5
            baseline_value = 1
            expected_trace = np.array([0, -25, -50, 25, float("nan"), 75.0])
            trace = compute_score_trace(
                optimization_trace=opt_trace,
                baseline_value=baseline_value,
                optimal_value=optimal_value,
            )
            self.assertTrue(np.array_equal(trace, expected_trace, equal_nan=True))

        with self.subTest("Lower is better"):
            optimal_value = -1
            baseline_value = 0
            expected_trace = np.array([-100, 0, 100, -200, float("nan"), -400])
            trace = compute_score_trace(
                optimization_trace=opt_trace,
                baseline_value=baseline_value,
                optimal_value=optimal_value,
            )
            self.assertTrue(np.array_equal(trace, expected_trace, equal_nan=True))

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
                res = benchmark_replication(
                    problem=problem,
                    method=method,
                    seed=0,
                    scheduler_logging_level=WARNING,
                )

                self.assertEqual(
                    problem.num_trials,
                    len(none_throws(res.experiment).trials),
                )

                self.assertTrue(np.isfinite(res.score_trace).all())
                self.assertTrue(np.all(res.score_trace <= 100))

    def _test_replication_async(self, map_data: bool) -> None:
        """
        The test function is the identity function, higher is better, observed
        to be noiseless, and the same at every point on the trajectory. And the
        generation strategy deterministically produces
        candidates with values 0, 1, 2, .... So if the trials complete in order,
        the optimization trace should be 0, 1, 2, .... If the trials complete
        out of order, the traces should track the argmax of the completion
        order.

        Args:
            map_data: If True, the test function produces time-series data with
                just one step, so behavior is the same as when map_data=False.
        """
        method = get_async_benchmark_method()

        complete_out_of_order_runtimes = {
            0: 2,
            1: 1,
            2: 3,
            3: 1,
        }
        step_runtime_fns = {
            "All complete at different times": lambda params: params["x0"] * 3,
            "Trials complete immediately": lambda params: 0,
            "Trials complete at same time": lambda params: 1,
            "Complete out of order": lambda params: complete_out_of_order_runtimes[
                params["x0"]
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
            # Without MapData, completing after 0 seconds (second case) has the
            # same effect as completing after 1 second (third case), because a
            # new trial can't start until the next time increment.
            # With MapData, trials complete at the same times as without
            # MapData, but an extra step accrues in the third case.
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

        for case_name, step_runtime_fn in step_runtime_fns.items():
            with self.subTest(case_name, step_runtime_fn=step_runtime_fn):
                problem = get_async_benchmark_problem(
                    map_data=map_data,
                    step_runtime_fn=step_runtime_fn,
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
                        scheduler_logging_level=WARNING,
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
                    params = trial.arms[0].parameters
                    self.assertEqual(trial.index, params["x0"])
                    expected_runtime = step_runtime_fn(params=params)
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
                if map_data:
                    data = assert_is_instance(experiment.lookup_data(), MapData)
                    self.assertEqual(len(data.df), 4, msg=case_name)
                    self.assertEqual(len(data.map_df), 4, msg=case_name)

    def test_replication_async(self) -> None:
        self._test_replication_async(map_data=False)
        self._test_replication_async(map_data=True)

    def test_early_stopping(self) -> None:
        """
        Test early stopping with a deterministic generation strategy and ESS
        that stops if the objective exceeds 0.5 when their progression ("t") hits 2,
        which happens when 3 steps have passed (t=[0, 1, 2]).

        Each arm produces values equaling the trial index everywhere on the
        progression, so Trials 1, 2, and 3 will stop early, and trial 0 will not.

        t=0-2: Trials 0 and 1 run.
        t=3: Trial 1 stops early. Trial 2 gets added to "_queued", and then to
            "_running", with a queued time of 3 and a sim_start_time of 4.
        t=4: Trials 0 and 2 run.
        t=5: Trial 0 completes. Trial 2 runs.
        t=6: Trials 2 runs. Trial 3 starts with a sim_queued_time of 5 and a
            sim_start_time of 5.
        t=7: Trial 2 stops early. Trial 3 runs.
        t=8-9: Trial 3 runs by itself then gets stopped early.
        """
        min_progression = 2
        progression_length_if_not_stopped = 5
        early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=0.5,
            min_progression=min_progression,
            min_curves=0,
        )

        method = get_async_benchmark_method(
            early_stopping_strategy=early_stopping_strategy
        )

        problem = get_async_benchmark_problem(
            map_data=True,
            n_steps=progression_length_if_not_stopped,
            lower_is_better=True,
        )
        result = benchmark_replication(
            problem=problem,
            method=method,
            seed=0,
            strip_runner_before_saving=False,
            scheduler_logging_level=WARNING,
        )
        data = assert_is_instance(none_throws(result.experiment).lookup_data(), MapData)
        expected_n_steps = {
            0: progression_length_if_not_stopped,
            # stopping after step=2, so 3 steps (0, 1, 2) have passed
            **{i: min_progression + 1 for i in range(1, 4)},
        }

        grouped = data.map_df.groupby("trial_index")
        self.assertEqual(
            dict(grouped["step"].count()),
            expected_n_steps,
        )
        for trial_index, sub_df in grouped:
            self.assertEqual(
                sub_df["step"].tolist(),
                list(range(expected_n_steps[trial_index])),
                msg=f"Trial {trial_index}",
            )
        self.assertEqual(
            dict(grouped["step"].max()),
            {
                0: progression_length_if_not_stopped - 1,
                **{i: min_progression for i in range(1, 4)},
            },
        )
        simulator = none_throws(
            assert_is_instance(
                none_throws(result.experiment).runner, BenchmarkRunner
            ).simulated_backend_runner
        ).simulator
        trials = {
            trial_index: none_throws(simulator.get_sim_trial_by_index(trial_index))
            for trial_index in range(4)
        }
        start_times = {
            trial_index: sim_trial.sim_start_time
            for trial_index, sim_trial in trials.items()
        }
        expected_start_times = {
            0: 0,
            1: 0,
            2: 4,
            3: 5,
        }
        self.assertEqual(start_times, expected_start_times)

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
        res = benchmark_replication(
            problem=problem, method=method, seed=seed, scheduler_logging_level=WARNING
        )
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
                res = benchmark_replication(
                    problem=problem,
                    method=method,
                    seed=0,
                    scheduler_logging_level=WARNING,
                )
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
            scheduler_logging_level=WARNING,
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
        method = get_sobol_benchmark_method(distribute_replications=False)
        with self.assertNoLogs(level="INFO"):
            agg = benchmark_one_method_problem(
                problem=problem, method=method, seeds=(0, 1)
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
        problems = [get_single_objective_benchmark_problem(num_trials=6)]
        methods = [
            get_sobol_benchmark_method(distribute_replications=False),
            get_sobol_botorch_modular_acquisition(
                model_cls=SingleTaskGP,
                acquisition_cls=qLogNoisyExpectedImprovement,
                distribute_replications=False,
            ),
        ]
        with self.assertNoLogs(level="INFO"):
            aggs = benchmark_multiple_problems_methods(
                problems=problems,
                methods=methods,
                seeds=(0, 1),
                scheduler_logging_level=WARNING,
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
            baseline_value=100,
        )

        generation_strategy = get_sobol_mbm_generation_strategy(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogNoisyExpectedImprovement,
            num_sobol_trials=1000,  # Ensures we don't use BO
        )
        timeout_seconds = 2.0
        method = BenchmarkMethod(
            generation_strategy=generation_strategy,
            timeout_hours=timeout_seconds / 3600,
        )

        # Each replication will have a different number of trials

        start = monotonic()
        with self.assertLogs("ax.benchmark.benchmark", level="WARNING") as cm:
            result = benchmark_one_method_problem(
                problem=problem,
                method=method,
                seeds=(0, 1),
                scheduler_logging_level=WARNING,
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
        with self.assertNoLogs(logger=get_logger("ax.core.experiment"), level="INFO"):
            res = benchmark_replication(problem=problem, method=method, seed=0)

        # Check that logger level has been reset
        self.assertEqual(get_logger("ax.core.experiment").level, logging.INFO)
        self.assertEqual(problem.num_trials, len(none_throws(res.experiment).trials))
        self.assertFalse(np.isnan(res.score_trace).any())

    def test_get_oracle_experiment_from_params(self) -> None:
        problem = create_problem_from_botorch(
            test_problem_class=Branin,
            test_problem_kwargs={},
            num_trials=5,
        )
        # first is near optimum
        near_opt_params = {"x0": -pi, "x1": 12.275}
        other_params = {"x0": 0.5, "x1": 0.5}
        unbatched_experiment = get_oracle_experiment_from_params(
            problem=problem,
            dict_of_dict_of_params={0: {"0": near_opt_params}, 1: {"1": other_params}},
        )
        self.assertEqual(len(unbatched_experiment.trials), 2)
        self.assertTrue(
            all(t.status.is_completed for t in unbatched_experiment.trials.values())
        )
        self.assertTrue(
            all(len(t.arms) == 1 for t in unbatched_experiment.trials.values())
        )
        df = unbatched_experiment.fetch_data().df
        self.assertAlmostEqual(df["mean"].iloc[0], Branin._optimal_value, places=5)

        batched_experiment = get_oracle_experiment_from_params(
            problem=problem,
            dict_of_dict_of_params={0: {"0_0": near_opt_params, "0_1": other_params}},
        )
        self.assertEqual(len(batched_experiment.trials), 1)
        self.assertEqual(len(batched_experiment.trials[0].arms), 2)
        df = batched_experiment.fetch_data().df
        self.assertAlmostEqual(df["mean"].iloc[0], Branin._optimal_value, places=5)

        # Test empty inputs
        experiment = get_oracle_experiment_from_params(
            problem=problem, dict_of_dict_of_params={}
        )
        self.assertEqual(len(experiment.trials), 0)

        with self.assertRaisesRegex(ValueError, "trial with no arms"):
            get_oracle_experiment_from_params(
                problem=problem, dict_of_dict_of_params={0: {}}
            )

    def test_get_oracle_experiment_from_experiment(self) -> None:
        problem = create_problem_from_botorch(
            test_problem_class=Branin,
            test_problem_kwargs={},
            num_trials=5,
        )

        # empty experiment
        empty_experiment = get_branin_experiment(with_trial=False)
        oracle_experiment = get_oracle_experiment_from_experiment(
            problem=problem, experiment=empty_experiment
        )
        self.assertEqual(oracle_experiment.search_space, problem.search_space)
        self.assertEqual(
            oracle_experiment.optimization_config, problem.optimization_config
        )
        self.assertEqual(oracle_experiment.trials.keys(), set())

        experiment = get_branin_experiment(
            with_trial=True,
            search_space=problem.search_space,
            with_status_quo=False,
        )
        oracle_experiment = get_oracle_experiment_from_experiment(
            problem=problem, experiment=experiment
        )
        self.assertEqual(oracle_experiment.search_space, problem.search_space)
        self.assertEqual(
            oracle_experiment.optimization_config, problem.optimization_config
        )
        self.assertEqual(oracle_experiment.trials.keys(), experiment.trials.keys())

    def _test_multi_fidelity_or_multi_task(self, fidelity_or_task: str) -> None:
        """
        Args:
            fidelity_or_task: "fidelity" or "task"
        """
        parameters = [
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(2)
        ] + [
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                values=[0, 1],
                is_fidelity=fidelity_or_task == "fidelity",
                is_task=fidelity_or_task == "task",
                target_value=1,
            )
        ]
        problem = create_problem_from_botorch(
            test_problem_class=AugmentedBranin,
            test_problem_kwargs={},
            search_space=SearchSpace(parameters),
            num_trials=3,
            baseline_value=3.0,
        )
        params = {"x0": 1.0, "x1": 0.0, "x2": 0.0}
        at_target = assert_is_instance(
            Branin()
            .evaluate_true(torch.tensor([1.0, 0.0], dtype=torch.double).unsqueeze(0))
            .item(),
            float,
        )
        oracle_experiment = get_oracle_experiment_from_params(
            problem=problem, dict_of_dict_of_params={0: {"0": params}}
        )
        self.assertAlmostEqual(
            oracle_experiment.fetch_data().df["mean"].iloc[0],
            at_target,
        )
        # first term: (-(b - 0.1) * (1 - x3)  + c - r)^2
        # low-fidelity: (-b - 0.1 + c - r)^2
        # high-fidelity: (-b + c - r)^2
        t = -5.1 / (4 * pi**2) + 5 / pi - 6
        expected_change = (t + 0.1) ** 2 - t**2
        self.assertAlmostEqual(
            problem.test_function.evaluate_true(params=params).item(),
            at_target + expected_change,
        )

    def test_multi_fidelity_or_multi_task(self) -> None:
        self._test_multi_fidelity_or_multi_task(fidelity_or_task="fidelity")
        self._test_multi_fidelity_or_multi_task(fidelity_or_task="task")

    def test_get_benchmark_scheduler_options(self) -> None:
        for include_sq, batch_size in product((False, True), (1, 2)):
            method = BenchmarkMethod(
                generation_strategy=get_sobol_mbm_generation_strategy(
                    model_cls=SingleTaskGP, acquisition_cls=qLogNoisyExpectedImprovement
                ),
                distribute_replications=False,
                max_pending_trials=2,
                batch_size=batch_size,
            )
            scheduler_options = get_benchmark_scheduler_options(
                method=method, include_sq=include_sq
            )
            self.assertEqual(scheduler_options.max_pending_trials, 2)
            self.assertEqual(scheduler_options.init_seconds_between_polls, 0)
            self.assertEqual(scheduler_options.min_seconds_before_poll, 0)
            self.assertEqual(scheduler_options.batch_size, batch_size)
            self.assertEqual(
                scheduler_options.run_trials_in_batches, method.run_trials_in_batches
            )
            self.assertEqual(
                scheduler_options.early_stopping_strategy,
                method.early_stopping_strategy,
            )
            self.assertEqual(
                scheduler_options.trial_type,
                TrialType.BATCH_TRIAL
                if include_sq or batch_size > 1
                else TrialType.TRIAL,
            )
            self.assertEqual(
                scheduler_options.status_quo_weight, 1.0 if include_sq else 0.0
            )

    def test_replication_with_status_quo(self) -> None:
        method = BenchmarkMethod(
            name="Sobol", generation_strategy=get_sobol_generation_strategy()
        )
        problem = get_single_objective_benchmark_problem(
            status_quo_params={"x0": 0.0, "x1": 0.0}
        )
        res = benchmark_replication(
            problem=problem, method=method, seed=0, scheduler_logging_level=WARNING
        )

        self.assertEqual(problem.num_trials, len(none_throws(res.experiment).trials))
        for t in none_throws(res.experiment).trials.values():
            self.assertEqual(len(t.arms), 2, msg=f"Trial index: {t.index}")
            self.assertEqual(
                sum(a.name == "status_quo" for a in t.arms),
                1,
                msg=f"Trial index: {t.index}",
            )

    def test_compute_baseline_value_from_sobol(self) -> None:
        """
        In this setting, every point from 0-4 will be evaluated,
        and it will produce outcomes 0-4.
        """
        search_space = get_discrete_search_space(n_values=5)
        test_function = IdentityTestFunction()

        with self.subTest("SOO, lower is better"):
            opt_config = get_soo_opt_config(outcome_names=test_function.outcome_names)
            result = compute_baseline_value_from_sobol(
                optimization_config=opt_config,
                search_space=search_space,
                test_function=test_function,
                n_repeats=1,
            )
            self.assertEqual(result, 0)

        with self.subTest("SOO, MapData"):
            map_test_function = IdentityTestFunction(n_steps=2)
            map_opt_config = get_soo_opt_config(
                outcome_names=test_function.outcome_names, use_map_metric=True
            )
            result = compute_baseline_value_from_sobol(
                optimization_config=map_opt_config,
                search_space=search_space,
                test_function=map_test_function,
                n_repeats=1,
            )
            self.assertEqual(result, 0)

        with self.subTest("SOO, higher is better"):
            opt_config = get_soo_opt_config(
                outcome_names=test_function.outcome_names, lower_is_better=False
            )
            result = compute_baseline_value_from_sobol(
                optimization_config=opt_config,
                search_space=search_space,
                test_function=test_function,
                n_repeats=1,
            )
            self.assertEqual(result, 4)

        moo_test_function = IdentityTestFunction(outcome_names=["foo", "bar"])
        with self.subTest("MOO"):
            moo_opt_config = get_moo_opt_config(
                outcome_names=moo_test_function.outcome_names, ref_point=[5, 5]
            )
            result = compute_baseline_value_from_sobol(
                optimization_config=moo_opt_config,
                search_space=search_space,
                test_function=moo_test_function,
                n_repeats=1,
            )
            # (5-0) * (5-0)
            self.assertEqual(result, 25)
