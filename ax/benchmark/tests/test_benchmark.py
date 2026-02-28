# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import itertools
import logging
import tempfile
from datetime import datetime
from itertools import product
from logging import WARNING
from math import pi
from time import monotonic
from typing import Literal
from unittest.mock import patch

import numpy as np
import torch
from ax.adapter.factory import get_sobol
from ax.adapter.registry import Generators
from ax.benchmark.benchmark import (
    _get_oracle_value_of_params,
    benchmark_multiple_problems_methods,
    benchmark_one_method_problem,
    benchmark_replication,
    compute_baseline_value_from_sobol,
    compute_score_trace,
    get_benchmark_orchestrator_options,
    get_benchmark_result_from_experiment_and_gs,
    get_benchmark_result_with_cumulative_steps,
    get_best_parameters,
    get_opt_trace_by_steps,
    get_oracle_experiment_from_params,
    run_optimization_with_orchestrator,
)
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    get_continuous_search_space,
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
from ax.benchmark.problems.registry import get_benchmark_problem
from ax.benchmark.problems.synthetic.from_botorch import (
    create_problem_from_botorch,
    get_augmented_branin_problem,
)
from ax.benchmark.testing.benchmark_stubs import (
    get_async_benchmark_method,
    get_async_benchmark_problem,
    get_discrete_search_space,
    get_moo_surrogate,
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_soo_surrogate,
)
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.early_stopping.strategies.threshold import ThresholdEarlyStoppingStrategy
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.service.utils.orchestrator_options import TrialType
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.utils.common.logger import get_logger
from ax.utils.common.mock import mock_patch_method_original
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.test_functions.synthetic import Branin, PressureVessel
from pyre_extensions import assert_is_instance, none_throws


class TestBenchmark(TestCase):
    def benchmark_replication(
        self,
        problem: BenchmarkProblem,
        method: BenchmarkMethod,
        seed: int,
        strip_runner_before_saving: bool = True,
    ) -> BenchmarkResult:
        """
        Run benchmark_replication with logs set to WARNING.

        Suppresses voluminous INFO logs from the orchestrator.
        """
        return benchmark_replication(
            problem=problem,
            method=method,
            seed=seed,
            strip_runner_before_saving=strip_runner_before_saving,
            orchestrator_logging_level=WARNING,
        )

    def run_optimization_with_orchestrator(
        self,
        problem: BenchmarkProblem,
        method: BenchmarkMethod,
        seed: int,
    ) -> Experiment:
        """
        Run run_optimization_with_orchestrator with logs set to WARNING.

        Suppresses voluminous INFO logs from the orchestrator.
        """
        return run_optimization_with_orchestrator(
            problem=problem,
            method=method,
            seed=seed,
            orchestrator_logging_level=WARNING,
        )

    @mock_botorch_optimize
    def test_batch(self) -> None:
        batch_size = 5

        problem = get_benchmark_problem("ackley4", num_trials=2)
        for sequential in [False, True]:
            with self.subTest(sequential=sequential):
                batch_method_joint = get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                    batch_size=batch_size,
                    generator_gen_kwargs={
                        "model_gen_options": {
                            "optimizer_kwargs": {"sequential": sequential}
                        }
                    },
                    num_sobol_trials=1,
                )
                with patch(
                    "ax.generators.torch.botorch_modular.acquisition.optimize_acqf",
                    wraps=optimize_acqf,
                ) as mock_optimize_acqf:
                    self.run_optimization_with_orchestrator(
                        problem=problem, method=batch_method_joint, seed=0
                    )
                mock_optimize_acqf.assert_called_once()
                self.assertEqual(
                    mock_optimize_acqf.call_args.kwargs["sequential"], sequential
                )
                self.assertEqual(mock_optimize_acqf.call_args.kwargs["q"], batch_size)

    def _test_storage(self, map_data: bool) -> None:
        problem = get_async_benchmark_problem(map_data=map_data)
        method = get_async_benchmark_method()
        res = self.benchmark_replication(problem=problem, method=method, seed=0)
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

    def test_replication_sobol_synthetic(self) -> None:
        method = get_sobol_benchmark_method()
        problems = [
            get_single_objective_benchmark_problem(),
            get_benchmark_problem("jenatton", num_trials=6),
        ]
        for problem in problems:
            res = self.benchmark_replication(problem=problem, method=method, seed=0)

            self.assertEqual(
                problem.num_trials, len(none_throws(res.experiment).trials)
            )
            self.assertTrue(np.isfinite(res.score_trace).all())
            self.assertTrue(np.all(np.array(res.score_trace) <= 100))
            experiment = none_throws(res.experiment)
            self.assertIn(f"{problem.name}|Sobol", experiment.name)
            self.assertEqual(experiment.search_space, problem.search_space)
            self.assertEqual(
                experiment.optimization_config, problem.optimization_config
            )

    def test_tracking_metrics(self) -> None:
        method = get_sobol_benchmark_method()
        problem = get_multi_objective_benchmark_problem()
        oc = problem.optimization_config
        tracking_metric = (
            assert_is_instance(oc.objective, MultiObjective).objectives[1].metric
        )
        problem = dataclasses.replace(
            problem,
            optimization_config=get_soo_opt_config(
                outcome_names=[f"{problem.name}_0"], lower_is_better=True
            ),
            tracking_metrics=[tracking_metric],
            baseline_value=3.0,
            optimal_value=Branin(negate=False).optimal_value,
        )
        res = self.benchmark_replication(problem=problem, method=method, seed=0)

        self.assertEqual(problem.num_trials, len(none_throws(res.experiment).trials))
        self.assertTrue(np.isfinite(res.score_trace).all())
        self.assertTrue(all(y <= 100 for y in res.score_trace))
        experiment = none_throws(res.experiment)
        self.assertIn(f"{problem.name}|Sobol", experiment.name)
        self.assertEqual(experiment.search_space, problem.search_space)
        self.assertEqual(experiment.optimization_config, problem.optimization_config)
        self.assertEqual(experiment.tracking_metrics, [tracking_metric])
        self.assertEqual(
            set(experiment.lookup_data().df["metric_name"].unique()),
            {f"{problem.name}_0", f"{problem.name}_1"},
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
                res = self.benchmark_replication(problem=problem, method=method, seed=0)

                self.assertEqual(
                    problem.num_trials,
                    len(none_throws(res.experiment).trials),
                )

                self.assertTrue(np.isfinite(res.score_trace).all())
                self.assertTrue(np.all(np.array(res.score_trace) <= 100))

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
            # Without Data with `has_step_column=True`, completing after 0
            # seconds (second case) has the same effect as completing after 1
            # second (third case), because a new trial can't start until the
            # next time increment.
            # With Data with `has_step_column=True`, trials complete at the same
            # times as without, but an extra step accrues in the third case.
            "Trials complete at same time": [0, 0, 1, 1],
            "Complete out of order": [0, 0, 1, 2],
        }
        expected_pending_in_each_gen = {
            "All complete at different times": [[None], [0], [1], [2]],
            "Trials complete immediately": [[None], [0], [None], [2]],
            "Trials complete at same time": [[None], [0], [None], [2]],
            "Complete out of order": [[None], [0], [0], [2]],
        }
        # When two trials complete at the same time, the inference trace uses
        # data from both to get the best point, and repeats it.
        expected_traces = {
            "All complete at different times": [0, 1, 2, 3],
            # 0 and 1 complete at the same time, as do 2 and 3
            "Trials complete immediately": [1, 3],
            "Trials complete at same time": [1, 3],
            "Complete out of order": [1, 1, 3, 3],
        }
        expected_costs = {
            "All complete at different times": [0, 3, 7, 12],
            "Trials complete immediately": [0, 1],
            "Trials complete at same time": [1, 2],
            "Complete out of order": [1, 2, 3, 4],
        }
        expected_num_trials = {
            "All complete at different times": [1, 2, 3, 4],
            "Trials complete immediately": [2, 4],
            "Trials complete at same time": [2, 4],
            "Complete out of order": [1, 2, 3, 4],
        }
        expected_backend_simulator_time = {
            "All complete at different times": 12,
            "Trials complete immediately": 2,
            "Trials complete at same time": 2,
            "Complete out of order": 4,
        }

        for case_name, step_runtime_fn in step_runtime_fns.items():
            with self.subTest(case_name, step_runtime_fn=step_runtime_fn):
                problem = get_async_benchmark_problem(
                    map_data=map_data,
                    step_runtime_fn=step_runtime_fn,
                    report_inference_value_as_trace=True,
                )

                with mock_patch_method_original(
                    mock_path=(
                        "ax.benchmark.testing.benchmark_stubs."
                        "ExternalGenerationNode._gen"
                    ),
                    original_method=ExternalGenerationNode._gen,
                ) as mock_gen:
                    result = self.benchmark_replication(
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
                self.assertEqual(
                    backend_simulator.time,
                    expected_backend_simulator_time[case_name],
                    msg=case_name,
                )
                completed_trials = backend_simulator.state().completed
                self.assertEqual(len(completed_trials), 4)

                params_dict = {
                    idx: trial.arms[0].parameters
                    for idx, trial in experiment.trials.items()
                }
                expected_runtimes = {
                    idx: step_runtime_fn(params=params)
                    for idx, params in params_dict.items()
                }
                for trial_index, expected_start_time in enumerate(
                    expected_start_times[case_name]
                ):
                    self.assertEqual(trial_index, params_dict[trial_index]["x0"])
                    expected_runtime = expected_runtimes[trial_index]
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
                    result.inference_trace,
                    expected_traces[case_name],
                    msg=case_name,
                )
                self.assertEqual(
                    result.oracle_trace,
                    expected_traces[case_name],
                    msg=case_name,
                )
                self.assertEqual(
                    result.cost_trace,
                    expected_costs[case_name],
                    msg=case_name,
                )
                self.assertEqual(
                    result.num_trials,
                    expected_num_trials[case_name],
                    msg=case_name,
                )
                if map_data:
                    data = experiment.lookup_data()
                    self.assertEqual(len(data.df), 4, msg=case_name)
                    self.assertEqual(len(data.full_df), 4, msg=case_name)

                # Check trial start and end times
                start_of_time = datetime.fromtimestamp(0)
                created_times = [
                    (trial.time_created - start_of_time).total_seconds()
                    for i, trial in experiment.trials.items()
                ]
                self.assertEqual(created_times, expected_start_times[case_name])
                queued_times = [
                    (
                        none_throws(experiment.trials[i].time_run_started)
                        - start_of_time
                    ).total_seconds()
                    for i in range(4)
                ]
                self.assertEqual(queued_times, expected_start_times[case_name])
                completed_times = [
                    (
                        none_throws(experiment.trials[i].time_completed) - start_of_time
                    ).total_seconds()
                    for i in range(4)
                ]
                expected_completed_times = [
                    expected_start_times[case_name][i] + expected_runtimes[i]
                    for i in range(4)
                ]
                self.assertEqual(completed_times, expected_completed_times)

    def test_replication_async(self) -> None:
        self._test_replication_async(map_data=False)
        self._test_replication_async(map_data=True)

    def test_run_optimization_with_orchestrator(self) -> None:
        method = get_async_benchmark_method()
        problem = get_async_benchmark_problem(
            map_data=True,
        )

        # Test logging
        logger = get_logger("utils.testing.backend_simulator")

        with self.subTest("Logs produced if level is DEBUG"):
            with self.assertLogs(level=logging.DEBUG, logger=logger):
                experiment = run_optimization_with_orchestrator(
                    problem=problem,
                    method=method,
                    seed=0,
                    orchestrator_logging_level=logging.DEBUG,
                )
            runner = assert_is_instance(experiment.runner, BenchmarkRunner)
            self.assertFalse(
                none_throws(runner.simulated_backend_runner).simulator._verbose_logging
            )

        with (
            self.subTest("Logs not produced by default"),
            self.assertNoLogs(level=logging.INFO, logger=logger),
            self.assertNoLogs(logger=logger),
        ):
            run_optimization_with_orchestrator(
                problem=problem,
                method=method,
                seed=0,
            )

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
        experiment = self.run_optimization_with_orchestrator(
            problem=problem, method=method, seed=0
        )
        data = experiment.lookup_data()
        expected_n_steps = {
            0: progression_length_if_not_stopped,
            # stopping after step=2, so 3 steps (0, 1, 2) have passed
            **{i: min_progression + 1 for i in range(1, 4)},
        }

        grouped = data.full_df.groupby("trial_index")
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
                experiment.runner, BenchmarkRunner
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

        with self.subTest("max_pending_trials = 1"):
            method = get_async_benchmark_method(
                early_stopping_strategy=early_stopping_strategy,
                max_pending_trials=1,
            )
            experiment = self.run_optimization_with_orchestrator(
                problem=problem, method=method, seed=0
            )
            simulated_backend_runner = assert_is_instance(
                experiment.runner, BenchmarkRunner
            ).simulated_backend_runner
            self.assertIsNotNone(simulated_backend_runner)
            expected_start_times = {
                0: 0,
                1: 5,  # Early-stopped at t=8
                2: 9,  # Early-stopped at t=11
                3: 13,  # Early-stopped at t=16
            }
            simulator = none_throws(simulated_backend_runner).simulator
            trials = {
                trial_index: none_throws(simulator.get_sim_trial_by_index(trial_index))
                for trial_index in range(4)
            }
            start_times = {
                trial_index: sim_trial.sim_start_time
                for trial_index, sim_trial in trials.items()
            }
            self.assertEqual(start_times, expected_start_times)
            full_df = experiment.lookup_data().full_df
            max_run = full_df.groupby("trial_index")["step"].max().to_dict()
            self.assertEqual(max_run, {0: 4, 1: 2, 2: 2, 3: 2})

    def test_replication_variable_runtime(self) -> None:
        method = get_async_benchmark_method(max_pending_trials=1)
        for map_data in [False, True]:
            with self.subTest(map_data=map_data):
                problem = get_async_benchmark_problem(
                    map_data=map_data,
                    step_runtime_fn=lambda params: params["x0"] + 1,
                )
                experiment = self.run_optimization_with_orchestrator(
                    problem=problem, method=method, seed=0
                )
                simulated_backend_runner = assert_is_instance(
                    experiment.runner, BenchmarkRunner
                ).simulated_backend_runner
                self.assertIsNotNone(simulated_backend_runner)
                expected_start_times = {
                    0: 0,
                    1: 1,
                    2: 3,
                    3: 6,
                }
                simulator = none_throws(simulated_backend_runner).simulator
                trials = {
                    trial_index: none_throws(
                        simulator.get_sim_trial_by_index(trial_index)
                    )
                    for trial_index in range(4)
                }
                start_times = {
                    trial_index: sim_trial.sim_start_time
                    for trial_index, sim_trial in trials.items()
                }
                self.assertEqual(start_times, expected_start_times)

    @mock_botorch_optimize
    def _test_replication_with_inference_value(
        self, batch_size: int, report_inference_value_as_trace: bool
    ) -> None:
        seed = 1
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogNoisyExpectedImprovement,
            num_sobol_trials=3,
            batch_size=batch_size,
        )

        num_trials = 4
        problem = get_single_objective_benchmark_problem(
            num_trials=num_trials,
            report_inference_value_as_trace=report_inference_value_as_trace,
            noise_std=100.0,
        )
        res = self.benchmark_replication(problem=problem, method=method, seed=seed)
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

        self.assertEqual(len(res.optimization_trace), problem.num_trials)
        if report_inference_value_as_trace:
            self.assertTrue(
                (np.array(res.inference_trace) >= np.array(res.oracle_trace)).all()
            )

    def test_replication_with_inference_value(self) -> None:
        for batch_size, report_inference_value_as_trace in product(
            [1, 2], [False, True]
        ):
            with self.subTest(
                batch_size=batch_size,
                report_inference_value_as_trace=report_inference_value_as_trace,
            ):
                self._test_replication_with_inference_value(
                    batch_size=batch_size,
                    report_inference_value_as_trace=report_inference_value_as_trace,
                )

        with self.assertRaisesRegex(
            NotImplementedError,
            "Inference trace is not supported for MOO",
        ):
            get_multi_objective_benchmark_problem(report_inference_value_as_trace=True)

    @mock_botorch_optimize
    def test_replication_mbm(self) -> None:
        for method, problem, expected_name in [
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                ),
                get_benchmark_problem(
                    "constrained_gramacy_observed_noise", num_trials=6
                ),
                "MBM::SingleTaskGP_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
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
                ),
                get_multi_objective_benchmark_problem(num_trials=6),
                "MBM::SAAS_qLogNEI",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qKnowledgeGradient,
                ),
                get_single_objective_benchmark_problem(
                    observe_noise_sd=False, num_trials=6
                ),
                "MBM::SingleTaskGP_qKnowledgeGradient",
            ),
            (
                get_sobol_botorch_modular_acquisition(
                    model_cls=SingleTaskGP,
                    acquisition_cls=qLogNoisyExpectedImprovement,
                ),
                get_augmented_branin_problem(fidelity_or_task="fidelity"),
                "MBM::SingleTaskGP_qLogNEI",
            ),
        ]:
            with self.subTest(method=method, problem=problem):
                res = self.benchmark_replication(problem=problem, method=method, seed=0)
                self.assertEqual(
                    problem.num_trials,
                    len(none_throws(res.experiment).trials),
                )
                self.assertTrue(np.all(np.array(res.score_trace) <= 100))
                self.assertEqual(method.name, method.generation_strategy.name)
                self.assertEqual(method.name, expected_name)

    def test_replication_moo_sobol(self) -> None:
        problem = get_multi_objective_benchmark_problem()

        res = self.benchmark_replication(
            problem=problem,
            method=get_sobol_benchmark_method(),
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

        self.assertTrue(np.all(np.array(res.score_trace) <= 100))
        self.assertEqual(len(res.cost_trace), problem.num_trials)
        self.assertEqual(len(none_throws(res.num_trials)), problem.num_trials)
        self.assertEqual(
            none_throws(res.num_trials), list(range(1, problem.num_trials + 1))
        )
        self.assertEqual(len(res.inference_trace), problem.num_trials)
        # since inference trace is not supported for MOO, it should be all NaN
        self.assertTrue(np.isnan(res.inference_trace).all())

    def test_benchmark_one_method_problem(self) -> None:
        problem = get_single_objective_benchmark_problem()
        method = get_sobol_benchmark_method()
        with self.assertNoLogs(level="INFO"):
            agg = benchmark_one_method_problem(
                problem=problem,
                method=method,
                seeds=(0, 1),
                orchestrator_logging_level=WARNING,
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
        self.assertIn("num_trials", agg.optimization_trace.columns)
        self.assertIn("num_trials", agg.score_trace.columns)
        self.assertTrue((agg.optimization_trace["num_trials"] > 0).all())
        self.assertTrue((agg.score_trace["num_trials"] > 0).all())

    @mock_botorch_optimize
    def test_benchmark_multiple_problems_methods(self) -> None:
        problems = [get_single_objective_benchmark_problem(num_trials=6)]
        methods = [
            get_sobol_benchmark_method(),
            get_sobol_botorch_modular_acquisition(
                model_cls=SingleTaskGP,
                acquisition_cls=qLogNoisyExpectedImprovement,
            ),
        ]
        with self.assertNoLogs(level="INFO"):
            aggs = benchmark_multiple_problems_methods(
                problems=problems,
                methods=methods,
                seeds=(0, 1),
                orchestrator_logging_level=WARNING,
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
        method = BenchmarkMethod(generation_strategy=generation_strategy)

        # Each replication will have a different number of trials

        start = monotonic()
        with self.assertLogs("ax.orchestration.orchestrator", level="ERROR") as cm:
            result = benchmark_one_method_problem(
                problem=problem,
                method=method,
                seeds=(0, 1),
                timeout_hours=timeout_seconds / 3600,
                orchestrator_logging_level=WARNING,
            )
        elapsed = monotonic() - start
        self.assertGreater(elapsed, timeout_seconds)
        self.assertTrue(any("Optimization timed out" in output for output in cm.output))

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
                        name="Sobol",
                        generator_specs=[
                            GeneratorSpec(
                                Generators.SOBOL, generator_kwargs={"deduplicate": True}
                            )
                        ],
                    )
                ]
            ),
        )
        problem = get_single_objective_benchmark_problem()
        with self.assertNoLogs(logger=get_logger("ax.core.experiment"), level="INFO"):
            res = self.benchmark_replication(problem=problem, method=method, seed=0)

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

        with self.subTest("trial_statuses"):
            trial_statuses = {
                0: TrialStatus.COMPLETED,
                1: TrialStatus.ABANDONED,
            }
            experiment = get_oracle_experiment_from_params(
                problem=problem,
                dict_of_dict_of_params={
                    0: {"0": near_opt_params},
                    1: {"1": other_params},
                },
                trial_statuses=trial_statuses,
            )
            self.assertEqual(len(experiment.trials), 2)
            self.assertTrue(experiment.trials[0].status.is_completed)
            self.assertEqual(experiment.trials[1].status, TrialStatus.ABANDONED)

        with self.subTest("trial_statuses with FAILED and EARLY_STOPPED"):
            trial_statuses = {
                0: TrialStatus.FAILED,
                1: TrialStatus.EARLY_STOPPED,
            }
            experiment = get_oracle_experiment_from_params(
                problem=problem,
                dict_of_dict_of_params={
                    0: {"0": near_opt_params},
                    1: {"1": other_params},
                },
                trial_statuses=trial_statuses,
            )
            self.assertEqual(experiment.trials[0].status, TrialStatus.FAILED)
            self.assertEqual(experiment.trials[1].status, TrialStatus.EARLY_STOPPED)

        with self.subTest("trial_statuses=None defaults to COMPLETED"):
            experiment = get_oracle_experiment_from_params(
                problem=problem,
                dict_of_dict_of_params={
                    0: {"0": near_opt_params},
                    1: {"1": other_params},
                },
                trial_statuses=None,
            )
            self.assertTrue(
                all(t.status.is_completed for t in experiment.trials.values())
            )

    def _test_multi_fidelity_or_multi_task(
        self, fidelity_or_task: Literal["fidelity", "task"]
    ) -> None:
        """
        Args:
            fidelity_or_task: "fidelity" or "task"
        """
        problem = get_augmented_branin_problem(fidelity_or_task=fidelity_or_task)
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

    def test_get_benchmark_orchestrator_options(self) -> None:
        for include_sq, batch_size in product((False, True), (1, 2)):
            method = BenchmarkMethod(
                generation_strategy=get_sobol_mbm_generation_strategy(
                    model_cls=SingleTaskGP, acquisition_cls=qLogNoisyExpectedImprovement
                ),
                max_pending_trials=2,
                batch_size=batch_size,
            )
            orchestrator_options = get_benchmark_orchestrator_options(
                batch_size=none_throws(method.batch_size),
                run_trials_in_batches=False,
                max_pending_trials=method.max_pending_trials,
                early_stopping_strategy=method.early_stopping_strategy,
                include_status_quo=include_sq,
            )
            self.assertEqual(orchestrator_options.max_pending_trials, 2)
            self.assertEqual(orchestrator_options.init_seconds_between_polls, 0)
            self.assertEqual(orchestrator_options.min_seconds_before_poll, 0)
            self.assertEqual(orchestrator_options.batch_size, batch_size)
            self.assertFalse(orchestrator_options.run_trials_in_batches)
            self.assertEqual(
                orchestrator_options.early_stopping_strategy,
                method.early_stopping_strategy,
            )
            self.assertEqual(
                orchestrator_options.trial_type,
                TrialType.BATCH_TRIAL
                if include_sq or batch_size > 1
                else TrialType.TRIAL,
            )
            self.assertEqual(
                orchestrator_options.status_quo_weight, 1.0 if include_sq else 0.0
            )

    def test_replication_with_status_quo(self) -> None:
        method = BenchmarkMethod(
            name="Sobol", generation_strategy=get_sobol_generation_strategy()
        )
        problem = get_single_objective_benchmark_problem(
            status_quo_params={"x0": 0.0, "x1": 0.0}
        )
        experiment = self.run_optimization_with_orchestrator(
            problem=problem, method=method, seed=0
        )

        self.assertEqual(problem.num_trials, len(experiment.trials))
        for t in experiment.trials.values():
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

        with self.subTest("SOO, Data with has_step_column=True"):
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

    def test_get_oracle_value_of_params(self) -> None:
        problem = get_augmented_branin_problem(fidelity_or_task="fidelity")
        # params are not at target value
        params = {"x0": 1.0, "x1": 0.0, "x2": 0.0}
        inference_value = _get_oracle_value_of_params(params=params, problem=problem)
        oracle_params = {"x0": 1.0, "x1": 0.0, "x2": 1.0}
        self.assertEqual(
            inference_value,
            problem.test_function.evaluate_true(params=oracle_params).item(),
        )

    def test_get_opt_trace_by_cumulative_epochs(self) -> None:
        # Time  | trial 0 | trial 1 | trial 2 | trial 3 | new steps
        #  t=0  |   ..    |  start  |         |         |   0, 0
        #  t=1  |         |    .    |  start  |         |   1
        #  t=2  |         |    ..   |         |  start  |   1
        #  t=3  |         |         |    .    |         |   2
        #  t=4  |         |         |         |         |
        #  t=5  |         |         |    ..   |    .    |   3, 2

        # ...
        problem = get_async_benchmark_problem(
            map_data=True,
            n_steps=2,
            # Ensure we don't have two finishing at the same time, for
            # determinism
            step_runtime_fn=lambda params: params["x0"] * (1 - 0.01 * params["x0"]),
        )
        method = get_async_benchmark_method()

        with self.subTest("Without early stopping"):
            experiment = self.run_optimization_with_orchestrator(
                problem=problem, method=method, seed=0
            )
            new_opt_trace = get_opt_trace_by_steps(experiment=experiment)

            self.assertEqual(
                list(new_opt_trace), [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
            )

        with self.subTest("With early stopping"):
            es_method = get_async_benchmark_method(
                early_stopping_strategy=ThresholdEarlyStoppingStrategy(
                    metric_threshold=10.0, min_progression=0, min_curves=2
                )
            )
            experiment = self.run_optimization_with_orchestrator(
                problem=problem, method=es_method, seed=0
            )
            new_opt_trace = get_opt_trace_by_steps(experiment=experiment)
            self.assertEqual(list(new_opt_trace), [0.0, 0.0, 1.0, 1.0, 2.0, 3.0])

        method = get_sobol_benchmark_method()
        with self.subTest("MOO"):
            problem = get_multi_objective_benchmark_problem()

            experiment = self.run_optimization_with_orchestrator(
                problem=problem, method=method, seed=0
            )
            with self.assertRaisesRegex(
                NotImplementedError, "only supported for single objective"
            ):
                get_opt_trace_by_steps(experiment=experiment)

        with self.subTest("Constrained"):
            problem = get_benchmark_problem("constrained_gramacy_observed_noise")
            experiment = self.run_optimization_with_orchestrator(
                problem=problem, method=method, seed=0
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "not supported for problems with outcome constraints",
            ):
                get_opt_trace_by_steps(experiment=experiment)

    def test_get_benchmark_result_with_cumulative_steps(self) -> None:
        """See test_get_opt_trace_by_cumulative_epochs for more info."""
        problem = get_async_benchmark_problem(
            map_data=True,
            n_steps=2,
            # Ensure we don't have two finishing at the same time, for
            # determinism
            step_runtime_fn=lambda params: params["x0"] * (1 - 0.01 * params["x0"]),
        )
        method = get_async_benchmark_method()
        result = self.benchmark_replication(problem=problem, method=method, seed=0)
        transformed = get_benchmark_result_with_cumulative_steps(
            result=result,
            optimal_value=problem.optimal_value,
            baseline_value=problem.baseline_value,
        )
        full_df = none_throws(result.experiment).lookup_data().full_df
        self.assertEqual(len(full_df), len(transformed.optimization_trace))
        self.assertEqual(
            list(transformed.optimization_trace),
            [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
        )
        self.assertTrue(np.isnan(transformed.oracle_trace).all())
        self.assertTrue(np.isnan(transformed.inference_trace).all())
        self.assertEqual(max(transformed.score_trace), max(result.score_trace))
        self.assertLessEqual(min(transformed.score_trace), min(result.score_trace))

    def test_get_best_parameters(self) -> None:
        """
        Whether this produces the correct values is tested more thoroughly in
        other tests such as `test_replication_with_inference_value` and
        `test_get_inference_trace_from_params`.  Setting up an experiment with
        data and trials without just running a benchmark is a pain, so in those
        tests, we just run a benchmark.
        """
        gs = get_sobol_generation_strategy()

        search_space = get_continuous_search_space(bounds=[(0, 1)])
        moo_config = get_moo_opt_config(outcome_names=["a", "b"], ref_point=[0, 0])
        experiment = Experiment(
            name="test",
            is_test=True,
            search_space=search_space,
            optimization_config=moo_config,
        )

        with (
            self.subTest("MOO not supported"),
            self.assertRaisesRegex(
                NotImplementedError, "Please use `get_pareto_optimal_parameters`"
            ),
        ):
            get_best_parameters(experiment=experiment, generation_strategy=gs)

        soo_config = get_soo_opt_config(outcome_names=["a"])
        with self.subTest("Empty experiment"):
            result = get_best_parameters(
                experiment=experiment.clone_with(optimization_config=soo_config),
                generation_strategy=gs,
            )
            self.assertIsNone(result)

        with self.subTest("All constraints violated"):
            experiment = get_experiment_with_observations(
                observations=[[1, -1], [2, -1]],
                constrained=True,
            )
            best_point = get_best_parameters(
                experiment=experiment, generation_strategy=gs
            )
            self.assertIsNone(best_point)

        with self.subTest("No completed trials"):
            experiment = get_experiment_with_observations(observations=[])
            sobol_generator = get_sobol(search_space=experiment.search_space)
            for _ in range(3):
                trial = experiment.new_trial(generator_run=sobol_generator.gen(n=1))
                trial.run()
            best_point = get_best_parameters(
                experiment=experiment, generation_strategy=gs
            )
            self.assertIsNone(best_point)

        experiment = get_experiment_with_observations(
            observations=[[1], [2]], constrained=False
        )
        with self.subTest("Working case"):
            best_point = get_best_parameters(
                experiment=experiment, generation_strategy=gs
            )
            self.assertEqual(best_point, experiment.trials[1].arms[0].parameters)

        with self.subTest("Trial indices"):
            best_point = get_best_parameters(
                experiment=experiment, generation_strategy=gs, trial_indices=[0]
            )
            self.assertEqual(best_point, experiment.trials[0].arms[0].parameters)

    def test_worst_feasible_value_validation(self) -> None:
        """Test validation logic for worst_feasible_value in BenchmarkProblem."""
        search_space = get_continuous_search_space(bounds=[(0, 1), (0, 1)])
        test_function = IdentityTestFunction(outcome_names=["objective", "constraint"])

        # MOO with constraints - must be 0.0
        moo_config = get_moo_opt_config(
            outcome_names=["objective", "constraint"],
            ref_point=[1.0],
            num_constraints=1,
        )
        BenchmarkProblem(
            name="moo_valid",
            optimization_config=moo_config,
            search_space=search_space,
            test_function=test_function,
            num_trials=4,
            baseline_value=5.0,
            optimal_value=10.0,
            worst_feasible_value=0.0,
        )
        with self.assertRaisesRegex(ValueError, "must be 0.0 for multi-objective"):
            BenchmarkProblem(
                name="moo_invalid",
                optimization_config=moo_config,
                search_space=search_space,
                test_function=test_function,
                num_trials=4,
                baseline_value=5.0,
                optimal_value=10.0,
                worst_feasible_value=5.0,
            )

        # SOO with constraints, `worst_feasible_value` must be provided
        soo_min_config = get_soo_opt_config(
            outcome_names=["objective", "constraint"], lower_is_better=True
        )
        soo_max_config = get_soo_opt_config(
            outcome_names=["objective", "constraint"], lower_is_better=False
        )
        with self.assertRaisesRegex(
            ValueError, "must be provided for constrained problems"
        ):
            BenchmarkProblem(
                name="none_invalid",
                optimization_config=soo_min_config,
                search_space=search_space,
                test_function=test_function,
                num_trials=4,
                baseline_value=10.0,
                optimal_value=5.0,
            )

        # Minimization: worst_feasible >= optimal
        with self.assertRaisesRegex(ValueError, "must be greater than or equal to"):
            BenchmarkProblem(
                name="min_invalid",
                optimization_config=soo_min_config,
                search_space=search_space,
                test_function=test_function,
                num_trials=4,
                baseline_value=10.0,
                optimal_value=5.0,
                worst_feasible_value=3.0,
            )
        BenchmarkProblem(
            name="min_valid",
            optimization_config=soo_min_config,
            search_space=search_space,
            test_function=test_function,
            num_trials=4,
            baseline_value=10.0,
            optimal_value=5.0,
            worst_feasible_value=8.0,
        )

        # Maximization: worst_feasible <= optimal
        with self.assertRaisesRegex(ValueError, "must be less than or equal to"):
            BenchmarkProblem(
                name="max_invalid",
                optimization_config=soo_max_config,
                search_space=search_space,
                test_function=test_function,
                num_trials=4,
                baseline_value=5.0,
                optimal_value=10.0,
                worst_feasible_value=15.0,
            )
        BenchmarkProblem(
            name="max_valid",
            optimization_config=soo_max_config,
            search_space=search_space,
            test_function=test_function,
            num_trials=4,
            baseline_value=5.0,
            optimal_value=10.0,
            worst_feasible_value=8.0,
        )

        # No constraints - validation skipped
        no_constraint_config = get_soo_opt_config(outcome_names=["objective"])
        no_constraint_test_function = IdentityTestFunction(outcome_names=["objective"])
        BenchmarkProblem(
            name="no_constraints",
            optimization_config=no_constraint_config,
            search_space=search_space,
            test_function=no_constraint_test_function,
            num_trials=4,
            baseline_value=10.0,
            optimal_value=5.0,
            worst_feasible_value=None,
        )

    def test_get_benchmark_result_from_experiment_and_gs(self) -> None:
        problem = get_single_objective_benchmark_problem()
        method = BenchmarkMethod(
            name="Sobol", generation_strategy=get_sobol_generation_strategy()
        )
        seed = 0
        result = self.benchmark_replication(
            problem=problem, method=method, seed=seed, strip_runner_before_saving=False
        )

        result2 = get_benchmark_result_from_experiment_and_gs(
            experiment=none_throws(result.experiment),
            generation_strategy=method.generation_strategy,
            problem=problem,
            seed=seed,
            strip_runner_before_saving=False,
        )
        # Idempotency
        self.assertEqual(result, result2)
        # Runner not stripped
        self.assertIsNotNone(none_throws(result2.experiment).runner)
        self.assertEqual(result2.seed, seed)
        # With report_inference_value_as_trace=False, inference trace is all null
        self.assertTrue(np.isnan(result.inference_trace).all())
        self.assertFalse(np.isnan(result.optimization_trace).any())
        self.assertTrue(np.array_equal(result.oracle_trace, result.optimization_trace))

        with self.subTest("runner stripped"):
            result_no_runner = get_benchmark_result_from_experiment_and_gs(
                experiment=none_throws(result.experiment),
                generation_strategy=method.generation_strategy,
                problem=problem,
                seed=3,
            )
            self.assertIsNone(none_throws(result_no_runner.experiment).runner)

        with self.subTest("inference value as trace"):
            problem.report_inference_value_as_trace = True
            experiment = run_optimization_with_orchestrator(
                problem=problem, method=method, seed=seed
            )
            result_inf = get_benchmark_result_from_experiment_and_gs(
                experiment=experiment,
                generation_strategy=method.generation_strategy,
                problem=problem,
                seed=seed,
            )
            self.assertFalse(np.isnan(result_inf.inference_trace).any())
            self.assertTrue(
                np.array_equal(
                    result_inf.inference_trace, result_inf.optimization_trace
                )
            )

    def test_scoring_constrained_problem(self) -> None:
        # Make sure the score is computed correctly for a constrained problem
        expected_score_traces_oracle = [
            [0.0, 0.0, 51.9295, 51.9295, 51.9295],
            [-20.8965, -20.8965, 41.8845, 41.8845, 41.8845],
        ]
        expected_score_traces_inference = [
            [0.0, 0.0, 51.9295, 0.0, 0.0],
            [-20.8965, -20.8965, 41.8845, -20.8965, -20.8965],
        ]
        expected_is_feasible_trace = [False, False, True, False, False]
        worst_feasible_value = PressureVessel().worst_feasible_value
        feasible_value = 118769.1124  # Value of the only feasible trial
        expected_oracle_trace = [
            worst_feasible_value,
            worst_feasible_value,
            feasible_value,
            feasible_value,
            feasible_value,
        ]
        expected_inference_trace = [
            worst_feasible_value,
            worst_feasible_value,
            feasible_value,  # The only feasible trial
            worst_feasible_value,
            worst_feasible_value,
        ]

        for (
            baseline_value,
            report_inference_value_as_trace,
            num_trials,
        ) in itertools.product([float("inf"), 200_000], [False, True], [2, 5]):
            sobol = BenchmarkMethod(generation_strategy=get_sobol_generation_strategy())
            problem = create_problem_from_botorch(
                test_problem_class=PressureVessel,
                test_problem_kwargs={},
                num_trials=num_trials,
                baseline_value=baseline_value,
                report_inference_value_as_trace=report_inference_value_as_trace,
            )
            results = benchmark_replication(problem=problem, method=sobol, seed=2)

            # Check that the traces are what we expect
            idx = 0 if baseline_value == float("inf") else 1
            expected = (
                expected_score_traces_inference
                if report_inference_value_as_trace
                else expected_score_traces_oracle
            )
            self.assertTrue(
                np.allclose(
                    results.score_trace,
                    expected[idx][:num_trials],
                    atol=1e-3,
                )
            )
            self.assertTrue(
                np.allclose(
                    results.oracle_trace,
                    expected_oracle_trace[:num_trials],
                    atol=1e-3,
                )
            )
            self.assertTrue(
                np.equal(
                    none_throws(results.is_feasible_trace),
                    expected_is_feasible_trace[:num_trials],
                ).all()
            )
            if report_inference_value_as_trace:
                self.assertTrue(
                    np.allclose(
                        results.inference_trace,
                        expected_inference_trace[:num_trials],
                        atol=1e-3,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        results.optimization_trace,
                        expected_inference_trace[:num_trials],
                        atol=1e-3,
                    )
                )
            else:
                self.assertTrue(np.isnan(none_throws(results.inference_trace)).all())
                self.assertTrue(
                    np.allclose(
                        results.optimization_trace,
                        expected_oracle_trace[:num_trials],
                        atol=1e-3,
                    )
                )
