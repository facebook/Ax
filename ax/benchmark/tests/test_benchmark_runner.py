#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from contextlib import nullcontext
from dataclasses import replace
from itertools import product
from unittest.mock import Mock, patch

import numpy as np

import torch
from ax.benchmark.benchmark_runner import _add_noise, BenchmarkRunner
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction

from ax.benchmark.benchmark_test_functions.synthetic import IdentityTestFunction
from ax.benchmark.problems.synthetic.hss.jenatton import (
    get_jenatton_benchmark_problem,
    Jenatton,
)
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    DummyTestFunction,
    get_jenatton_trials,
    get_soo_surrogate_test_function,
)

from botorch.test_functions.synthetic import Ackley, ConstrainedHartmann, Hartmann
from botorch.utils.transforms import normalize
from pandas import DataFrame
from pyre_extensions import assert_is_instance, none_throws


class TestBenchmarkRunner(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    def test_simulated_backend_runner(self) -> None:
        # Initialize
        runner = BenchmarkRunner(
            test_function=Jenatton(outcome_names=["objective"]),
            step_runtime_function=lambda params: params["x1"] + 1,
            max_concurrency=2,
        )
        simulated_backend_runner = none_throws(runner.simulated_backend_runner)
        simulator = simulated_backend_runner.simulator
        self.assertEqual(simulator.max_concurrency, 2)
        self.assertTrue(simulator.use_internal_clock)
        self.assertEqual(simulator.time, 0)
        trials = get_jenatton_trials(n_trials=3)

        # Run trials in parallel
        runner.run(trial=trials[0])
        runner.run(trial=trials[1])
        self.assertEqual(simulator.time, 0)

        # Check status (updating clock)
        statuses = runner.poll_trial_status(trials=[trials[0], trials[1]])
        self.assertEqual(simulator.time, 1)
        self.assertEqual(statuses[TrialStatus.COMPLETED], {0})
        self.assertEqual(statuses[TrialStatus.RUNNING], {1})

        # Run another trial
        runner.run(trial=trials[2])
        self.assertEqual(simulator.time, 1)
        statuses = runner.poll_trial_status(trials=list(trials.values()))
        self.assertEqual(simulator.time, 2)
        self.assertEqual(statuses[TrialStatus.COMPLETED], {0, 1})
        self.assertEqual(statuses[TrialStatus.RUNNING], {2})

    def test_runner(self) -> None:
        botorch_cases = [
            (
                BoTorchTestFunction(
                    botorch_problem=test_problem_class(dim=6),
                    modified_bounds=modified_bounds,
                    outcome_names=outcome_names,
                ),
                noise_std,
                num_outcomes,
            )
            for (test_problem_class, num_outcomes, outcome_names) in (
                (Hartmann, 1, ["objective_0"]),
                (ConstrainedHartmann, 2, ["objective_0", "constraint"]),
            )
            for modified_bounds, noise_std in product(
                (None, [(0.0, 2.0)] * 6),
                (0.0, [0.1] * num_outcomes),
            )
        ]
        param_based_cases = [
            (
                DummyTestFunction(dim=6, num_outcomes=num_outcomes),
                noise_std,
                num_outcomes,
            )
            for num_outcomes in (1, 2)
            for noise_std in (0.0, [float(i) for i in range(num_outcomes)])
        ]
        surrogate_cases = [
            (get_soo_surrogate_test_function(lazy=False), noise_std, 1)
            for noise_std in (0.0, 1.0, [0.0], [1.0])
        ]
        for test_function, noise_std, num_outcomes in (
            botorch_cases + param_based_cases + surrogate_cases
        ):
            # Set up outcome names
            if isinstance(test_function, BoTorchTestFunction):
                if isinstance(test_function.botorch_problem, ConstrainedHartmann):
                    outcome_names = ["objective_0", "constraint"]
                else:
                    outcome_names = ["objective_0"]
            elif isinstance(test_function, DummyTestFunction):
                outcome_names = [f"objective_{i}" for i in range(num_outcomes)]
            else:  # SurrogateTestFunction
                outcome_names = ["branin"]

            # Set up runner
            runner = BenchmarkRunner(test_function=test_function, noise_std=noise_std)

            test_description = f"{test_function=}, {noise_std=}"
            with self.subTest(
                f"Test basic construction, {test_function=}, {noise_std=}"
            ):
                self.assertIs(runner.test_function, test_function)
                self.assertEqual(runner.outcome_names, outcome_names)
                if isinstance(noise_std, list):
                    self.assertEqual(
                        runner.get_noise_stds(),
                        dict(zip(runner.outcome_names, noise_std)),
                    )
                else:  # float
                    self.assertEqual(
                        runner.get_noise_stds(),
                        {name: noise_std for name in runner.outcome_names},
                    )

                # check equality
                new_runner = replace(
                    runner,
                    test_function=BoTorchTestFunction(
                        botorch_problem=Ackley(),
                        outcome_names=test_function.outcome_names,
                    ),
                )
                self.assertNotEqual(runner, new_runner)

                self.assertEqual(runner, runner)
                if isinstance(test_function, BoTorchTestFunction):
                    self.assertEqual(
                        test_function.botorch_problem.bounds.dtype, torch.double
                    )

            is_botorch = isinstance(test_function, BoTorchTestFunction)
            with self.subTest(f"test `get_Y_true()`, {test_description}"):
                dim = 6 if is_botorch else 9
                X = torch.rand(1, dim, dtype=torch.double)
                param_names = (
                    [f"x{i}" for i in range(6)]
                    if is_botorch
                    else list(
                        get_jenatton_benchmark_problem().search_space.parameters.keys()
                    )
                )
                params = dict(zip(param_names, (x.item() for x in X.unbind(-1))))

                with (
                    nullcontext()
                    if not isinstance(test_function, SurrogateTestFunction)
                    else patch.object(
                        # pyre-fixme: BenchmarkTestFunction` has no attribute
                        # `_surrogate`.
                        runner.test_function._surrogate,
                        "predict",
                        return_value=({"branin": [4.2]}, None),
                    )
                ):
                    Y = runner.get_Y_true(params=params)

                if (
                    isinstance(test_function, BoTorchTestFunction)
                    and test_function.modified_bounds is not None
                ):
                    X_tf = normalize(
                        X,
                        torch.tensor(
                            test_function.modified_bounds, dtype=torch.double
                        ).T,
                    )
                else:
                    X_tf = X
                if isinstance(test_function, BoTorchTestFunction):
                    botorch_problem = test_function.botorch_problem
                    obj = botorch_problem.evaluate_true(X_tf)
                    if isinstance(botorch_problem, ConstrainedHartmann):
                        expected_Y = np.array(
                            [
                                [obj.item()],
                                [botorch_problem.evaluate_slack(X_tf).item()],
                            ]
                        )
                    else:
                        expected_Y = obj.numpy()[:, None]
                        self.assertEqual(expected_Y.ndim, 2)
                elif isinstance(test_function, SurrogateTestFunction):
                    expected_Y = np.array([[4.2]])
                else:
                    expected_Y = np.full((2, 1), X.pow(2).sum().item())
                self.assertTrue(np.allclose(Y, expected_Y))

            with self.subTest(f"test `run()`, {test_description}"):
                trial = Mock(spec=Trial)
                arm = Arm(name="0_0", parameters=params)
                trial.arms = [arm]
                trial.arm = arm
                trial.index = 0

                with (
                    nullcontext()
                    if not isinstance(test_function, SurrogateTestFunction)
                    else patch.object(
                        runner.test_function._surrogate,
                        "predict",
                        return_value=({"branin": [4.2]}, None),
                    )
                ):
                    res = runner.run(trial=trial)["benchmark_metadata"].dfs
                df = next(iter(res.values()))
                self.assertEqual({"0_0"}, set(df["arm_name"]))
                self.assertEqual(
                    set(runner.test_function.outcome_names), set(res.keys())
                )

                for i, df in enumerate(res.values()):
                    if isinstance(noise_std, list):
                        self.assertEqual(df["sem"].item(), noise_std[i])
                        if all((n == 0 for n in noise_std)):
                            self.assertTrue(np.array_equal(df["mean"], Y[i, :]))
                    else:  # float
                        self.assertEqual(df["sem"].item(), noise_std)
                        if noise_std == 0:
                            self.assertTrue(np.array_equal(df["mean"], Y[i, :]))

            with self.subTest(f"test `poll_trial_status()`, {test_description}"):
                self.assertEqual(
                    {TrialStatus.COMPLETED: {0}}, runner.poll_trial_status([trial])
                )

            with self.subTest(f"test `serialize_init_args()`, {test_description}"):
                with self.assertRaisesRegex(
                    UnsupportedError, "serialize_init_args is not a supported method"
                ):
                    BenchmarkRunner.serialize_init_args(obj=runner)
                with self.assertRaisesRegex(
                    UnsupportedError, "deserialize_init_args is not a supported method"
                ):
                    BenchmarkRunner.deserialize_init_args({})

    def test__add_noise(self) -> None:
        np.random.seed(0)
        y_true = np.arange(6)
        arm_name = ["0_0", "0_1", "0_0", "0_1", "0_0", "0_1"]
        metric_name = ["foo", "foo", "bar", "bar", "baz", "baz"]

        df = DataFrame(
            {"Y_true": y_true, "metric_name": metric_name, "arm_name": arm_name}
        )

        noise_stds = {"foo": 1, "bar": 2, "baz": 3}
        arm_weights = {"0_0": 1, "0_1": 2}
        result = _add_noise(df=df, noise_stds=noise_stds, arm_weights=arm_weights)
        self.assertEqual(set(result.columns), set(df.columns) | {"mean", "sem"})
        expected_sem = df["metric_name"].map(noise_stds) / np.sqrt(
            df["arm_name"].map(arm_weights) / 3
        )
        self.assertEqual(result["sem"].tolist(), expected_sem.tolist())
        noise = df["mean"] - df["Y_true"]
        self.assertNotEqual(noise.std(), 0)

        z_scores = noise / expected_sem
        self.assertNotEqual(z_scores.std(), 0)

        chi_squared_stat = (z_scores**2).sum()
        # None of these assertions would have failed in 10M simulations.
        # Each has some tolerance from the most extreme value seen in 10M sims.
        self.assertGreater(chi_squared_stat, 0.005)
        self.assertLess(chi_squared_stat, 45)
        self.assertLess(np.abs(z_scores).min(), 2)
        self.assertGreater(z_scores.max(), 0.05)

    def test_heterogeneous_noise(self) -> None:
        outcome_names = ["objective_0", "constraint"]
        noise_dict = {"objective_0": 0.1, "constraint": 0.05}
        for noise_std in [[0.1, 0.05], noise_dict]:
            runner = BenchmarkRunner(
                test_function=BoTorchTestFunction(
                    botorch_problem=ConstrainedHartmann(dim=6),
                    outcome_names=outcome_names,
                ),
                noise_std=noise_std,
            )
            self.assertDictEqual(
                assert_is_instance(runner.get_noise_stds(), dict), noise_dict
            )

            X = torch.rand(1, 6, dtype=torch.double)
            arm = Arm(
                name="0_0",
                parameters={f"x{i}": x.item() for i, x in enumerate(X.unbind(-1))},
            )
            trial = Mock(spec=Trial)
            trial.arms = [arm]
            trial.arm = arm
            trial.index = 0
            res = runner.run(trial=trial)["benchmark_metadata"].dfs
            self.assertEqual(
                set(outcome_names),
                res.keys(),
            )
            obj_df = res["objective_0"]
            self.assertEqual(len(obj_df), 1)
            self.assertEqual(
                {
                    "arm_name",
                    "metric_name",
                    "mean",
                    "sem",
                    "trial_index",
                    "step",
                    "virtual runtime",
                },
                set(obj_df.columns),
            )
            self.assertEqual(obj_df["arm_name"].item(), "0_0")
            self.assertEqual(obj_df["sem"].item(), 0.1)
            self.assertEqual(res["constraint"]["sem"].item(), 0.05)

            with self.subTest("heterogeneous arm weights"):
                arm_0 = Arm(name="0_0", parameters={"x0": 0.0})
                arm_1 = Arm(name="0_1", parameters={"x0": 2.0})
                trial = Mock(spec=BatchTrial)
                trial.arms = [arm_0, arm_1]
                trial.index = 0
                # arm stds get multiplied by sqrt([25/9, 25/16])
                trial.arm_weights = {arm_0: 9, arm_1: 16}
                res = runner.run(trial=trial)["benchmark_metadata"].dfs
                expected_relative_noise_levels = np.array([5 / 3, 5 / 4])
                for metric_name, df in res.items():
                    self.assertTrue(
                        np.array_equal(
                            df["sem"],
                            noise_dict[metric_name] * expected_relative_noise_levels,
                        )
                    )

    def test_with_learning_curve(self) -> None:
        test_function = IdentityTestFunction(outcome_names=["foo", "bar"], n_steps=10)

        params = {"x0": 1.2}
        runner = BenchmarkRunner(test_function=test_function, noise_std=0.0)

        experiment = Experiment(
            name="test",
            is_test=True,
            runner=runner,
            search_space=Mock(spec=SearchSpace),
        )

        for noise_std in [0.0, 0.1]:
            with self.subTest(noise_std=noise_std):
                runner = BenchmarkRunner(
                    test_function=test_function, noise_std=noise_std
                )

                trial = Trial(experiment=experiment)
                arm = Arm(name="0_0", parameters=params)
                trial.add_arm(arm=arm)
                metadata_dict = runner.run(trial=trial)
                self.assertEqual({"benchmark_metadata"}, metadata_dict.keys())
                metadata = metadata_dict["benchmark_metadata"].dfs
                self.assertEqual(set(metadata.keys()), set(test_function.outcome_names))
                for df in metadata.values():
                    self.assertEqual(len(df), 10)
                    self.assertTrue((df["arm_name"] == "0_0").all())
                    self.assertTrue(np.array_equal(df["step"], np.arange(10)))
                    self.assertTrue((df["sem"] == noise_std).all())

                noiseless = test_function.evaluate_true(params=params)
                all_close = np.allclose(metadata["foo"]["mean"], noiseless[0, :])
                self.assertEqual(all_close, noise_std == 0.0)

        with self.subTest("with SimulatedBackendRunner"):
            runner = BenchmarkRunner(
                test_function=test_function, noise_std=0.0, max_concurrency=2
            )

            arm = Arm(name="0_0", parameters=params)
            trial = Trial(experiment=experiment)
            trial.add_arm(arm=arm)
            metadata = runner.run(trial=trial)["benchmark_metadata"]
            backend_simulator = none_throws(metadata.backend_simulator)
            self.assertNotIn(trial.index, backend_simulator._completed)
            sim_trial = none_throws(
                backend_simulator.get_sim_trial_by_index(trial.index)
            )
            self.assertIsNone(sim_trial.sim_completed_time)
            self.assertEqual(sim_trial.sim_start_time, 0)
            self.assertEqual(backend_simulator.time, 0)

    def test_heterogeneous_step_runtime(self) -> None:
        n_steps = 10
        test_function = IdentityTestFunction(
            outcome_names=["foo", "bar"], n_steps=n_steps
        )
        runner = BenchmarkRunner(
            test_function=test_function,
            noise_std=0.0,
            step_runtime_function=lambda params: params["x0"],
        )
        experiment = Experiment(
            name="test",
            is_test=True,
            runner=runner,
            search_space=Mock(spec=SearchSpace),
        )
        trial = BatchTrial(experiment=experiment)
        arm_0_step_time = 0.5
        arm_1_step_time = 1.5
        trial.add_arm(Arm(name="0_0", parameters={"x0": arm_0_step_time}))
        trial.add_arm(Arm(name="0_1", parameters={"x0": arm_1_step_time}))
        df = runner.run(trial=trial)["benchmark_metadata"].dfs["foo"]
        total_runtime = df.groupby("arm_name")["virtual runtime"].max()
        self.assertEqual(
            total_runtime.to_dict(),
            {"0_0": arm_0_step_time * n_steps, "0_1": arm_1_step_time * n_steps},
        )
        max_step = df.groupby("arm_name")["step"].max()
        self.assertEqual(max_step.to_list(), [9, 9])

        with self.subTest("Test runtimes non-negative"):
            trial = BatchTrial(experiment=experiment)
            trial.add_arm(Arm(name="0_0", parameters={"x0": -1}))
            with self.assertRaisesRegex(
                ValueError, "Step duration must be non-negative"
            ):
                runner.run(trial=trial)

    def test_wrong_noise_std_keys(self) -> None:
        test_function = IdentityTestFunction(outcome_names=["foo", "bar"])
        runner = BenchmarkRunner(test_function=test_function, noise_std={"alpaca": 4})
        with self.assertRaisesRegex(
            ValueError, "Noise std must have keys equal to outcome names"
        ):
            runner.get_noise_stds()
