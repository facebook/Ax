# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from random import random

from ax.benchmark.metrics.benchmark import BenchmarkMetric, GroundTruthBenchmarkMetric

from ax.benchmark.metrics.jenatton import jenatton_test_function
from ax.benchmark.problems.synthetic.hss.jenatton import get_jenatton_benchmark_problem
from ax.benchmark.runners.base import BenchmarkRunner
from ax.benchmark.runners.botorch_test import ParamBasedTestProblemRunner
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.trial import Trial
from ax.core.types import TParameterization
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


class JenattonTest(TestCase):

    def test_jenatton_test_function(self) -> None:
        benchmark_problem = get_jenatton_benchmark_problem()

        rand_params = {f"x{i}": random() for i in range(4, 8)}
        rand_params["r8"] = random()
        rand_params["r9"] = random()

        cases: list[tuple[TParameterization, float]] = []

        for x3 in (0, 1):
            # list of (param dict, expected value)
            cases.append(
                (
                    {
                        "x1": 0,
                        "x2": 0,
                        "x3": x3,
                        **{**rand_params, "x4": 2.0, "r8": 0.05},
                    },
                    4.15,
                ),
            )
            cases.append(
                (
                    {
                        "x1": 0,
                        "x2": 1,
                        "x3": x3,
                        **{**rand_params, "x5": 2.0, "r8": 0.05},
                    },
                    4.25,
                )
            )

        for x2 in (0, 1):
            cases.append(
                (
                    {
                        "x1": 1,
                        "x2": x2,
                        "x3": 0,
                        **{**rand_params, "x6": 2.0, "r9": 0.05},
                    },
                    4.35,
                )
            )
            cases.append(
                (
                    {
                        "x1": 1,
                        "x2": x2,
                        "x3": 1,
                        **{**rand_params, "x7": 2.0, "r9": 0.05},
                    },
                    4.45,
                )
            )

        for params, value in cases:
            arm = Arm(parameters=params)
            self.assertAlmostEqual(
                # pyre-fixme: Incompatible parameter type [6]: In call
                # `jenatton_test_function`, for 1st positional argument,
                # expected `Optional[float]` but got `Union[None, bool, float,
                # int, str]`.
                jenatton_test_function(**params),
                value,
            )
            self.assertAlmostEqual(
                assert_is_instance(benchmark_problem.runner, BenchmarkRunner)
                .get_Y_true(arm)
                .item(),
                value,
                places=6,
            )

    def test_create_problem(self) -> None:
        problem = get_jenatton_benchmark_problem()
        objective = problem.optimization_config.objective
        metric = objective.metric

        self.assertEqual(metric.name, "Jenatton")
        self.assertTrue(objective.minimize)
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(
            assert_is_instance(
                problem.runner, ParamBasedTestProblemRunner
            ).test_problem.noise_std,
            0.0,
        )
        self.assertTrue(problem.is_noiseless)
        self.assertFalse(assert_is_instance(metric, BenchmarkMetric).observe_noise_sd)

        problem = get_jenatton_benchmark_problem(
            num_trials=10, noise_std=0.1, observe_noise_sd=True
        )
        objective = problem.optimization_config.objective
        metric = objective.metric
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(
            assert_is_instance(
                problem.runner, ParamBasedTestProblemRunner
            ).test_problem.noise_std,
            0.1,
        )
        self.assertFalse(problem.is_noiseless)
        self.assertTrue(assert_is_instance(metric, BenchmarkMetric).observe_noise_sd)

    def test_fetch_trial_data(self) -> None:
        problem = get_jenatton_benchmark_problem()
        arm = Arm(parameters={"x1": 0, "x2": 1, "x5": 2.0, "r8": 0.05}, name="0_0")

        experiment = Experiment(
            search_space=problem.search_space,
            name="Jenatton",
            optimization_config=problem.optimization_config,
        )

        trial = Trial(experiment=experiment)
        trial.add_arm(arm)
        metadata = problem.runner.run(trial=trial)
        trial.update_run_metadata(metadata)

        expected_metadata = {
            "Ys": {"0_0": [4.25]},
            "Ystds": {"0_0": [0.0]},
            "outcome_names": ["Jenatton"],
            "Ys_true": {"0_0": [4.25]},
        }
        self.assertEqual(metadata, expected_metadata)

        metric = problem.optimization_config.objective.metric

        df = assert_is_instance(metric.fetch_trial_data(trial=trial).value, Data).df
        self.assertEqual(len(df), 1)
        res_dict = df.iloc[0].to_dict()
        self.assertEqual(res_dict["arm_name"], "0_0")
        self.assertEqual(res_dict["metric_name"], "Jenatton")
        self.assertEqual(res_dict["mean"], 4.25)
        self.assertTrue(math.isnan(res_dict["sem"]))
        self.assertEqual(res_dict["trial_index"], 0)

        problem = get_jenatton_benchmark_problem(noise_std=0.1, observe_noise_sd=True)
        experiment = Experiment(
            search_space=problem.search_space,
            name="Jenatton",
            optimization_config=problem.optimization_config,
        )

        trial = Trial(experiment=experiment)
        trial.add_arm(arm)
        metadata = problem.runner.run(trial=trial)
        trial.update_run_metadata(metadata)

        metric = problem.optimization_config.objective.metric
        df = assert_is_instance(metric.fetch_trial_data(trial=trial).value, Data).df
        self.assertEqual(len(df), 1)
        res_dict = df.iloc[0].to_dict()
        self.assertEqual(res_dict["arm_name"], "0_0")
        self.assertNotEqual(res_dict["mean"], 4.25)
        self.assertAlmostEqual(res_dict["sem"], 0.1)
        self.assertEqual(res_dict["trial_index"], 0)

    def test_make_ground_truth_metric(self) -> None:
        problem = get_jenatton_benchmark_problem()

        arm = Arm(parameters={"x1": 0, "x2": 1, "x5": 2.0, "r8": 0.05}, name="0_0")

        experiment = Experiment(
            search_space=problem.search_space,
            name="Jenatton",
            optimization_config=problem.optimization_config,
        )

        trial = Trial(experiment=experiment)
        trial.add_arm(arm)
        problem.runner.run(trial=trial)
        metadata = problem.runner.run(trial=trial)
        trial.update_run_metadata(metadata)

        metric = assert_is_instance(
            problem.optimization_config.objective.metric, BenchmarkMetric
        )
        gt_metric = metric.make_ground_truth_metric()
        self.assertIsInstance(gt_metric, GroundTruthBenchmarkMetric)
        runner = assert_is_instance(problem.runner, ParamBasedTestProblemRunner)
        self.assertEqual(runner.test_problem.noise_std, 0.0)
        self.assertFalse(
            assert_is_instance(gt_metric, BenchmarkMetric).observe_noise_sd
        )

        self.assertIsInstance(metric, BenchmarkMetric)
        self.assertNotIsInstance(metric, GroundTruthBenchmarkMetric)
        self.assertEqual(runner.test_problem.noise_std, 0.0)
        self.assertFalse(metric.observe_noise_sd)
