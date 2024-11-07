# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from random import random

from ax.benchmark.benchmark_metric import BenchmarkMetric

from ax.benchmark.problems.synthetic.hss.jenatton import (
    get_jenatton_benchmark_problem,
    jenatton_test_function,
)
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
            self.assertAlmostEqual(
                # pyre-fixme: Incompatible parameter type [6]: In call
                # `jenatton_test_function`, for 1st positional argument,
                # expected `Optional[float]` but got `Union[None, bool, float,
                # int, str]`.
                jenatton_test_function(**params),
                value,
            )
            self.assertAlmostEqual(
                benchmark_problem.test_function.evaluate_true(params=params).item(),
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
        self.assertEqual(problem.noise_std, 0.0)
        self.assertFalse(assert_is_instance(metric, BenchmarkMetric).observe_noise_sd)

        problem = get_jenatton_benchmark_problem(
            num_trials=10, noise_std=0.1, observe_noise_sd=True
        )
        objective = problem.optimization_config.objective
        metric = objective.metric
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(problem.noise_std, 0.1)
        self.assertTrue(assert_is_instance(metric, BenchmarkMetric).observe_noise_sd)
