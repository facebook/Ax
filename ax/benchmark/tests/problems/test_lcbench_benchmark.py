#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.benchmark.problems.surrogate.lcbench.transfer_learning import (
    DEFAULT_AND_OPTIMAL_VALUES,
    get_lcbench_benchmark_problem,
)
from ax.core.observation import ObservationFeatures
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


class TestLCBenchBenchmark(TestCase):
    @TestCase.ax_long_test(reason="Training random forest regressor")
    def test_lcbench_predictions(self) -> None:
        self.assertEqual(len(DEFAULT_AND_OPTIMAL_VALUES), 22)
        # NOTE: lots of tasks, so testing only one here o/w this is very slow
        dataset_name = "car"
        problem = get_lcbench_benchmark_problem(
            dataset_name=dataset_name,
            num_trials=32,
        )
        test_function = assert_is_instance(problem.test_function, SurrogateTestFunction)
        metric = assert_is_instance(
            problem.optimization_config.objective.metric, BenchmarkMetric
        )
        self.assertFalse(metric.observe_noise_sd)
        self.assertEqual(problem.num_trials, 32)
        default_val, opt_val = DEFAULT_AND_OPTIMAL_VALUES[dataset_name]
        self.assertAlmostEqual(
            float(problem.optimal_value),
            opt_val,
            places=4,
        )
        surrogate = test_function.surrogate

        # Predict for arm 0_0 and make sure it matches the expected value
        training_data = surrogate.get_training_data()
        features = training_data.arm_data.loc[0, "0_0"].to_dict()
        pred, _ = surrogate.predict(
            observation_features=[ObservationFeatures(parameters=features)]
        )
        self.assertAlmostEqual(pred["Train/val_accuracy"][0], default_val, places=3)
