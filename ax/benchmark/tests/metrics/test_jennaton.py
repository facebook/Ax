# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from random import random
from unittest import mock

from ax.benchmark.metrics.jenatton import jenatton_test_function, JenattonMetric
from ax.core.arm import Arm
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase


class JenattonMetricTest(TestCase):

    def test_jenatton_test_function(self) -> None:
        rand_params = {f"x{i}": random() for i in range(4, 8)}
        rand_params["r8"] = random()
        rand_params["r9"] = random()

        for x3 in (0, 1):
            self.assertAlmostEqual(
                jenatton_test_function(
                    x1=0,
                    x2=0,
                    x3=x3,
                    **{**rand_params, "x4": 2.0, "r8": 0.05},
                ),
                4.15,
            )
            self.assertAlmostEqual(
                jenatton_test_function(
                    x1=0,
                    x2=1,
                    x3=x3,
                    **{**rand_params, "x5": 2.0, "r8": 0.05},
                ),
                4.25,
            )
        for x2 in (0, 1):
            self.assertAlmostEqual(
                jenatton_test_function(
                    x1=1,
                    x2=x2,
                    x3=0,
                    **{**rand_params, "x6": 2.0, "r9": 0.05},
                ),
                4.35,
            )
            self.assertAlmostEqual(
                jenatton_test_function(
                    x1=1,
                    x2=x2,
                    x3=1,
                    **{**rand_params, "x7": 2.0, "r9": 0.05},
                ),
                4.45,
            )

    def test_init(self) -> None:
        metric = JenattonMetric()
        self.assertEqual(metric.name, "jenatton")
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(metric.noise_std, 0.0)
        self.assertFalse(metric.observe_noise_sd)
        metric = JenattonMetric(name="nottanej", noise_std=0.1, observe_noise_sd=True)
        self.assertEqual(metric.name, "nottanej")
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(metric.noise_std, 0.1)
        self.assertTrue(metric.observe_noise_sd)

    def test_fetch_trial_data(self) -> None:
        arm = mock.Mock(spec=Arm)
        arm.parameters = {"x1": 0, "x2": 1, "x5": 2.0, "r8": 0.05}
        trial = mock.Mock(spec=Trial)
        trial.arms_by_name = {"0_0": arm}
        trial.index = 0

        metric = JenattonMetric()
        df = metric.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df), 1)
        res_dict = df.iloc[0].to_dict()
        self.assertEqual(res_dict["arm_name"], "0_0")
        self.assertEqual(res_dict["metric_name"], "jenatton")
        self.assertEqual(res_dict["mean"], 4.25)
        self.assertTrue(math.isnan(res_dict["sem"]))
        self.assertEqual(res_dict["trial_index"], 0)

        metric = JenattonMetric(name="nottanej", noise_std=0.1, observe_noise_sd=True)
        df = metric.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df), 1)
        res_dict = df.iloc[0].to_dict()
        self.assertEqual(res_dict["arm_name"], "0_0")
        self.assertEqual(res_dict["metric_name"], "nottanej")
        self.assertNotEqual(res_dict["mean"], 4.25)
        self.assertEqual(res_dict["sem"], 0.1)
        self.assertEqual(res_dict["trial_index"], 0)

    def test_make_ground_truth_metric(self) -> None:
        metric = JenattonMetric()
        gt_metric = metric.make_ground_truth_metric()
        self.assertIsInstance(gt_metric, JenattonMetric)
        self.assertEqual(gt_metric.noise_std, 0.0)
        self.assertFalse(gt_metric.observe_noise_sd)
        metric = JenattonMetric(noise_std=0.1, observe_noise_sd=True)
        gt_metric = metric.make_ground_truth_metric()
        self.assertIsInstance(gt_metric, JenattonMetric)
        self.assertEqual(gt_metric.noise_std, 0.0)
        self.assertFalse(gt_metric.observe_noise_sd)
