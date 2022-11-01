#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial


class GenericNoisyFunctionMetricTest(TestCase):
    def testGenericNoisyFunctionMetric(self) -> None:
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def f(params):
            return params["x"] + 1.0

        # noiseless
        metric = GenericNoisyFunctionMetric(
            name="test_metric",
            f=f,
        )
        trial = get_trial()
        df = metric.fetch_trial_data(trial).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        # pyre-fixme[16]: Optional type has no attribute `parameters`.
        self.assertEqual(df["mean"].tolist(), [trial.arm.parameters["x"] + 1.0])
        self.assertEqual(df["sem"].tolist(), [0.0])

        # noisy
        metric = GenericNoisyFunctionMetric(
            name="test_metric",
            f=f,
            noise_sd=1.0,
        )
        trial = get_trial()
        df = metric.fetch_trial_data(trial).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        self.assertNotEqual(df["mean"].tolist(), [trial.arm.parameters["x"] + 1.0])
        self.assertEqual(df["sem"].tolist(), [1.0])
        df = metric.fetch_trial_data(trial, noisy=False).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        self.assertEqual(df["mean"].tolist(), [trial.arm.parameters["x"] + 1.0])
        self.assertEqual(df["sem"].tolist(), [0.0])

        # unknown noise level
        metric = GenericNoisyFunctionMetric(
            name="test_metric",
            f=f,
            noise_sd=None,
        )
        trial = get_trial()
        df = metric.fetch_trial_data(trial).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        self.assertEqual(df["mean"].tolist(), [trial.arm.parameters["x"] + 1.0])
        self.assertEqual(df["mean"].tolist(), [trial.arm.parameters["x"] + 1.0])
        self.assertTrue(math.isnan(df["sem"].tolist()[0]))
