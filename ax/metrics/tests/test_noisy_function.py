#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math

from ax.core.types import TParamValue
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial
from pyre_extensions import none_throws


class GenericNoisyFunctionMetricTest(TestCase):
    def test_GenericNoisyFunctionMetric(self) -> None:
        def f(params: dict[str, TParamValue]) -> float:
            return float(params["x"]) + 1.0

        # noiseless
        metric = GenericNoisyFunctionMetric(
            name="test_metric",
            f=f,
        )
        trial = get_trial()
        df = metric.fetch_trial_data(trial).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        self.assertEqual(
            df["mean"].tolist(),
            [float(none_throws(trial.arm).parameters["x"]) + 1.0],
        )
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
        self.assertNotEqual(
            df["mean"].tolist(),
            [float(none_throws(trial.arm).parameters["x"]) + 1.0],
        )
        self.assertEqual(df["sem"].tolist(), [1.0])
        df = metric.fetch_trial_data(trial, noisy=False).unwrap().df
        self.assertEqual(df["arm_name"].tolist(), ["0_0"])
        self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
        arm = none_throws(trial.arm)
        self.assertEqual(df["mean"].tolist(), [float(arm.parameters["x"]) + 1.0])
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
        arm = none_throws(trial.arm)
        self.assertEqual(df["mean"].tolist(), [float(arm.parameters["x"]) + 1.0])
        self.assertEqual(df["mean"].tolist(), [float(arm.parameters["x"]) + 1.0])
        self.assertTrue(math.isnan(df["sem"].tolist()[0]))
