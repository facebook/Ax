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


class GenericNoisyFunctionMetricTest(TestCase):
    def test_GenericNoisyFunctionMetric(self) -> None:
        def f(params: dict[str, TParamValue]) -> float:
            return float(params["x"]) + 1.0

        noise_configs = [
            # noise_sd=0 -> deterministic, sem=0
            ("noiseless", 0.0),
            # noise_sd=1.0 -> stochastic, sem=1.0; also test noisy=False override
            ("noisy", 1.0),
            # noise_sd=None -> unknown noise level, sem=NaN
            ("unknown_noise", None),
        ]
        for label, noise_sd in noise_configs:
            with self.subTest(noise_config=label, noise_sd=noise_sd):
                metric = GenericNoisyFunctionMetric(
                    name="test_metric",
                    f=f,
                    noise_sd=noise_sd if label != "noiseless" else 0.0,
                )
                trial = get_trial()
                df = metric.fetch_trial_data(trial).unwrap().df
                self.assertEqual(df["arm_name"].tolist(), ["0_0"])
                self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
                # pyre-fixme[16]: Optional type has no attribute `parameters`.
                expected_mean = trial.arm.parameters["x"] + 1.0

                if label == "noisy":
                    self.assertNotEqual(df["mean"].tolist(), [expected_mean])
                    self.assertEqual(df["sem"].tolist(), [1.0])
                    # Also verify noisy=False returns exact mean
                    df_noiseless = (
                        metric.fetch_trial_data(trial, noisy=False).unwrap().df
                    )
                    self.assertEqual(df_noiseless["arm_name"].tolist(), ["0_0"])
                    self.assertEqual(
                        df_noiseless["metric_name"].tolist(), ["test_metric"]
                    )
                    self.assertEqual(df_noiseless["mean"].tolist(), [expected_mean])
                    self.assertEqual(df_noiseless["sem"].tolist(), [0.0])
                elif label == "noiseless":
                    self.assertEqual(df["mean"].tolist(), [expected_mean])
                    self.assertEqual(df["sem"].tolist(), [0.0])
                else:  # unknown_noise
                    self.assertEqual(df["mean"].tolist(), [expected_mean])
                    self.assertTrue(math.isnan(df["sem"].tolist()[0]))
