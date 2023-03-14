#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.dict_lookup import DictLookupMetric
from ax.utils.common.result import Err
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial


class TestDictLookupMetric(TestCase):
    def setUp(self) -> None:
        self.lookup_dict = {
            (1.0, 2.0): 3.0,
            (2.0, 3.0): 4.0,
            (3.0, 4.0): 5.0,
        }
        self.param_names = ["p1", "p2"]
        self.metric = DictLookupMetric(
            name="test_metric",
            param_names=self.param_names,
            lookup_dict=self.lookup_dict,  # pyre-ignore
            noise_sd=0.0,
            lower_is_better=True,
        )

    def test_dict_lookup_metric(self) -> None:
        # Test init and basic attributes.
        self.assertTrue(self.metric.is_available_while_running())
        self.assertEqual(self.metric, self.metric.clone())
        self.assertTrue(self.metric.lower_is_better)
        self.assertEqual(self.metric.noise_sd, 0.0)
        self.assertEqual(self.metric.param_names, self.param_names)

        # Test trial evaluation.
        trial = get_trial()
        trial._generator_run = GeneratorRun(
            arms=[Arm(name="0_0", parameters={"p1": 1.0, "p2": 2.0})]
        )
        df = self.metric.fetch_trial_data(trial).unwrap().df
        self.assertEqual(df["mean"].values[0], 3.0)
        self.assertEqual(df["sem"].values[0], 0.0)

        # Invalid parameterization.
        trial = get_trial()
        trial._generator_run = GeneratorRun(
            arms=[Arm(name="0_0", parameters={"p1": 5.0, "p2": 2.0})]
        )
        fetch_res = self.metric.fetch_trial_data(trial)
        self.assertIsInstance(fetch_res, Err)
        self.assertIn("while attempting to retrieve", str(fetch_res.value))

        # With noise std.
        self.metric.noise_sd = 0.5
        trial = get_trial()
        trial._generator_run = GeneratorRun(
            arms=[Arm(name="0_0", parameters={"p1": 1.0, "p2": 2.0})]
        )
        with mock.patch(
            "ax.metrics.dict_lookup.np.random.randn", return_value=1.0
        ) as mock_randn:
            df = self.metric.fetch_trial_data(trial).unwrap().df
        mock_randn.assert_called_once()
        self.assertEqual(df["mean"].values[0], 3.5)
        self.assertEqual(df["sem"].values[0], 0.5)
