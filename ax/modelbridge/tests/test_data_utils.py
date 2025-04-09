#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.trial_status import NON_ABANDONED_STATUSES, TrialStatus
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.data_utils import DataLoaderConfig
from ax.utils.common.testutils import TestCase


class TestDataUtils(TestCase):
    def test_data_loader_config(self) -> None:
        # Defaults
        config = DataLoaderConfig()
        self.assertFalse(config.fit_out_of_design)
        self.assertFalse(config.fit_abandoned)
        self.assertTrue(config.fit_only_completed_map_metrics)
        self.assertEqual(config.latest_rows_per_group, 1)
        self.assertIsNone(config.limit_rows_per_group)
        self.assertIsNone(config.limit_rows_per_metric)
        self.assertEqual(config.statuses_to_fit, NON_ABANDONED_STATUSES)
        self.assertEqual(config.statuses_to_fit_map_metric, {TrialStatus.COMPLETED})
        # Validation for latest / limit rows.
        with self.assertRaisesRegex(UnsupportedError, "must be None if either of"):
            DataLoaderConfig(latest_rows_per_group=1, limit_rows_per_metric=5)
        # With a bunch of modifications.
        config = DataLoaderConfig(
            fit_out_of_design=True,
            fit_abandoned=True,
            fit_only_completed_map_metrics=False,
            latest_rows_per_group=None,
            limit_rows_per_metric=10,
            limit_rows_per_group=20,
        )
        self.assertTrue(config.fit_out_of_design)
        self.assertTrue(config.fit_abandoned)
        self.assertFalse(config.fit_only_completed_map_metrics)
        self.assertIsNone(config.latest_rows_per_group)
        self.assertEqual(config.limit_rows_per_metric, 10)
        self.assertEqual(config.limit_rows_per_group, 20)
        self.assertEqual(config.statuses_to_fit, set(TrialStatus))
        self.assertEqual(config.statuses_to_fit_map_metric, set(TrialStatus))
