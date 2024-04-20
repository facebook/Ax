#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.map_metric import MapMetric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_map_data


METRIC_STRING = "MapMetric('m1')"


class MapMetricTest(TestCase):
    def test_Init(self) -> None:
        metric = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(str(metric), METRIC_STRING)

    def test_Eq(self) -> None:
        metric1 = MapMetric(name="m1", lower_is_better=False)
        metric2 = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric2)

        metric3 = MapMetric(name="m1", lower_is_better=True)
        self.assertNotEqual(metric1, metric3)

    def test_Clone(self) -> None:
        metric1 = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric1.clone())

    def test_Sortable(self) -> None:
        metric1 = MapMetric(name="m1", lower_is_better=False)
        metric2 = MapMetric(name="m2", lower_is_better=False)
        self.assertTrue(metric1 < metric2)

    def test_WrapUnwrap(self) -> None:
        data = get_map_data()

        trial_multi = MapMetric._unwrap_trial_data_multi(
            results=MapMetric._wrap_trial_data_multi(data=data)
        )
        self.assertEqual(trial_multi, data)

        experiment = MapMetric._unwrap_experiment_data(
            results=MapMetric._wrap_experiment_data(data=data)
        )
        self.assertEqual(experiment, data)

        experiment_multi = MapMetric._unwrap_experiment_data_multi(
            results=MapMetric._wrap_experiment_data_multi(data=data)
        )
        self.assertEqual(experiment_multi, data)
