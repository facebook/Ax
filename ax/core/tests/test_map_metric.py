#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.map_metric import MapMetric
from ax.utils.common.testutils import TestCase


METRIC_STRING = "MapMetric('m1')"


class MapMetricTest(TestCase):
    def setUp(self):
        pass

    def testInit(self):
        metric = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(str(metric), METRIC_STRING)

    def testEq(self):
        metric1 = MapMetric(name="m1", lower_is_better=False)
        metric2 = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric2)

        metric3 = MapMetric(name="m1", lower_is_better=True)
        self.assertNotEqual(metric1, metric3)

    def testClone(self):
        metric1 = MapMetric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric1.clone())

    def testSortable(self):
        metric1 = MapMetric(name="m1", lower_is_better=False)
        metric2 = MapMetric(name="m2", lower_is_better=False)
        self.assertTrue(metric1 < metric2)
