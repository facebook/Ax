#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.metric import Metric
from ax.utils.common.testutils import TestCase


METRIC_STRING = "Metric('m1')"


class MetricTest(TestCase):
    def setUp(self):
        pass

    def testInit(self):
        metric = Metric(name="m1", lower_is_better=False)
        self.assertEqual(str(metric), METRIC_STRING)

    def testEq(self):
        metric1 = Metric(name="m1", lower_is_better=False)
        metric2 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric2)

        metric3 = Metric(name="m1", lower_is_better=True)
        self.assertNotEqual(metric1, metric3)

    def testClone(self):
        metric1 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric1.clone())
