#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_metric, get_factorial_metric


METRIC_STRING = "Metric('m1')"


class MetricTest(TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def testInit(self):
        metric = Metric(name="m1", lower_is_better=False)
        self.assertEqual(str(metric), METRIC_STRING)

    # pyre-fixme[3]: Return type must be annotated.
    def testEq(self):
        metric1 = Metric(name="m1", lower_is_better=False)
        metric2 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric2)

        metric3 = Metric(name="m1", lower_is_better=True)
        self.assertNotEqual(metric1, metric3)

    # pyre-fixme[3]: Return type must be annotated.
    def testClone(self):
        metric1 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric1.clone())

        metric2 = get_branin_metric(name="branin")
        self.assertEqual(metric2, metric2.clone())

        metric3 = get_factorial_metric(name="factorial")
        self.assertEqual(metric3, metric3.clone())

    # pyre-fixme[3]: Return type must be annotated.
    def testSortable(self):
        metric1 = Metric(name="m1", lower_is_better=False)
        metric2 = Metric(name="m2", lower_is_better=False)
        self.assertTrue(metric1 < metric2)
