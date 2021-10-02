#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.metrics.curve import AbstractCurveMetric
from ax.utils.common.testutils import TestCase


class AbstractCurveMetricTest(TestCase):
    def testAbstractCurveMetric(self):
        self.assertTrue(AbstractCurveMetric.is_available_while_running())
        self.assertTrue(AbstractCurveMetric.overwrite_existing_data())
        with self.assertRaises(TypeError):
            AbstractCurveMetric("foo", "bar")
