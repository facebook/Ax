#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.metrics.curve import AbstractCurveMetric
from ax.utils.common.testutils import TestCase


class AbstractCurveMetricTest(TestCase):
    def testAbstractCurveMetric(self) -> None:
        self.assertTrue(AbstractCurveMetric.is_available_while_running())
        with self.assertRaises(TypeError):
            # pyre-fixme[45]: Cannot instantiate abstract class `AbstractCurveMetric`.
            AbstractCurveMetric("foo", "bar")
