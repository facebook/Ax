#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric, MetricFetchE
from ax.utils.common.result import Err
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_metric,
    get_data,
    get_factorial_metric,
)


METRIC_STRING = "Metric('m1')"


class MetricTest(TestCase):
    def setUp(self) -> None:
        pass

    def testInit(self) -> None:
        metric = Metric(name="m1", lower_is_better=False)
        self.assertEqual(str(metric), METRIC_STRING)

    def testEq(self) -> None:
        metric1 = Metric(name="m1", lower_is_better=False)
        metric2 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric2)

        metric3 = Metric(name="m1", lower_is_better=True)
        self.assertNotEqual(metric1, metric3)

    def testClone(self) -> None:
        metric1 = Metric(name="m1", lower_is_better=False)
        self.assertEqual(metric1, metric1.clone())

        metric2 = get_branin_metric(name="branin")
        self.assertEqual(metric2, metric2.clone())

        metric3 = get_factorial_metric(name="factorial")
        self.assertEqual(metric3, metric3.clone())

    def testSortable(self) -> None:
        metric1 = Metric(name="m1", lower_is_better=False)
        metric2 = Metric(name="m2", lower_is_better=False)
        self.assertTrue(metric1 < metric2)

    def testWrapUnwrap(self) -> None:
        data = get_data()

        trial_multi = Metric._unwrap_trial_data_multi(
            results=Metric._wrap_trial_data_multi(data=data)
        )
        self.assertEqual(trial_multi, data)

        experiment = Metric._unwrap_experiment_data(
            results=Metric._wrap_experiment_data(data=data)
        )
        self.assertEqual(experiment, data)

        experiment_multi = Metric._unwrap_experiment_data_multi(
            results=Metric._wrap_experiment_data_multi(data=data)
        )
        self.assertEqual(experiment_multi, data)

    def testWrapErr(self) -> None:
        err = Err(MetricFetchE(message="failed!", exception=Exception("panic!")))

        with self.assertRaisesRegex(Exception, "panic"):
            Metric._unwrap_experiment_data_multi(results={0: {"foo": err}})

        with self.assertRaisesRegex(Exception, "panic"):
            Metric._unwrap_experiment_data(results={0: err})

        with self.assertRaisesRegex(Exception, "panic"):
            Metric._unwrap_trial_data_multi(results={"foo": err})

    def testMetricFetchE(self) -> None:
        def foo() -> bool:
            raise ValueError("bad value")

        def bar() -> bool:
            return foo()

        exception = None
        try:
            bar()
        except Exception as e:
            print(e)
            exception = e

        metric_fetch_e = MetricFetchE(message="foo", exception=exception)

        self.assertEqual(metric_fetch_e.message, "foo")

        self.assertIn("in foo", metric_fetch_e.__repr__())
        self.assertIn("in bar", metric_fetch_e.__repr__())
        self.assertIn("ValueError: bad value", metric_fetch_e.__repr__())
