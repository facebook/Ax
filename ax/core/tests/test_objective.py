#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.utils.common.testutils import TestCase


class ObjectiveTest(TestCase):
    def setUp(self):
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2", lower_is_better=True),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.multi_objective = MultiObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"], self.metrics["m3"]]
        )
        self.scalarized_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )

    def testBadInit(self):
        with self.assertRaises(ValueError):
            self.scalarized_objective_weighted = ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]], weights=[1.0]
            )

    def testMultiObjective(self):
        with self.assertRaises(NotImplementedError):
            return self.multi_objective.metric

        self.assertEqual(self.multi_objective.metrics, list(self.metrics.values()))
        weights = [mw[1] for mw in self.multi_objective.metric_weights]
        self.assertEqual(weights, [0, -1.0, 1.0])
        self.assertEqual(self.multi_objective.clone(), self.multi_objective)
        self.assertEqual(
            str(self.multi_objective),
            "MultiObjective(metric_names=['m1', 'm2', 'm3'], minimize=False)",
        )

    def testScalarizedObjective(self):
        with self.assertRaises(NotImplementedError):
            return self.scalarized_objective.metric

        self.assertEqual(
            self.scalarized_objective.metrics, [self.metrics["m1"], self.metrics["m2"]]
        )
        weights = [mw[1] for mw in self.scalarized_objective.metric_weights]
        self.assertEqual(weights, [1.0, 1.0])
        self.assertEqual(self.scalarized_objective.clone(), self.scalarized_objective)
        self.assertEqual(
            str(self.scalarized_objective),
            (
                "ScalarizedObjective(metric_names=['m1', 'm2'], weights=[1.0, 1.0], "
                "minimize=False)"
            ),
        )
