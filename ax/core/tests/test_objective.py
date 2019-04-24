#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.utils.common.testutils import TestCase


class ObjectiveTest(TestCase):
    def setUp(self):
        self.metrics = {"m1": Metric(name="m1"), "m2": Metric(name="m2")}
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.multi_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )

    def testBadInit(self):
        with self.assertRaises(ValueError):
            self.multi_objective_weighted = ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]], weights=[1.0]
            )

    def testScalarizedObjective(self):
        with self.assertRaises(NotImplementedError):
            return self.multi_objective.metric

        self.assertEqual(self.multi_objective.metrics, list(self.metrics.values()))
        weights = [mw[1] for mw in self.multi_objective.metric_weights]
        self.assertEqual(weights, [1.0, 1.0])
        self.assertEqual(self.multi_objective.clone(), self.multi_objective)
