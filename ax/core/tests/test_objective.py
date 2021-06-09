#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.utils.common.testutils import TestCase

MULTI_OBJECTIVE_REPR = """
MultiObjective(objectives=
[Objective(metric_name="m1", minimize=False),
Objective(metric_name="m2", minimize=True),
Objective(metric_name="m3", minimize=False)])
"""


class ObjectiveTest(TestCase):
    def setUp(self):
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2", lower_is_better=True),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        self.objectives = {
            "o1": Objective(metric=self.metrics["m1"]),
            "o2": Objective(metric=self.metrics["m2"], minimize=True),
            "o3": Objective(metric=self.metrics["m3"], minimize=False),
        }
        self.objective = Objective(metric=self.metrics["m1"], minimize=False)
        self.multi_objective = MultiObjective(
            objectives=[
                self.objectives["o1"],
                self.objectives["o2"],
                self.objectives["o3"],
            ]
        )
        self.scalarized_objective = ScalarizedObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"]]
        )

    def testInit(self):
        with self.assertRaises(ValueError):
            ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]], weights=[1.0]
            )
        warnings.resetwarnings()
        warnings.simplefilter("always", append=True)
        with warnings.catch_warnings(record=True) as ws:
            Objective(metric=self.metrics["m1"])
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in ws))
            self.assertTrue(
                any("Defaulting to `minimize=False`" in str(w.message) for w in ws)
            )
        with warnings.catch_warnings(record=True) as ws:
            Objective(Metric(name="m4", lower_is_better=True), minimize=False)
            self.assertTrue(any("Attempting to maximize" in str(w.message) for w in ws))
        with warnings.catch_warnings(record=True) as ws:
            Objective(Metric(name="m4", lower_is_better=False), minimize=True)
            self.assertTrue(any("Attempting to minimize" in str(w.message) for w in ws))
        self.assertEqual(
            self.objective.get_unconstrainable_metrics(), [self.metrics["m1"]]
        )

    def testMultiObjective(self):
        with self.assertRaises(NotImplementedError):
            return self.multi_objective.metric

        self.assertEqual(self.multi_objective.metrics, list(self.metrics.values()))
        minimizes = [obj.minimize for obj in self.multi_objective.objectives]
        self.assertEqual(minimizes, [False, True, False])
        weights = [mw[1] for mw in self.multi_objective.objective_weights]
        self.assertEqual(weights, [1.0, 1.0, 1.0])
        self.assertEqual(self.multi_objective.clone(), self.multi_objective)
        self.assertEqual(
            str(self.multi_objective),
            (
                "MultiObjective(objectives="
                '[Objective(metric_name="m1", minimize=False), '
                'Objective(metric_name="m2", minimize=True), '
                'Objective(metric_name="m3", minimize=False)])'
            ),
        )
        self.assertEqual(self.multi_objective.get_unconstrainable_metrics(), [])

    def testMultiObjectiveBackwardsCompatibility(self):
        multi_objective = MultiObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"], self.metrics["m3"]]
        )
        minimizes = [obj.minimize for obj in multi_objective.objectives]
        self.assertEqual(multi_objective.metrics, list(self.metrics.values()))
        self.assertEqual(minimizes, [False, True, False])

        multi_objective_min = MultiObjective(
            metrics=[self.metrics["m1"], self.metrics["m2"], self.metrics["m3"]],
            minimize=True,
        )
        minimizes = [obj.minimize for obj in multi_objective_min.objectives]
        self.assertEqual(minimizes, [True, False, True])

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
        self.assertEqual(self.scalarized_objective.get_unconstrainable_metrics(), [])
