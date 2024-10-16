#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class ObjectiveTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.metrics = {
            "m1": Metric(name="m1"),
            "m2": Metric(name="m2", lower_is_better=True),
            "m3": Metric(name="m3", lower_is_better=False),
        }
        self.objectives = {
            "o1": Objective(metric=self.metrics["m1"], minimize=True),
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
            metrics=[self.metrics["m1"], self.metrics["m2"]], minimize=True
        )

    def test_Init(self) -> None:
        with self.assertRaisesRegex(UserInputError, "does not specify"):
            (Objective(metric=self.metrics["m1"]),)
        with self.assertRaisesRegex(
            UserInputError, "doesn't match the specified optimization direction"
        ):
            Objective(metric=self.metrics["m2"], minimize=False)
        with self.assertRaises(ValueError):
            ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]], weights=[1.0]
            )
        with self.assertRaisesRegex(
            ValueError,
            "Metric with name m2 specifies `lower_is_better` = "
            "True, which doesn't match the specified optimization direction.",
        ):
            # Should fail since m2 specifies lower_is_better=True
            ScalarizedObjective(
                metrics=[self.metrics["m1"], self.metrics["m2"]],
                minimize=False,
            )
        self.assertEqual(
            self.objective.get_unconstrainable_metrics(), [self.metrics["m1"]]
        )

    def test_MultiObjective(self) -> None:
        with self.assertRaises(NotImplementedError):
            # pyre-fixme[7]: Expected `None` but got `Metric`.
            return self.multi_objective.metric

        self.assertEqual(self.multi_objective.metrics, list(self.metrics.values()))
        minimizes = [obj.minimize for obj in self.multi_objective.objectives]
        self.assertEqual(minimizes, [True, True, False])
        weights = [mw[1] for mw in self.multi_objective.objective_weights]
        self.assertEqual(weights, [1.0, 1.0, 1.0])
        self.assertEqual(self.multi_objective.clone(), self.multi_objective)
        self.assertEqual(
            str(self.multi_objective),
            (
                "MultiObjective(objectives="
                '[Objective(metric_name="m1", minimize=True), '
                'Objective(metric_name="m2", minimize=True), '
                'Objective(metric_name="m3", minimize=False)])'
            ),
        )
        self.assertEqual(
            self.multi_objective.get_unconstrainable_metrics(),
            [self.metrics["m1"], self.metrics["m2"], self.metrics["m3"]],
        )

    def test_MultiObjectiveBackwardsCompatibility(self) -> None:
        metrics = [
            Metric(name="m1", lower_is_better=False),
            self.metrics["m2"],
            self.metrics["m3"],
        ]
        multi_objective = MultiObjective(metrics=metrics)
        minimizes = [obj.minimize for obj in multi_objective.objectives]
        self.assertEqual(multi_objective.metrics, metrics)
        self.assertEqual(minimizes, [False, True, False])

        multi_objective_min = MultiObjective(
            metrics=[
                Metric(name="m1"),
                Metric(name="m2"),
                Metric(name="m3", lower_is_better=True),
            ],
            minimize=True,
        )
        minimizes = [obj.minimize for obj in multi_objective_min.objectives]
        self.assertEqual(minimizes, [True, True, True])

    def test_ScalarizedObjective(self) -> None:
        with self.assertRaises(NotImplementedError):
            # pyre-fixme[7]: Expected `None` but got `Metric`.
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
                "minimize=True)"
            ),
        )
        self.assertEqual(
            self.scalarized_objective.get_unconstrainable_metrics(),
            [self.metrics["m1"], self.metrics["m2"]],
        )
