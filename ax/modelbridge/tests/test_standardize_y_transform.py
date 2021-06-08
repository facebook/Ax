#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from math import sqrt

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.utils.common.testutils import TestCase


class StandardizeYTransformTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m1", "m2", "m2"],
            means=np.array([1.0, 1.0, 2.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2, 0.4],
                    [0.0, 0.2, 2.0, 0.8],
                    [0.0, 0.4, 0.8, 3.0],
                ]
            ),
        )
        self.t = StandardizeY(
            search_space=None,
            observation_features=None,
            observation_data=[self.obsd1, self.obsd2],
        )

    def testInit(self):
        self.assertEqual(self.t.Ymean, {"m1": 1.0, "m2": 1.5})
        self.assertEqual(self.t.Ystd, {"m1": 1.0, "m2": sqrt(1 / 3)})
        with self.assertRaises(ValueError):
            StandardizeY(
                search_space=None, observation_features=None, observation_data=[]
            )

    def testTransformObservations(self):
        obsd1_t = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([0.0, sqrt(3 / 4), -sqrt(3 / 4)]),
            covariance=np.array(
                [
                    [1.0, 0.2 * sqrt(3), 0.4 * sqrt(3)],
                    [0.2 * sqrt(3), 6.0, 2.4],
                    [0.4 * sqrt(3), 2.4, 9.0],
                ],
            ),
        )
        obsd2 = [deepcopy(self.obsd1)]
        obsd2 = self.t.transform_observation_data(obsd2, [])
        self.assertTrue(osd_allclose(obsd2[0], obsd1_t))
        obsd2 = self.t.untransform_observation_data(obsd2, [])
        self.assertTrue(osd_allclose(obsd2[0], self.obsd1))

    def testTransformOptimizationConfig(self):
        m1 = Metric(name="m1")
        m2 = Metric(name="m2")
        m3 = Metric(name="m3")
        objective = Objective(metric=m3, minimize=False)
        cons = [
            OutcomeConstraint(
                metric=m1, op=ComparisonOp.GEQ, bound=2.0, relative=False
            ),
            OutcomeConstraint(
                metric=m2, op=ComparisonOp.LEQ, bound=3.5, relative=False
            ),
            ScalarizedOutcomeConstraint(
                metrics=[m1, m2],
                weights=[0.5, 0.5],
                op=ComparisonOp.LEQ,
                bound=3.5,
                relative=False,
            ),
        ]
        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        oc = self.t.transform_optimization_config(oc, None, None)
        cons_t = [
            OutcomeConstraint(
                metric=m1, op=ComparisonOp.GEQ, bound=1.0, relative=False
            ),
            OutcomeConstraint(
                metric=m2,
                op=ComparisonOp.LEQ,
                bound=2.0 * sqrt(3),  # (3.5 - 1.5) / sqrt(1/3)
                relative=False,
            ),
            ScalarizedOutcomeConstraint(
                metrics=[m1, m2],
                weights=[0.5 * 1.0, 0.5 * sqrt(1 / 3)],
                op=ComparisonOp.LEQ,
                bound=2.25,  # 3.5 - (0.5 * 1.0 + 0.5 * 1.5)
                relative=False,
            ),
        ]
        self.assertTrue(oc.outcome_constraints == cons_t)
        self.assertTrue(oc.objective == objective)

        # Check fail with relative
        con = OutcomeConstraint(
            metric=m1, op=ComparisonOp.GEQ, bound=2.0, relative=True
        )
        oc = OptimizationConfig(objective=objective, outcome_constraints=[con])
        with self.assertRaises(ValueError):
            oc = self.t.transform_optimization_config(oc, None, None)


def osd_allclose(osd1: ObservationData, osd2: ObservationData) -> bool:
    if osd1.metric_names != osd2.metric_names:
        return False
    if not np.allclose(osd1.means, osd2.means):
        return False
    if not np.allclose(osd1.covariance, osd2.covariance):
        return False
    return True
