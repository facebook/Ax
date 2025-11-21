#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from math import sqrt

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas import DataFrame
from pandas.testing import assert_frame_equal


class StandardizeYTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.obsd1 = ObservationData(
            metric_signatures=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obsd2 = ObservationData(
            metric_signatures=["m1", "m1", "m2", "m2"],
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
        self.experiment_data = extract_experiment_data(
            experiment=get_experiment_with_observations(
                observations=[  # Same means as above.
                    [1.0, 2.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [1.0, 1.0],
                ],
                sems=[
                    [1.0, sqrt(2.0)],
                    [1.0, sqrt(3.0)],
                    [1.0, sqrt(2.0)],
                    [1.0, sqrt(3.0)],
                ],
            ),
            data_loader_config=DataLoaderConfig(),
        )
        self.t = StandardizeY(
            search_space=None,
            experiment_data=self.experiment_data,
        )

    def test_Init(self) -> None:
        self.assertEqual(self.t.Ymean, {"m1": 1.0, "m2": 1.5})
        self.assertEqual(self.t.Ystd, {"m1": 1.0, "m2": sqrt(1 / 3)})
        with self.assertRaises(DataRequiredError):
            StandardizeY(search_space=None)
        # Initialize with experiment data.
        t = StandardizeY(
            search_space=None,
            experiment_data=self.experiment_data,
        )
        self.assertEqual(t.Ymean, {"m1": 1.0, "m2": 1.5})
        self.assertEqual(t.Ystd, {"m1": 1.0, "m2": sqrt(1 / 3)})

    def test_TransformObservations(self) -> None:
        obsd1_t = ObservationData(
            metric_signatures=["m1", "m2", "m2"],
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
        obsd2 = self.t._transform_observation_data(obsd2)
        self.assertTrue(osd_allclose(obsd2[0], obsd1_t))
        obsd2 = self.t._untransform_observation_data(obsd2)
        self.assertTrue(osd_allclose(obsd2[0], self.obsd1))

    def test_TransformOptimizationConfig(self) -> None:
        m1 = Metric(name="m1")
        m2 = Metric(name="m2")
        m3 = Metric(name="m3")
        m4 = Metric(name="m4")
        objective = Objective(metric=m3, minimize=False)
        cons: list[OutcomeConstraint] = [
            OutcomeConstraint(
                metric=m4, op=ComparisonOp.GEQ, bound=2.0, relative=False
            ),
        ]

        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        with self.assertRaisesRegex(
            DataRequiredError, "`StandardizeY` transform requires constraint metric"
        ):
            oc = self.t.transform_optimization_config(oc, None, None)
        cons = [
            ScalarizedOutcomeConstraint(
                metrics=[m1, m4],
                weights=[0.5, 0.5],
                op=ComparisonOp.LEQ,
                bound=3.5,
                relative=False,
            ),
        ]
        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        with self.assertRaisesRegex(
            DataRequiredError, "`StandardizeY` transform requires constraint metric"
        ):
            oc = self.t.transform_optimization_config(oc, None, None)
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

    def test_transform_experiment_data(self) -> None:
        experiment_data = deepcopy(self.experiment_data)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=experiment_data
        )
        # Check that arm data is identical.
        self.assertIs(experiment_data.arm_data, transformed_data.arm_data)
        assert_frame_equal(
            self.experiment_data.arm_data,
            transformed_data.arm_data,
        )
        # Check that observation data is transformed correctly.
        observation_data = transformed_data.observation_data
        sq3 = sqrt(3.0)
        expected_means = DataFrame(
            index=observation_data.index,
            columns=observation_data["mean"].columns,
            data=[
                [0.0, 0.5 * sq3],
                [0.0, -0.5 * sq3],
                [0.0, 0.5 * sq3],
                [0.0, -0.5 * sq3],
            ],
        )
        assert_frame_equal(observation_data["mean"], expected_means)
        expected_sems = DataFrame(
            index=observation_data.index,
            columns=observation_data["sem"].columns,
            data=[
                [1.0, sqrt(2.0) * sq3],
                [1.0, sqrt(3.0) * sq3],
                [1.0, sqrt(2.0) * sq3],
                [1.0, sqrt(3.0) * sq3],
            ],
        )
        assert_frame_equal(observation_data["sem"], expected_sems)


def osd_allclose(osd1: ObservationData, osd2: ObservationData) -> bool:
    if osd1.metric_signatures != osd2.metric_signatures:
        return False
    if not np.allclose(osd1.means, osd2.means):
        return False
    if not np.allclose(osd1.covariance, osd2.covariance):
        return False
    return True
