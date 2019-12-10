#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.utils.common.testutils import TestCase


class StratifiedStandardizeYTransformTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 8.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m1", "m2", "m2"],
            means=np.array([1.0, 5.0, 2.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2, 0.4],
                    [0.0, 0.2, 2.0, 0.8],
                    [0.0, 0.4, 0.8, 3.0],
                ]
            ),
        )
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=10
                ),
                ChoiceParameter(
                    name="z", parameter_type=ParameterType.STRING, values=["a", "b"]
                ),
            ]
        )
        self.obsf1 = ObservationFeatures({"x": 2, "z": "a"})
        self.obsf2 = ObservationFeatures({"x": 5, "z": "b"})
        self.t = StratifiedStandardizeY(
            search_space=self.search_space,
            observation_features=[self.obsf1, self.obsf2],
            observation_data=[self.obsd1, self.obsd2],
            config={"parameter_name": "z"},
        )

    def testInit(self):
        self.assertEqual(
            self.t.Ymean,
            {("m1", "a"): 1.0, ("m1", "b"): 3.0, ("m2", "a"): 5.0, ("m2", "b"): 1.5},
        )
        self.assertEqual(
            self.t.Ystd,
            {("m1", "a"): 1.0, ("m1", "b"): 2.0, ("m2", "a"): 3.0, ("m2", "b"): 0.5},
        )
        with self.assertRaises(ValueError):
            # No parameter specified
            StratifiedStandardizeY(
                search_space=self.search_space,
                observation_features=[self.obsf1, self.obsf2],
                observation_data=[self.obsd1, self.obsd2],
            )
        with self.assertRaises(ValueError):
            # Wrong parameter type
            StratifiedStandardizeY(
                search_space=self.search_space,
                observation_features=[self.obsf1, self.obsf2],
                observation_data=[self.obsd1, self.obsd2],
                config={"parameter_name": "x"},
            )
        # Multiple tasks parameters
        ss3 = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=10
                ),
                ChoiceParameter(
                    name="z",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b"],
                    is_task=True,
                ),
                ChoiceParameter(
                    name="z2",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b"],
                    is_task=True,
                ),
            ]
        )
        with self.assertRaises(ValueError):
            StratifiedStandardizeY(
                search_space=ss3,
                observation_features=[self.obsf1, self.obsf2],
                observation_data=[self.obsd1, self.obsd2],
            )

        # Grab from task feature
        ss2 = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=10
                ),
                ChoiceParameter(
                    name="z",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b"],
                    is_task=True,
                ),
            ]
        )
        t2 = StratifiedStandardizeY(
            search_space=ss2,
            observation_features=[self.obsf1, self.obsf2],
            observation_data=[self.obsd1, self.obsd2],
        )
        self.assertEqual(
            t2.Ymean,
            {("m1", "a"): 1.0, ("m1", "b"): 3.0, ("m2", "a"): 5.0, ("m2", "b"): 1.5},
        )
        self.assertEqual(
            t2.Ystd,
            {("m1", "a"): 1.0, ("m1", "b"): 2.0, ("m2", "a"): 3.0, ("m2", "b"): 0.5},
        )

    def testTransformObservations(self):
        obsd1_ta = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([0.0, -1.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.2 / 3, 0.4 / 3],
                    [0.2 / 3, 2.0 / 9, 0.8 / 9],
                    [0.4 / 3, 0.8 / 9, 3.0 / 9],
                ]
            ),
        )
        obsd1_tb = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([-1.0, 1.0, 13.0]),
            covariance=np.array([[0.25, 0.2, 0.4], [0.2, 8.0, 3.2], [0.4, 3.2, 12.0]]),
        )
        obsd2 = [deepcopy(self.obsd1)]
        obsd2 = self.t.transform_observation_data(
            obsd2, [ObservationFeatures({"z": "a"})]
        )
        self.assertEqual(obsd2[0], obsd1_ta)
        obsd2 = self.t.untransform_observation_data(
            obsd2, [ObservationFeatures({"z": "a"})]
        )
        self.assertEqual(obsd2[0], self.obsd1)
        obsd2 = [deepcopy(self.obsd1)]
        obsd2 = self.t.transform_observation_data(
            obsd2, [ObservationFeatures({"z": "b"})]
        )
        self.assertEqual(obsd2[0], obsd1_tb)
        obsd2 = self.t.untransform_observation_data(
            obsd2, [ObservationFeatures({"z": "b"})]
        )
        self.assertEqual(obsd2[0], self.obsd1)

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
        ]
        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        fixed_features = ObservationFeatures({"z": "a"})
        oc = self.t.transform_optimization_config(oc, None, fixed_features)
        cons_t = [
            OutcomeConstraint(
                metric=m1, op=ComparisonOp.GEQ, bound=1.0, relative=False
            ),
            OutcomeConstraint(
                metric=m2, op=ComparisonOp.LEQ, bound=-0.5, relative=False
            ),
        ]
        self.assertTrue(oc.outcome_constraints == cons_t)
        self.assertTrue(oc.objective == objective)

        # No constraints
        oc2 = OptimizationConfig(objective=objective)
        oc3 = deepcopy(oc2)
        oc3 = self.t.transform_optimization_config(oc3, None, fixed_features)
        self.assertTrue(oc2 == oc3)

        # Check fail with relative
        con = OutcomeConstraint(
            metric=m1, op=ComparisonOp.GEQ, bound=2.0, relative=True
        )
        oc = OptimizationConfig(objective=objective, outcome_constraints=[con])
        with self.assertRaises(ValueError):
            oc = self.t.transform_optimization_config(oc, None, fixed_features)
        # Fail without strat param fixed
        fixed_features = ObservationFeatures({"x": 2.0})
        with self.assertRaises(ValueError):
            oc = self.t.transform_optimization_config(oc, None, fixed_features)
