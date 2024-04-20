#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from math import sqrt

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.tests.test_standardize_y_transform import osd_allclose
from ax.utils.common.testutils import TestCase


class StratifiedStandardizeYTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.obs1 = Observation(features=self.obsf1, data=self.obsd1)
        self.obs2 = Observation(features=self.obsf2, data=self.obsd2)
        self.t = StratifiedStandardizeY(
            search_space=self.search_space,
            observations=[self.obs1, self.obs2],
            config={"parameter_name": "z"},
        )
        self.search_space2 = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=10
                ),
                ChoiceParameter(
                    name="z",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                ),
            ]
        )
        self.strata_mapping = {"a": 0, "b": 1, "c": 1}
        self.obsf3 = ObservationFeatures({"x": 5, "z": "c"})
        self.obsd3 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([2.0, 4.0, 16.0]),
            covariance=np.array([[1.2, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obs3 = Observation(features=self.obsf3, data=self.obsd3)
        self.t2 = StratifiedStandardizeY(
            search_space=self.search_space2,
            observations=[self.obs1, self.obs2, self.obs3],
            config={"parameter_name": "z", "strata_mapping": self.strata_mapping},
        )
        self.m1 = Metric(name="m1")
        self.m2 = Metric(name="m2")
        self.m3 = Metric(name="m3")
        self.objective = Objective(metric=self.m3, minimize=False)
        self.cons = [
            OutcomeConstraint(
                metric=self.m1, op=ComparisonOp.GEQ, bound=2.0, relative=False
            ),
            OutcomeConstraint(
                metric=self.m2, op=ComparisonOp.LEQ, bound=3.5, relative=False
            ),
        ]

    def test_Init(self) -> None:
        Ymean_expected = {
            ("m1", "a"): 1.0,
            ("m1", "b"): 3.0,
            ("m2", "a"): 5.0,
            ("m2", "b"): 1.5,
        }
        Ystd_expected = {
            ("m1", "a"): 1.0,
            ("m1", "b"): sqrt(2) * 2.0,
            ("m2", "a"): sqrt(2) * 3.0,
            ("m2", "b"): sqrt(2) * 0.5,
        }
        self.assertEqual(
            self.t.Ymean,
            Ymean_expected,
        )
        self.assertEqual(set(self.t.Ystd), set(Ystd_expected))
        for k, v in self.t.Ystd.items():
            self.assertAlmostEqual(v, Ystd_expected[k])
        with self.assertRaises(ValueError):
            # No parameter specified
            StratifiedStandardizeY(
                search_space=self.search_space,
                observations=[self.obs1, self.obs2],
            )
        with self.assertRaises(ValueError):
            # Wrong parameter type
            StratifiedStandardizeY(
                search_space=self.search_space,
                observations=[self.obs1, self.obs2],
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
                    target_value="a",
                ),
                ChoiceParameter(
                    name="z2",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b"],
                    is_task=True,
                    target_value="b",
                ),
            ]
        )
        with self.assertRaises(ValueError):
            StratifiedStandardizeY(
                search_space=ss3,
                observations=[self.obs1, self.obs2],
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
                    target_value="a",
                ),
            ]
        )
        t2 = StratifiedStandardizeY(
            search_space=ss2,
            observations=[self.obs1, self.obs2],
        )
        self.assertEqual(
            t2.Ymean,
            Ymean_expected,
        )
        self.assertEqual(set(t2.Ystd), set(Ystd_expected))
        for k, v in t2.Ystd.items():
            self.assertAlmostEqual(v, Ystd_expected[k])

        # test strata_mapping
        self.assertEqual(self.t2.strata_mapping, self.strata_mapping)

    def test_TransformObservations(self) -> None:
        std_m2_a = sqrt(2) * 3
        obsd1_ta = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([0.0, -3.0 / std_m2_a, 3.0 / std_m2_a]),
            covariance=np.array(
                [
                    [1.0, 0.2 / std_m2_a, 0.4 / std_m2_a],
                    [0.2 / std_m2_a, 2.0 / 18, 0.8 / 18],
                    [0.4 / std_m2_a, 0.8 / 18, 3.0 / 18],
                ]
            ),
        )
        std_m1_b, std_m2_b = 2 * sqrt(2), sqrt(1 / 2)
        obsd1_tb = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([-2.0 / std_m1_b, 0.5 / std_m2_b, 6.5 / std_m2_b]),
            covariance=np.array(
                [
                    [1.0 / 8, 0.2 / 2, 0.4 / 2],
                    [0.2 / 2, 2.0 * 2, 0.8 * 2],
                    [0.4 / 2, 0.8 * 2, 3.0 * 2],
                ]
            ),
        )
        obsd2 = deepcopy(self.obsd1)
        obsd2 = self.t.transform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "a"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, obsd1_ta))
        obsd2 = self.t.untransform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "a"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, self.obsd1))
        obsd2 = deepcopy(self.obsd1)
        obsd2 = self.t.transform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "b"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, obsd1_tb))
        obsd2 = self.t.untransform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "b"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, self.obsd1))
        # test strata_mapping
        # "a" should be the same as above
        obsd2 = deepcopy(self.obsd1)
        obsd2 = self.t2.transform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "a"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, obsd1_ta))
        # "b" and "c" should result in the same transformed values
        obsd2 = deepcopy(self.obsd1)
        obsd2 = self.t2.transform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "b"}))]
        )[0].data
        obsd3 = deepcopy(self.obsd1)
        obsd3 = self.t2.transform_observations(
            [Observation(data=obsd3, features=ObservationFeatures({"z": "c"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, obsd3))
        obsd3 = self.t2.untransform_observations(
            [Observation(data=obsd3, features=ObservationFeatures({"z": "c"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd3, self.obsd1))
        obsd2 = self.t2.untransform_observations(
            [Observation(data=obsd2, features=ObservationFeatures({"z": "b"}))]
        )[0].data
        self.assertTrue(osd_allclose(obsd2, self.obsd1))

    def test_TransformOptimizationConfig(self) -> None:
        cons2 = deepcopy(self.cons)
        oc = OptimizationConfig(objective=self.objective, outcome_constraints=cons2)
        fixed_features = ObservationFeatures({"z": "a"})
        oc = self.t.transform_optimization_config(oc, None, fixed_features)
        cons_t = [
            OutcomeConstraint(
                metric=self.m1, op=ComparisonOp.GEQ, bound=1.0, relative=False
            ),
            OutcomeConstraint(
                metric=self.m2,
                op=ComparisonOp.LEQ,
                bound=(3.5 - 5.0) / (sqrt(2) * 3),
                relative=False,
            ),
        ]
        self.assertTrue(oc.outcome_constraints == cons_t)
        self.assertTrue(oc.objective == self.objective)

        # No constraints
        oc2 = OptimizationConfig(objective=self.objective)
        oc3 = deepcopy(oc2)
        fixed_features = ObservationFeatures({"z": "a"})
        oc3 = self.t.transform_optimization_config(oc3, None, fixed_features)
        self.assertTrue(oc2 == oc3)

        # Check fail with relative
        con = OutcomeConstraint(
            metric=self.m1, op=ComparisonOp.GEQ, bound=2.0, relative=True
        )
        oc = OptimizationConfig(objective=self.objective, outcome_constraints=[con])
        with self.assertRaises(ValueError):
            oc = self.t.transform_optimization_config(oc, None, fixed_features)
        # Fail without strat param fixed
        fixed_features = ObservationFeatures({"x": 2.0})
        with self.assertRaises(ValueError):
            oc = self.t.transform_optimization_config(oc, None, fixed_features)

    def test_TransformOptimizationConfigWithStrataMapping(self) -> None:
        cons2 = deepcopy(self.cons)
        oc = OptimizationConfig(objective=self.objective, outcome_constraints=cons2)
        fixed_features = ObservationFeatures({"z": "a"})
        cons_t = [
            OutcomeConstraint(
                metric=self.m1, op=ComparisonOp.GEQ, bound=1.0, relative=False
            ),
            OutcomeConstraint(
                metric=self.m2,
                op=ComparisonOp.LEQ,
                bound=(3.5 - 5.0) / (sqrt(2) * 3),
                relative=False,
            ),
        ]
        cons2 = deepcopy(self.cons)
        oc = OptimizationConfig(objective=self.objective, outcome_constraints=cons2)
        oc = self.t2.transform_optimization_config(oc, None, fixed_features)
        self.assertTrue(oc.outcome_constraints == cons_t)
        self.assertTrue(oc.objective == self.objective)
        fixed_features = ObservationFeatures({"z": "c"})
        cons2 = deepcopy(self.cons)
        oc = OptimizationConfig(objective=self.objective, outcome_constraints=cons2)
        oc = self.t2.transform_optimization_config(oc, None, fixed_features)
        cons_t2 = [
            OutcomeConstraint(
                metric=self.m1,
                op=ComparisonOp.GEQ,
                bound=(2.0 - self.t2.Ymean[("m1", 1)]) / self.t2.Ystd[("m1", 1)],
                relative=False,
            ),
            OutcomeConstraint(
                metric=self.m2,
                op=ComparisonOp.LEQ,
                bound=(3.5 - self.t2.Ymean[("m2", 1)]) / self.t2.Ystd[("m2", 1)],
                relative=False,
            ),
        ]
        self.assertEqual(oc.outcome_constraints, cons_t2)
        self.assertTrue(oc.objective == self.objective)
        cons_t = [
            OutcomeConstraint(
                metric=self.m1, op=ComparisonOp.GEQ, bound=1.0, relative=False
            ),
            OutcomeConstraint(
                metric=self.m2,
                op=ComparisonOp.LEQ,
                bound=(3.5 - 5.0) / (sqrt(2) * 3),
                relative=False,
            ),
        ]
