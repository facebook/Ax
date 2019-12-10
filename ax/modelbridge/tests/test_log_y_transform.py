#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.modelbridge.transforms.log_y import (
    LogY,
    lognorm_to_norm,
    match_ci_width,
    norm_to_lognorm,
)
from ax.utils.common.testutils import TestCase


class LogYTransformTest(TestCase):
    def setUp(self):
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2", "m3"],
            means=np.array([0.5, 1.0, 1.0]),
            covariance=np.diag(np.array([1.0, 1.0, np.exp(1) - 1])),
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

    def testInit(self):
        shared_init_args = {
            "search_space": None,
            "observation_features": None,
            "observation_data": [self.obsd1, self.obsd2],
        }
        # test error for not specifying a config
        with self.assertRaises(ValueError):
            LogY(**shared_init_args)
        # test error for not specifying at least one metric
        with self.assertRaises(ValueError):
            LogY(**shared_init_args, config={})
        # test default init
        tf = LogY(**shared_init_args, config={"metrics": ["m1"]})
        self.assertEqual(tf._transform, lognorm_to_norm)
        self.assertEqual(tf._untransform, norm_to_lognorm)
        # test match_ci_width init
        tf = LogY(
            **shared_init_args, config={"metrics": ["m1"], "match_ci_width": True}
        )
        self.assertTrue("<lambda>" in tf._transform.__name__)
        self.assertTrue("<lambda>" in tf._untransform.__name__)
        self.assertEqual(tf._transform(1.0, 0.1), match_ci_width(1.0, 0.1, np.log))
        self.assertEqual(tf._untransform(0.0, 0.1), match_ci_width(0.0, 0.1, np.exp))

    def testTransformObservations(self):
        # test default transform
        obsd1_t = ObservationData(
            metric_names=["m1", "m2", "m3"],
            means=np.array([0.5, 1.0, -0.5]),
            covariance=np.diag(np.array([1.0, 1.0, 1.0])),
        )
        tf = LogY(
            search_space=None,
            observation_features=None,
            observation_data=[],
            config={"metrics": ["m3"]},
        )
        obsd1 = deepcopy(self.obsd1)
        obsd1_ = tf.transform_observation_data([obsd1], [])
        self.assertTrue(obsd1_[0] == obsd1_t)
        obsd1 = tf.untransform_observation_data(obsd1_, [])
        self.assertTrue(obsd1[0] == self.obsd1)
        # test raise on non-independent noise
        obsd1_ = deepcopy(self.obsd1)
        obsd1_.covariance[0, 2] = 0.1
        obsd1_.covariance[2, 0] = 0.1
        with self.assertRaises(NotImplementedError):
            tf.transform_observation_data([obsd1_], [])
        # test full covariance for single metric
        Z = np.zeros((3, 3))
        Z[0, 2] = np.sqrt(np.exp(1)) - 1
        Z[2, 0] = np.sqrt(np.exp(1)) - 1
        obsd1 = ObservationData(
            metric_names=["m3", "m3", "m3"],
            means=np.ones(3),
            covariance=np.diag(np.ones(3) * (np.exp(1) - 1)) + Z,
        )
        obsd1_ = tf.transform_observation_data([obsd1], [])
        cov_expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
        self.assertTrue(np.allclose(obsd1_[0].means, -0.5 * np.ones(3)))
        self.assertTrue(np.allclose(obsd1_[0].covariance, cov_expected))
        # TODO: match_ci_width test

    def testTransformOptimizationConfig(self):
        # basic test
        m1 = Metric(name="m1")
        objective = Objective(metric=m1, minimize=False)
        oc = OptimizationConfig(objective=objective, outcome_constraints=[])
        tf = LogY(
            search_space=None,
            observation_features=None,
            observation_data=[self.obsd1, self.obsd2],
            config={"metrics": ["m1"]},
        )
        oc_tf = tf.transform_optimization_config(oc, None, None)
        self.assertTrue(oc_tf == oc)
        # test error if transformed metric appears in outcome constraints
        m2 = Metric(name="m2")
        cons = [
            OutcomeConstraint(metric=m2, op=ComparisonOp.GEQ, bound=0.0, relative=False)
        ]
        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        oc_tf = tf.transform_optimization_config(oc, None, None)
        self.assertTrue(oc_tf == oc)
        m2 = Metric(name="m2")
        cons = [
            OutcomeConstraint(metric=m2, op=ComparisonOp.GEQ, bound=0.0, relative=False)
        ]
        oc = OptimizationConfig(objective=objective, outcome_constraints=cons)
        tf2 = LogY(
            search_space=None,
            observation_features=None,
            observation_data=[self.obsd1, self.obsd2],
            config={"metrics": ["m2"]},
        )
        with self.assertRaises(ValueError):
            tf2.transform_optimization_config(oc, None, None)


class LogNormTest(TestCase):
    def test_lognorm_to_norm(self):
        mu_ln = np.ones(3)
        Cov_ln = np.diag(mu_ln * (np.exp(1) - 1))
        mu_n, Cov_n = lognorm_to_norm(mu_ln, Cov_ln)
        self.assertTrue(np.allclose(mu_n, -0.5 * np.ones_like(mu_n)))
        self.assertTrue(np.allclose(Cov_n, np.eye(3)))
        Z = np.zeros_like(Cov_ln)
        Z[0, 2] = np.sqrt(np.exp(1)) - 1
        Z[2, 0] = np.sqrt(np.exp(1)) - 1
        Cov_ln2 = Cov_ln + Z
        mu_n2, Cov_n2 = lognorm_to_norm(mu_ln, Cov_ln2)
        self.assertTrue(np.allclose(mu_n2, -0.5 * np.ones_like(mu_n2)))
        Cov_n2_expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
        self.assertTrue(np.allclose(Cov_n2, Cov_n2_expected))

    def test_norm_to_lognorm(self):
        mu_n = -0.5 * np.ones(3)
        Cov_n = np.eye(3)
        mu_ln, Cov_ln = norm_to_lognorm(mu_n, Cov_n)
        self.assertTrue(np.allclose(mu_ln, np.ones(3)))
        self.assertTrue(np.allclose(Cov_ln, (np.exp(1) - 1) * np.eye(3)))
        Cov_n2 = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
        mu_ln2, Cov_ln2 = norm_to_lognorm(mu_n, Cov_n2)
        Z = np.zeros_like(Cov_ln2)
        Z[0, 2] = np.sqrt(np.exp(1)) - 1
        Z[2, 0] = np.sqrt(np.exp(1)) - 1
        self.assertTrue(np.allclose(mu_ln2, np.ones(3)))
        self.assertTrue(np.allclose(Cov_ln2, np.exp(np.eye(3)) + Z - 1))
