#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from copy import deepcopy

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.modelbridge.transforms.log_y import (
    lognorm_to_norm,
    LogY,
    match_ci_width,
    norm_to_lognorm,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_outcome_constraint


class LogYTransformTest(TestCase):
    def test_Init(self) -> None:
        # test error for not specifying a config
        with self.assertRaises(ValueError):
            LogY()
        # test error for not specifying at least one metric
        with self.assertRaises(ValueError):
            LogY(config={"metrics": []})
        # test default init
        tf = LogY(config={"metrics": ["m1"]})
        self.assertEqual(tf._transform, lognorm_to_norm)
        self.assertEqual(tf._untransform, norm_to_lognorm)
        # test match_ci_width init
        tf = LogY(
            config={"metrics": ["m1"], "match_ci_width": True},
        )
        self.assertTrue("<lambda>" in tf._transform.__name__)
        self.assertTrue("<lambda>" in tf._untransform.__name__)
        # pyre-fixme[6]: For 1st param expected `ndarray` but got `float`.
        # pyre-fixme[6]: For 2nd param expected `ndarray` but got `float`.
        self.assertEqual(tf._transform(1.0, 0.1), match_ci_width(1.0, 0.1, np.log))
        # pyre-fixme[6]: For 1st param expected `ndarray` but got `float`.
        # pyre-fixme[6]: For 2nd param expected `ndarray` but got `float`.
        self.assertEqual(tf._untransform(0.0, 0.1), match_ci_width(0.0, 0.1, np.exp))

    def test_TransformObservations(self) -> None:
        obsd_with_noise = ObservationData(
            metric_names=["m1", "m2", "m3"],
            means=np.array([0.5, 1.0, 1.0]),
            covariance=np.diag(np.array([1.0, 1.0, np.exp(1) - 1])),
        )
        # test default transform
        obsd1_t = ObservationData(
            metric_names=["m1", "m2", "m3"],
            means=np.array([0.5, 1.0, -0.5]),
            covariance=np.diag(np.array([1.0, 1.0, 1.0])),
        )
        tf = LogY(config={"metrics": ["m3"]})
        obsd1 = deepcopy(obsd_with_noise)
        obsd1_ = tf._transform_observation_data([obsd1])
        self.assertTrue(np.allclose(obsd1_[0].means, obsd1_t.means))
        self.assertTrue(np.allclose(obsd1_[0].covariance, obsd1_t.covariance))

        obsd1 = tf._untransform_observation_data(obsd1_)
        self.assertTrue(np.allclose(obsd1[0].means, obsd_with_noise.means))
        self.assertTrue(np.allclose(obsd1[0].covariance, obsd_with_noise.covariance))

        # test raise on non-independent noise
        obsd1_ = deepcopy(obsd_with_noise)
        obsd1_.covariance[0, 2] = 0.1
        obsd1_.covariance[2, 0] = 0.1
        with self.assertRaises(NotImplementedError):
            tf._transform_observation_data([obsd1_])
        # test full covariance for single metric
        Z = np.zeros((3, 3))
        Z[0, 2] = np.sqrt(np.exp(1)) - 1
        Z[2, 0] = np.sqrt(np.exp(1)) - 1
        obsd1 = ObservationData(
            metric_names=["m3", "m3", "m3"],
            means=np.ones(3),
            covariance=np.diag(np.ones(3) * (np.exp(1) - 1)) + Z,
        )
        obsd1_ = tf._transform_observation_data([obsd1])
        cov_expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
        self.assertTrue(np.allclose(obsd1_[0].means, -0.5 * np.ones(3)))
        self.assertTrue(np.allclose(obsd1_[0].covariance, cov_expected))

        # Test with unknown noise.
        obsd_without_noise = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([1.0, 2.0]),
            covariance=np.diag(np.array([float("nan"), float("nan")])),
        )
        tf = LogY(config={"metrics": ["m1", "m2"]})
        tf_obsd = tf._transform_observation_data([deepcopy(obsd_without_noise)])
        expected_mean = np.log(obsd_without_noise.means)
        self.assertTrue(np.allclose(tf_obsd[0].means, expected_mean))
        self.assertTrue(
            np.array_equal(
                tf_obsd[0].covariance, obsd_without_noise.covariance, equal_nan=True
            )
        )

        # TODO: match_ci_width test

    def test_TransformOptimizationConfig(self) -> None:
        # basic test
        m1 = Metric(name="m1")
        objective_m1 = Objective(metric=m1, minimize=False)
        oc = OptimizationConfig(objective=objective_m1, outcome_constraints=[])
        tf = LogY(search_space=None, observations=[], config={"metrics": ["m1"]})
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        self.assertEqual(oc_tf, oc)
        # output constraint on a different metric should work
        m2 = Metric(name="m2")
        oc = OptimizationConfig(
            objective=objective_m1,
            outcome_constraints=[
                get_outcome_constraint(metric=m2, bound=-1, relative=False)
            ],
        )
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        self.assertEqual(oc_tf, oc)
        # output constraint with a negative bound should fail
        objective_m2 = Objective(metric=m2, minimize=False)
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=[
                get_outcome_constraint(metric=m1, bound=-1.234, relative=False)
            ],
        )
        with self.assertRaises(ValueError) as cm:
            #  `None`.
            tf.transform_optimization_config(oc, None, None)
        self.assertEqual(
            "LogY transform cannot be applied to metric m1 since the "
            "bound isn't positive, got: -1.234.",
            str(cm.exception),
        )
        # output constraint with a zero bound should also fail
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=[
                get_outcome_constraint(metric=m1, bound=0, relative=False)
            ],
        )
        with self.assertRaises(ValueError) as cm:
            #  `None`.
            tf.transform_optimization_config(oc, None, None)
        self.assertEqual(
            "LogY transform cannot be applied to metric m1 since the "
            "bound isn't positive, got: 0.",
            str(cm.exception),
        )
        # output constraint with a positive bound should work
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=[
                get_outcome_constraint(metric=m1, bound=2.345, relative=False)
            ],
        )
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        oc.outcome_constraints[0].bound = math.log(2.345)
        self.assertEqual(oc_tf, oc)
        # output constraint with a relative bound should fail
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=[
                get_outcome_constraint(metric=m1, bound=2.345, relative=True)
            ],
        )
        with self.assertRaises(ValueError) as cm:
            #  `None`.
            tf.transform_optimization_config(oc, None, None)
        self.assertEqual(
            "LogY transform cannot be applied to metric m1 since it is "
            "subject to a relative constraint.",
            str(cm.exception),
        )

    def test_TransformOptimizationConfigMOO(self) -> None:
        m1 = Metric(name="m1", lower_is_better=False)
        m2 = Metric(name="m2", lower_is_better=True)
        mo = MultiObjective(
            objectives=[
                Objective(metric=m1, minimize=False),
                Objective(metric=m2, minimize=True),
            ],
        )
        objective_thresholds = [
            ObjectiveThreshold(metric=m1, bound=1.234, relative=False),
            ObjectiveThreshold(metric=m2, bound=3.456, relative=False),
        ]
        oc = MultiObjectiveOptimizationConfig(
            objective=mo,
            objective_thresholds=objective_thresholds,
        )
        tf = LogY(search_space=None, observations=[], config={"metrics": ["m1"]})
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        oc.objective_thresholds[0].bound = math.log(1.234)
        self.assertEqual(oc_tf, oc)


class LogNormTest(TestCase):
    def test_lognorm_to_norm(self) -> None:
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

    def test_norm_to_lognorm(self) -> None:
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
