#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.modelbridge.transforms.sklearn_y import LogWarpingTransformer, LogWarpingY
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_observations_with_invalid_value


def get_constraint(
    metric: Metric, bound: float, relative: bool
) -> list[OutcomeConstraint]:
    return [
        OutcomeConstraint(
            metric=metric, op=ComparisonOp.GEQ, bound=bound, relative=relative
        )
    ]


class LogWarpingYTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.6, 0.9]),
            covariance=np.array([[0.03, 0.0], [0.0, 0.001]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.1, 0.4]),
            covariance=np.array([[0.005, 0.0], [0.0, 0.05]]),
        )
        self.obsd3 = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.9, 0.8]),
            covariance=np.array([[0.02, 0.0], [0.0, 0.01]]),
        )
        self.obsd_nan = ObservationData(
            metric_names=["m1", "m2"],
            means=np.array([0.3, 0.2]),
            covariance=np.array([[float("nan"), 0.0], [0.0, float("nan")]]),
        )
        self.observations = [
            Observation(features=ObservationFeatures({}), data=obsd)
            for obsd in [self.obsd1, self.obsd2, self.obsd3, self.obsd_nan]
        ]

    def test_Init(self) -> None:
        shared_init_args = {
            "search_space": None,
            "observations": self.observations[:2],
        }
        # Init without a config
        t = LogWarpingY(**shared_init_args)
        self.assertTrue(t.clip_mean)
        self.assertEqual(t.metric_names, ["m1", "m2"])

        # Test init with config
        for m in ["m1", "m2"]:
            tf = LogWarpingY(**shared_init_args, config={"metrics": [m]})
            # tf.transforms should only exist for m and be a LogWarpingTransformer
            self.assertIsInstance(tf.transforms, dict)
            self.assertEqual([*tf.transforms], [m])  # Check keys
            self.assertIsInstance(tf.transforms[m], LogWarpingTransformer)
            # tf.inv_bounds should only exist for m and be a tuple of length 2
            self.assertIsInstance(tf.inv_bounds, dict)
            self.assertEqual([*tf.inv_bounds], [m])  # Check keys
            self.assertIsInstance(tf.inv_bounds[m], tuple)
            margin = 1e-3  # TODO clean this up
            self.assertEqual(tf.inv_bounds[m], (-0.5 - margin, 0.5 + margin))

    def test_TransformAndUntransformOneMetric(self) -> None:
        t = LogWarpingY(
            search_space=None,
            observations=deepcopy(self.observations[:2]),
            config={"metrics": ["m1"]},
        )

        # Transform the data and make sure we don't touch m2
        observation_data_tf = t._transform_observation_data(
            deepcopy([self.obsd1, self.obsd2])
        )
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            self.assertNotAlmostEqual(obsd.means[0], obsd_orig.means[0])
            self.assertNotAlmostEqual(obsd.covariance[0][0], obsd_orig.covariance[0][0])
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1])
            self.assertAlmostEqual(obsd.covariance[1][1], obsd_orig.covariance[1][1])

        # Untransform the data and make sure the means are the same
        observation_data_untf = t._untransform_observation_data(observation_data_tf)
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            self.assertAlmostEqual(obsd.means[0], obsd_orig.means[0], places=4)
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1], places=4)

        # NaN covar values remain as NaNs
        transformed_obsd_nan = t._transform_observation_data([deepcopy(self.obsd_nan)])[
            0
        ]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))
        untransformed = t._untransform_observation_data([transformed_obsd_nan])[0]
        self.assertTrue(
            np.array_equal(
                untransformed.covariance, self.obsd_nan.covariance, equal_nan=True
            )
        )

    def test_TransformAndUntransformAllMetrics(self) -> None:
        t = LogWarpingY(
            search_space=None,
            observations=deepcopy(self.observations[:2]),
            config={"metrics": ["m1", "m2"]},
        )

        observation_data_tf = t._transform_observation_data(
            deepcopy([self.obsd1, self.obsd2])
        )
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertNotAlmostEqual(obsd.means[i], obsd_orig.means[i])
                self.assertNotAlmostEqual(
                    obsd.covariance[i][i], obsd_orig.covariance[i][i]
                )

        # Untransform the data and make sure the means are the same
        observation_data_untf = t._untransform_observation_data(observation_data_tf)
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertAlmostEqual(obsd.means[i], obsd_orig.means[i], places=4)

        # NaN covar values remain as NaNs
        transformed_obsd_nan = t._transform_observation_data([deepcopy(self.obsd_nan)])[
            0
        ]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))

    def test_TransformOptimizationConfig(self) -> None:
        m1 = Metric(name="m1")
        objective_m1 = Objective(metric=m1, minimize=False)
        m2 = Metric(name="m2")
        objective_m2 = Objective(metric=m2, minimize=False)

        # No constraints
        oc = OptimizationConfig(objective=objective_m1, outcome_constraints=[])
        tf = LogWarpingY(
            search_space=None,
            observations=self.observations[:2],
            config={"metrics": ["m1"]},
        )
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        self.assertEqual(oc_tf, oc)

        # Output constraint on a different metric should not transform the bound
        for bound in [-1.234, 0, 2.345]:
            oc = OptimizationConfig(
                objective=objective_m1,
                outcome_constraints=get_constraint(
                    metric=m2, bound=bound, relative=False
                ),
            )
            oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
            self.assertEqual(oc_tf, oc)

        # Output constraint on the same metric should transform the bound
        for bound in [-1.234, 0, 2.345]:
            oc = OptimizationConfig(
                objective=objective_m2,
                outcome_constraints=get_constraint(
                    metric=m1, bound=bound, relative=False
                ),
            )
            oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
            oc_true = deepcopy(oc)
            tf_bound = tf.transforms["m1"].transform(np.array(bound, ndmin=2)).item()
            oc_true.outcome_constraints[0].bound = tf_bound
            self.assertEqual(oc_tf, oc_true)

        # Check untransform of outcome constraint
        cons = tf.untransform_outcome_constraints(
            outcome_constraints=oc_tf.outcome_constraints, fixed_features=None
        )
        self.assertEqual(cons, oc.outcome_constraints)

        # Relative constraints aren't supported
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=get_constraint(metric=m1, bound=2.345, relative=True),
        )
        with self.assertRaisesRegex(
            ValueError,
            "LogWarpingY cannot be applied to metric m1 since it is "
            "subject to a relative constraint.",
        ):
            tf.transform_optimization_config(oc, None, None)

        # Untransform doesn't work if relative
        with self.assertRaises(ValueError):
            tf.untransform_outcome_constraints(
                outcome_constraints=oc.outcome_constraints,
                fixed_features=None,
            )

        # Support for scalarized outcome constraints isn't implemented
        m3 = Metric(name="m3")
        oc = OptimizationConfig(
            objective=objective_m2,
            outcome_constraints=[
                ScalarizedOutcomeConstraint(
                    metrics=[m1, m3], op=ComparisonOp.GEQ, bound=2.345, relative=False
                )
            ],
        )
        with self.assertRaises(NotImplementedError) as cm:
            tf.transform_optimization_config(oc, None, None)
        self.assertEqual(
            "LogWarpingY cannot be used for metric(s) {'m1'} "
            "that are part of a ScalarizedOutcomeConstraint.",
            str(cm.exception),
        )

    def test_non_finite_data_raises(self) -> None:
        for invalid_value in [float("nan"), float("inf")]:
            observations = get_observations_with_invalid_value(invalid_value)
            with self.assertRaisesRegex(
                ValueError, f"Non-finite data found for metric m1: {invalid_value}"
            ):
                LogWarpingY(observations=observations, config={"metrics": ["m1"]})


if __name__ == "__main__":
    import unittest

    unittest.main()
