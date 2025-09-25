#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy
from math import isnan

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.power_transform_y import (
    _compute_inverse_bounds,
    _compute_power_transforms,
    PowerTransformY,
)
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import observations_from_data
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from sklearn.preprocessing import PowerTransformer


def get_constraint(
    metric: Metric, bound: float, relative: bool
) -> list[OutcomeConstraint]:
    return [
        OutcomeConstraint(
            metric=metric, op=ComparisonOp.GEQ, bound=bound, relative=relative
        )
    ]


class PowerTransformYTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_experiment_with_observations(
            observations=[[0.5, 0.9], [0.1, 0.4], [0.9, 0.8], [0.3, 0.2]],
            sems=[[0.2, 0.1], [0.03, 0.05], [0.14, 0.1], [float("nan"), float("nan")]],
        )
        self.experiment_data = extract_experiment_data(
            experiment=self.experiment, data_loader_config=DataLoaderConfig()
        )
        self.observations = observations_from_data(
            experiment=self.experiment, data=self.experiment.lookup_data()
        )
        self.obsd1, self.obsd2, self.obsd3, self.obsd_nan = (
            obs.data for obs in self.observations
        )

    def test_init(self) -> None:
        shared_init_args = {
            "search_space": None,
            "experiment_data": self.experiment_data,
        }
        # Init without a config.
        t = PowerTransformY(**shared_init_args)
        self.assertTrue(t.clip_mean)
        self.assertEqual(t.metric_signatures, ["m1", "m2"])

        # Test init with config.
        for m in ["m1", "m2"]:
            tf = PowerTransformY(**shared_init_args, config={"metrics": [m]})
            # tf.power_transforms should only exist for m and be a PowerTransformer
            self.assertIsInstance(tf.power_transforms, dict)
            self.assertEqual([*tf.power_transforms], [m])  # Check keys
            self.assertIsInstance(tf.power_transforms[m], PowerTransformer)
            # tf.inv_bounds should only exist for m and be a tuple of length 2
            self.assertIsInstance(tf.inv_bounds, dict)
            self.assertEqual([*tf.inv_bounds], [m])  # Check keys
            self.assertIsInstance(tf.inv_bounds[m], tuple)
            self.assertTrue(len(tf.inv_bounds[m]) == 2)

    def test_compute_power_transform(self) -> None:
        Ys = {"m2": [0.9, 0.4, 0.8]}
        pts = _compute_power_transforms(Ys)
        self.assertEqual(pts["m2"].method, "yeo-johnson")
        # pyre-fixme[16]: `PowerTransformer` has no attribute `lambdas_`.
        self.assertIsInstance(pts["m2"].lambdas_, np.ndarray)
        self.assertEqual(pts["m2"].lambdas_.shape, (1,))
        Y_np = np.array(Ys["m2"])[:, None]
        Y_trans = pts["m2"].transform(Y_np)
        # Output should be standardized
        self.assertAlmostEqual(Y_trans.mean(), 0.0)
        self.assertAlmostEqual(Y_trans.std(), 1.0)
        # Transform back
        Y_np2 = pts["m2"].inverse_transform(Y_trans)
        self.assertAlmostEqual(np.max(np.abs(Y_np - Y_np2)), 0.0)

    def test_compute_inverse_bounds(self) -> None:
        Ys = {"m2": [0.9, 0.4, 0.8]}
        pt = _compute_power_transforms(Ys)["m2"]
        # lambda < 0: im(f) = (-inf, -1/lambda) without standardization
        # pyre-fixme[16]: `PowerTransformer` has no attribute `lambdas_`.
        pt.lambdas_.fill(-2.5)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertEqual(bounds[0], -np.inf)
        # Make sure we got the boundary right
        left = pt.inverse_transform(np.array(bounds[1] - 0.01, ndmin=2))
        right = pt.inverse_transform(np.array(bounds[1] + 0.01, ndmin=2))
        self.assertTrue(isnan(right) and not isnan(left))
        # 0 <= lambda <= 2: im(f) = R
        pt.lambdas_.fill(1.0)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertTrue(bounds == (-np.inf, np.inf))
        # lambda > 2: im(f) = (1 / (2 - lambda), inf) without standardization
        pt.lambdas_.fill(3.5)
        bounds = _compute_inverse_bounds({"m2": pt})["m2"]
        self.assertEqual(bounds[1], np.inf)
        # Make sure we got the boundary right
        left = pt.inverse_transform(np.array(bounds[0] - 0.01, ndmin=2))
        right = pt.inverse_transform(np.array(bounds[0] + 0.01, ndmin=2))
        self.assertTrue(not isnan(right) and isnan(left))

    def test_transform_and_untransform_one_metric(self) -> None:
        pt = PowerTransformY(
            search_space=None,
            experiment_data=self.experiment_data,
            config={"metrics": ["m1"]},
        )

        # Transform the data and make sure we don't touch m1
        observation_data_tf = pt._transform_observation_data(
            deepcopy([self.obsd1, self.obsd2])
        )
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            self.assertNotAlmostEqual(obsd.means[0], obsd_orig.means[0])
            self.assertNotAlmostEqual(obsd.covariance[0][0], obsd_orig.covariance[0][0])
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1])
            self.assertAlmostEqual(obsd.covariance[1][1], obsd_orig.covariance[1][1])

        # Untransform the data and make sure the means are the same
        observation_data_untf = pt._untransform_observation_data(observation_data_tf)
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            self.assertAlmostEqual(obsd.means[0], obsd_orig.means[0], places=4)
            self.assertAlmostEqual(obsd.means[1], obsd_orig.means[1], places=4)

        # NaN covar values remain as NaNs
        transformed_obsd_nan = pt._transform_observation_data(
            [deepcopy(self.obsd_nan)]
        )[0]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))
        untransformed = pt._untransform_observation_data([transformed_obsd_nan])[0]
        self.assertTrue(
            np.array_equal(
                untransformed.covariance, self.obsd_nan.covariance, equal_nan=True
            )
        )

    def test_transform_and_untransform_all_metrics(self) -> None:
        pt = PowerTransformY(
            search_space=None,
            experiment_data=self.experiment_data,
            config={"metrics": ["m1", "m2"]},
        )

        observation_data_tf = pt._transform_observation_data(
            deepcopy([self.obsd1, self.obsd2])
        )
        for obsd, obsd_orig in zip(observation_data_tf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertNotAlmostEqual(obsd.means[i], obsd_orig.means[i])
                self.assertNotAlmostEqual(
                    obsd.covariance[i][i], obsd_orig.covariance[i][i]
                )

        # Untransform the data and make sure the means are the same
        observation_data_untf = pt._untransform_observation_data(observation_data_tf)
        for obsd, obsd_orig in zip(observation_data_untf, [self.obsd1, self.obsd2]):
            for i in range(2):  # Both metrics should be transformed
                self.assertAlmostEqual(obsd.means[i], obsd_orig.means[i])

        # NaN covar values remain as NaNs
        transformed_obsd_nan = pt._transform_observation_data(
            [deepcopy(self.obsd_nan)]
        )[0]
        cov_results = np.array(transformed_obsd_nan.covariance)
        self.assertTrue(np.all(np.isnan(np.diag(cov_results))))

    def test_compare_to_sklearn(self) -> None:
        # Make sure the transformed values agree with Sklearn
        observation_data = [self.obsd1, self.obsd2, self.obsd3, self.obsd_nan]

        y_orig = np.array([data.means[0] for data in observation_data])[:, None]
        y1 = PowerTransformer("yeo-johnson").fit(y_orig).transform(y_orig).ravel()

        pt = PowerTransformY(
            search_space=None,
            experiment_data=self.experiment_data,
            config={"metrics": ["m1"]},
        )
        observation_data_tf = pt._transform_observation_data(observation_data)
        y2 = [data.means[0] for data in observation_data_tf]
        for y1_, y2_ in zip(y1, y2):
            self.assertAlmostEqual(y1_, y2_)

    def test_transform_optimization_config(self) -> None:
        # basic test
        m1 = Metric(name="m1")
        objective_m1 = Objective(metric=m1, minimize=False)
        oc = OptimizationConfig(objective=objective_m1, outcome_constraints=[])
        tf = PowerTransformY(
            search_space=None,
            experiment_data=self.experiment_data,
            config={"metrics": ["m1"]},
        )
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        self.assertEqual(oc_tf, oc)
        # Output constraint on a different metric should not transform the bound
        m2 = Metric(name="m2")
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
        objective_m2 = Objective(metric=m2, minimize=False)
        for bound in [-1.234, 0, 2.345]:
            oc = OptimizationConfig(
                objective=objective_m2,
                outcome_constraints=get_constraint(
                    metric=m1, bound=bound, relative=False
                ),
            )
            oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
            oc_true = deepcopy(oc)
            tf_bound = (
                tf.power_transforms["m1"].transform(np.array(bound, ndmin=2)).item()
            )
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
            "PowerTransformY cannot be applied to metric m1 since it is "
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
            "PowerTransformY cannot be used for metric(s) {'m1'} "
            "that are part of a ScalarizedOutcomeConstraint.",
            str(cm.exception),
        )

    def test_with_experiment_data(self) -> None:
        experiment_data = extract_experiment_data(
            experiment=self.experiment, data_loader_config=DataLoaderConfig()
        )
        for metrics in (["m1"], ["m1", "m2"]):
            t = PowerTransformY(
                search_space=self.experiment.search_space,
                experiment_data=experiment_data,
                config={"metrics": metrics},
            )
            self.assertEqual(t.metric_signatures, metrics)
            self.assertEqual(list(t.power_transforms), metrics)
            # Check that the transform is the same as if we had
            # initialized it using observations.
            t_old = PowerTransformY(
                search_space=self.experiment.search_space,
                experiment_data=self.experiment_data,
                config={"metrics": metrics},
            )
            transformed_data = t.transform_experiment_data(
                experiment_data=deepcopy(experiment_data)
            )
            transformed_data_old = t_old.transform_experiment_data(
                experiment_data=deepcopy(experiment_data)
            )
            self.assertEqual(transformed_data, transformed_data_old)
            # Compare the transformed values to transformed Observation.
            observations = t.transform_observations(
                observations=deepcopy(self.observations)
            )
            converted_obs = transformed_data.convert_to_list_of_observations()
            self.assertEqual(observations, converted_obs)
