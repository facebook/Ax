#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import torch
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.log_y import (
    lognorm_to_norm,
    LogY,
    match_ci_width,
    norm_to_lognorm,
)
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.types import TConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_outcome_constraint,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pandas.testing import assert_frame_equal


class LogYTransformTest(TestCase):
    def test_Init(self) -> None:
        # test default init with no config (no-op transform)
        tf = LogY()
        self.assertEqual(len(tf.metric_signatures), 0)
        # test init with empty metrics list (no-op transform)
        tf = LogY(config={"metrics": []})
        self.assertEqual(len(tf.metric_signatures), 0)
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
        # Test with variance (1D array)
        test_mean = np.array([1.0])
        test_var = np.array([0.1])
        expected_mean, expected_var = match_ci_width(
            mean=test_mean, sem=None, variance=test_var, transform=np.log
        )
        self.assertEqual(
            tf._transform(test_mean, test_var),
            (expected_mean, expected_var),
        )
        test_mean_exp = np.array([0.0])
        test_var_exp = np.array([0.1])
        expected_mean_exp, expected_var_exp = match_ci_width(
            mean=test_mean_exp, sem=None, variance=test_var_exp, transform=np.exp
        )
        self.assertEqual(
            tf._untransform(test_mean_exp, test_var_exp),
            (expected_mean_exp, expected_var_exp),
        )

    def test_TransformObservations(self) -> None:
        obsd_with_noise = ObservationData(
            metric_signatures=["m1", "m2", "m3"],
            means=np.array([0.5, 1.0, 1.0]),
            covariance=np.diag(np.array([1.0, 1.0, np.exp(1) - 1])),
        )
        # test default transform
        obsd1_t = ObservationData(
            metric_signatures=["m1", "m2", "m3"],
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
        # test raise on full covariance for single metric (no longer supported)
        Z = np.zeros((3, 3))
        Z[0, 2] = np.sqrt(np.exp(1)) - 1
        Z[2, 0] = np.sqrt(np.exp(1)) - 1
        obsd1 = ObservationData(
            metric_signatures=["m3", "m3", "m3"],
            means=np.ones(3),
            covariance=np.diag(np.ones(3) * (np.exp(1) - 1)) + Z,
        )
        with self.assertRaises(NotImplementedError):
            tf._transform_observation_data([obsd1])

        # Test with unknown noise.
        obsd_without_noise = ObservationData(
            metric_signatures=["m1", "m2"],
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

    def test_TransformOptimizationConfig(self) -> None:
        # basic test
        m1 = Metric(name="m1")
        objective_m1 = Objective(metric=m1, minimize=False)
        oc = OptimizationConfig(objective=objective_m1, outcome_constraints=[])
        tf = LogY(search_space=None, config={"metrics": ["m1"]})
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
            "bound isn't positive, got: 0.0.",
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
        tf = LogY(search_space=None, config={"metrics": ["m1"]})
        oc_tf = tf.transform_optimization_config(deepcopy(oc), None, None)
        oc.objective_thresholds[0].bound = math.log(1.234)
        self.assertEqual(oc_tf, oc)

    def _base_test_transform_experiment_data(
        self,
        config: TConfig,
        observations: list[list[float]],
        sems: list[list[float]],
        expected_m1_means: npt.NDArray[np.float64],
        expected_m1_sems: npt.NDArray[np.float64],
        expected_m2_means: npt.NDArray[np.float64],
        expected_m2_sems: npt.NDArray[np.float64],
    ) -> None:
        # Setup: Create experiment data with observations
        experiment_data = extract_experiment_data(
            experiment=get_experiment_with_observations(
                observations=observations, sems=sems
            ),
            data_loader_config=DataLoaderConfig(),
        )

        # Execute: Transform the experiment data
        tf = LogY(config=config)
        transformed_data = tf.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Assert: Arm data should be identical
        assert_frame_equal(
            experiment_data.arm_data,
            transformed_data.arm_data,
        )

        # Assert: Observation data should be transformed correctly
        observation_data = transformed_data.observation_data

        # Check m1 transformation
        self.assertTrue(
            np.allclose(
                observation_data["mean", "m1"].values,
                expected_m1_means,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                observation_data["sem", "m1"].values,
                expected_m1_sems,
                equal_nan=True,
            )
        )

        # Check m2 transformation
        self.assertTrue(
            np.allclose(
                observation_data["mean", "m2"].values,
                expected_m2_means,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                observation_data["sem", "m2"].values,
                expected_m2_sems,
                equal_nan=True,
            )
        )

    def test_transform_experiment_data(self) -> None:
        # For m1: means are [1.0, 1.0], variances are [0.25, 0.25]
        m1_var = 0.25
        m1_mean = 1.0
        expected_m1_mean = np.log(m1_mean) - 0.5 * np.log(1 + m1_var / (m1_mean**2))
        expected_m1_var = np.log(1 + m1_var / (m1_mean**2))

        # For m2: means are [2.0, 3.0], variances are [1.0, 2.25]
        m2_mean_0 = 2.0
        m2_var_0 = 1.0
        expected_m2_mean_0 = np.log(m2_mean_0) - 0.5 * np.log(
            1 + m2_var_0 / (m2_mean_0**2)
        )
        expected_m2_var_0 = np.log(1 + m2_var_0 / (m2_mean_0**2))

        m2_mean_1 = 3.0
        m2_var_1 = 2.25
        expected_m2_mean_1 = np.log(m2_mean_1) - 0.5 * np.log(
            1 + m2_var_1 / (m2_mean_1**2)
        )
        expected_m2_var_1 = np.log(1 + m2_var_1 / (m2_mean_1**2))

        self._base_test_transform_experiment_data(
            config={"metrics": ["m1", "m2"]},
            observations=[[1.0, 2.0], [1.0, 3.0]],
            sems=[[0.5, 1.0], [0.5, 1.5]],
            expected_m1_means=np.array([expected_m1_mean, expected_m1_mean]),
            expected_m1_sems=np.array(
                [np.sqrt(expected_m1_var), np.sqrt(expected_m1_var)]
            ),
            expected_m2_means=np.array([expected_m2_mean_0, expected_m2_mean_1]),
            expected_m2_sems=np.array(
                [np.sqrt(expected_m2_var_0), np.sqrt(expected_m2_var_1)]
            ),
        )

    def test_transform_experiment_data_with_match_ci_width(self) -> None:
        # m1 should be transformed using match_ci_width
        expected_m1_mean, expected_m1_var = match_ci_width(
            mean=np.array([1.0]),
            sem=None,
            variance=np.array([0.25]),
            transform=np.log,
        )

        self._base_test_transform_experiment_data(
            config={"metrics": ["m1"], "match_ci_width": True},
            observations=[[1.0, 2.0]],
            sems=[[0.5, 1.0]],
            expected_m1_means=expected_m1_mean,
            expected_m1_sems=np.sqrt(expected_m1_var),
            expected_m2_means=np.array([2.0]),  # m2 should remain unchanged
            expected_m2_sems=np.array([1.0]),
        )

    def test_transform_experiment_data_with_all_nan_sems(self) -> None:
        # When SEM is NaN, the transform should use zero variance
        self._base_test_transform_experiment_data(
            config={"metrics": ["m1", "m2"]},
            observations=[[1.0, 2.0], [np.e, 3.0]],
            sems=[[np.nan, np.nan], [np.nan, np.nan]],
            expected_m1_means=np.array([0.0, 1.0]),  # log([1.0, e]) = [0.0, 1.0]
            expected_m1_sems=np.array([np.nan, np.nan]),
            expected_m2_means=np.log(np.array([2.0, 3.0])),
            expected_m2_sems=np.array([np.nan, np.nan]),
        )

    @mock_botorch_optimize
    def test_gen_with_log_y_transform(self) -> None:
        observations = [[1.0, 2.0], [1.0, 3.0]]
        experiment = get_experiment_with_observations(
            observations=observations, scalarized=True
        )
        generator = BoTorchGenerator()
        adapter = TorchAdapter(
            experiment=experiment,
            generator=generator,
            transforms=[LogY],
            transform_configs={
                "LogY": {"metrics": ["m1", "m2"]},
            },
        )
        adapter.gen(n=1)
        # Check that model training data is log-transformed.
        model = generator.surrogate.model
        self.assertAllClose(
            model.train_targets, torch.tensor(observations, dtype=torch.float64).log().T
        )


class LogNormTest(TestCase):
    def test_lognorm_to_norm(self) -> None:
        # Test with variance (diagonal elements only)
        mu_ln = np.ones(3)
        var_ln = mu_ln * (np.exp(1) - 1)
        mu_n, var_n = lognorm_to_norm(mu_ln, var_ln)
        self.assertTrue(np.allclose(mu_n, -0.5 * np.ones_like(mu_n)))
        self.assertTrue(np.allclose(var_n, np.ones(3)))

    def test_norm_to_lognorm(self) -> None:
        # Test with variance (diagonal elements only)
        mu_n = -0.5 * np.ones(3)
        var_n = np.ones(3)
        mu_ln, var_ln = norm_to_lognorm(mu_n, var_n)
        self.assertTrue(np.allclose(mu_ln, np.ones(3)))
        self.assertTrue(np.allclose(var_ln, (np.exp(1) - 1) * np.ones(3)))
