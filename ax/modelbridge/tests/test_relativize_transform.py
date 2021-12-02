# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

import numpy as np
from ax.core.observation import ObservationFeatures, ObservationData
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.metrics.branin import BraninMetric
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.relativize import Relativize
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_multi_objective_optimization_config,
    get_branin_optimization_config,
    get_search_space,
)


class RelativizeDataTest(TestCase):
    def test_relativize_transform_requires_a_modelbridge(self):
        with self.assertRaisesRegex(ValueError, "modelbridge"):
            Relativize(
                search_space=None,
                observation_features=[],
                observation_data=[],
            )

    def test_relativize_transform_requires_a_modelbridge_to_have_status_quo_data(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        self.assertIsNone(sobol.status_quo)
        with self.assertRaisesRegex(ValueError, "status quo data"):
            Relativize(
                search_space=None,
                observation_features=[],
                observation_data=[],
                modelbridge=sobol,
            ).transform_observation_data(
                observation_data=[
                    ObservationData(
                        metric_names=["foo"],
                        means=np.array([2]),
                        covariance=np.array([[0.1]]),
                    )
                ],
                observation_features=[ObservationFeatures(parameters={"x": 1})],
            )

    def test_relativize_transform_observation_data(self):
        obs_data = [
            ObservationData(
                metric_names=["foobar", "foobaz"],
                means=np.array([2, 5]),
                covariance=np.array([[0.1, 0.0], [0.0, 0.2]]),
            ),
            ObservationData(
                metric_names=["foobar", "foobaz"],
                means=np.array([1.0, 10.0]),
                covariance=np.array([[0.3, 0.0], [0.0, 0.4]]),
            ),
        ]
        obs_features = [
            ObservationFeatures(parameters={"x": 1}),
            ObservationFeatures(parameters={"x": 2}),
        ]
        modelbridge = Mock(status_quo=Mock(data=obs_data[0]))
        results = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=modelbridge,
        ).transform_observation_data(obs_data, obs_features)
        self.assertEqual(results[0].metric_names, ["foobar", "foobaz"])
        self.assertTrue(
            np.allclose(results[0].means, np.array([0.0, 0.0])), results[0].means
        )
        self.assertTrue(
            np.allclose(results[0].covariance, np.array([[500.0, 0.0], [0.0, 160.0]])),
            results[0].covariance,
        )
        self.assertEqual(results[1].metric_names, ["foobar", "foobaz"])
        self.assertTrue(
            np.allclose(results[1].means, np.array([-51.25, 98.4])), results[1].means
        )
        self.assertTrue(
            np.allclose(results[1].covariance, np.array([[812.5, 0.0], [0.0, 480.0]])),
            results[1].covariance,
        )


class RelativizeDataOptConfigTest(TestCase):
    def test_transform_optimization_config_without_constraints(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        relativize = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=sobol,
        )
        optimization_config = get_branin_optimization_config()
        new_config = relativize.transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)

    def test_transform_optimization_config_with_relative_constraints(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        relativize = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=sobol,
        )
        optimization_config = get_branin_optimization_config()
        optimization_config.outcome_constraints = [
            OutcomeConstraint(
                metric=BraninMetric("b2", ["x2", "x1"]),
                op=ComparisonOp.GEQ,
                bound=-200.0,
                relative=True,
            )
        ]
        new_config = relativize.transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)
        self.assertEqual(
            new_config.outcome_constraints[0].bound,
            optimization_config.outcome_constraints[0].bound,
        )
        self.assertFalse(new_config.outcome_constraints[0].relative)

    def test_transform_optimization_config_with_non_relative_constraints(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        relativize = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=sobol,
        )
        optimization_config = get_branin_optimization_config()
        optimization_config.outcome_constraints = [
            OutcomeConstraint(
                metric=BraninMetric("b2", ["x2", "x1"]),
                op=ComparisonOp.GEQ,
                bound=-200.0,
                relative=False,
            )
        ]
        with self.assertRaisesRegex(ValueError, "All constraints must be relative"):
            relativize.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=None,
                fixed_features=Mock(),
            )

    def test_transform_optimization_config_with_relative_thresholds(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        relativize = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=sobol,
        )
        optimization_config = get_branin_multi_objective_optimization_config(
            has_objective_thresholds=True,
        )
        for threshold in optimization_config.objective_thresholds:
            threshold.relative = True

        new_config = relativize.transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=None,
            fixed_features=Mock(),
        )
        self.assertEqual(new_config.objective, optimization_config.objective)
        self.assertEqual(
            new_config.objective_thresholds[0].bound,
            optimization_config.objective_thresholds[0].bound,
        )
        self.assertFalse(new_config.objective_thresholds[0].relative)
        self.assertEqual(
            new_config.objective_thresholds[1].bound,
            optimization_config.objective_thresholds[1].bound,
        )
        self.assertFalse(new_config.objective_thresholds[1].relative)

    def test_transform_optimization_config_with_non_relative_thresholds(self):
        sobol = Models.SOBOL(search_space=get_search_space())
        relativize = Relativize(
            search_space=None,
            observation_features=[],
            observation_data=[],
            modelbridge=sobol,
        )
        optimization_config = get_branin_multi_objective_optimization_config(
            has_objective_thresholds=True,
        )
        optimization_config.objective_thresholds[1].relative = False

        with self.assertRaisesRegex(
            ValueError, "All objective thresholds must be relative"
        ):
            relativize.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=None,
                fixed_features=Mock(),
            )
