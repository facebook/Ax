#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.base import Transform
from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class SomeTransform(Transform):
    pass


# Create a custom transform that doubles parameter values
class DoubleParameterTransform(Transform):
    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obs_feat in observation_features:
            for param_name, param_value in obs_feat.parameters.items():
                obs_feat.parameters[param_name] = float(param_value) * 2
        return observation_features


class TransformsTest(TestCase):
    def test_IdentityTransform(self) -> None:
        # Test that the identity transform does not mutate anything
        t = Transform(MagicMock(), MagicMock())
        x = MagicMock()
        ys = []
        ys.append(t.transform_search_space(x))
        ys.append(t.transform_observation_features(x))
        ys.append(t._transform_observation_data(x))
        ys.append(t.untransform_observation_features(x))
        ys.append(t._untransform_observation_data(x))
        self.assertEqual(len(x.mock_calls), 0)
        for y in ys:
            self.assertEqual(y, x)

        # Test transform_optimization_config separately since it has special behavior
        # for pruning_target_parameterization
        x_opt_config = MagicMock()
        x_opt_config.pruning_target_parameterization = (
            None  # No target arm means no transformation
        )
        y_opt_config = t.transform_optimization_config(x_opt_config, x, x)
        self.assertEqual(y_opt_config, x_opt_config)

    def test_TransformObservations(self) -> None:
        # Test that this is an identity transform
        means = np.array([3.0, 4.0])
        metric_signatures = ["a", "b"]
        covariance = np.array([[1.0, 2.0], [3.0, 4.0]])
        parameters = {"x": 1.0, "y": "cat"}
        arm_name = "armmy"
        observation = Observation(
            features=ObservationFeatures(parameters=parameters),  # pyre-ignore
            data=ObservationData(
                metric_signatures=metric_signatures, means=means, covariance=covariance
            ),
            arm_name=arm_name,
        )
        t = Transform()
        obs1 = t.transform_observations([deepcopy(observation)])[0]
        obs2 = t.untransform_observations([deepcopy(obs1)])[0]
        for obs in [obs1, obs2]:
            self.assertTrue(np.array_equal(obs.data.means, means))
            self.assertTrue(np.array_equal(obs.data.covariance, covariance))
            self.assertEqual(obs.data.metric_signatures, metric_signatures)
            self.assertEqual(obs.features.parameters, parameters)
            self.assertEqual(obs.arm_name, arm_name)

    def test_with_experiment_data(self) -> None:
        experiment = get_branin_experiment(with_completed_batch=True)
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        t = SomeTransform(experiment_data=experiment_data)
        # Errors out since no_op_for_experiment_data defaults to False.
        with self.assertRaisesRegex(NotImplementedError, "transform_experiment_data"):
            t.transform_experiment_data(experiment_data=experiment_data)
        # No-op when no_op_for_experiment_data is True.
        t.no_op_for_experiment_data = True
        self.assertIs(
            t.transform_experiment_data(experiment_data=experiment_data),
            experiment_data,
        )
        # Base transform itself doesn't error out.
        t = Transform(experiment_data=experiment_data)
        self.assertFalse(t.no_op_for_experiment_data)
        self.assertIs(
            t.transform_experiment_data(experiment_data=experiment_data),
            experiment_data,
        )

    def test_transform_optimization_config_with_pruning_target_parameterization(
        self,
    ) -> None:
        # Setup: create optimization config with target arm and transform that
        # modifies parameters
        pruning_target_parameterization = Arm(parameters={"x1": 2.5, "x2": 7.5})
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("m1"), minimize=False),
            pruning_target_parameterization=pruning_target_parameterization,
        )

        transform = DoubleParameterTransform()

        # Execute: transform the optimization config
        transformed_config = transform.transform_optimization_config(
            optimization_config
        )

        # Assert: confirm target arm parameters are correctly transformed
        self.assertIsNotNone(transformed_config.pruning_target_parameterization)
        expected_parameters = {"x1": 5.0, "x2": 15.0}  # doubled values
        self.assertEqual(
            transformed_config.pruning_target_parameterization.parameters,
            expected_parameters,
        )
        # Confirm the optimization config object is the same (in-place transformation)
        self.assertIs(transformed_config, optimization_config)

    def test_transform_optimization_config_without_pruning_target_parameterization(
        self,
    ) -> None:
        # Setup: create optimization config without target arm
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("m1"), minimize=False),
            pruning_target_parameterization=None,
        )

        transform = Transform()

        # Execute: transform the optimization config
        transformed_config = transform.transform_optimization_config(
            optimization_config
        )

        # Assert: confirm no target arm exists and config remains unchanged
        self.assertIsNone(transformed_config.pruning_target_parameterization)
        self.assertIs(transformed_config, optimization_config)

    def test_transform_optimization_config_preserves_other_fields(self) -> None:
        # Setup: create optimization config with target arm and other fields
        from ax.core.outcome_constraint import OutcomeConstraint
        from ax.core.types import ComparisonOp

        pruning_target_parameterization = Arm(parameters={"x1": 1.0, "x2": 2.0})
        outcome_constraints = [
            OutcomeConstraint(
                metric=Metric("m2"), op=ComparisonOp.LEQ, bound=10.0, relative=False
            )
        ]
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("m1"), minimize=True),
            outcome_constraints=outcome_constraints,
            pruning_target_parameterization=pruning_target_parameterization,
        )

        transform = DoubleParameterTransform()

        # Execute: transform the optimization config
        transformed_config = transform.transform_optimization_config(
            optimization_config
        )

        # Assert: confirm target arm is transformed but other fields are preserved
        self.assertIsNotNone(transformed_config.pruning_target_parameterization)
        expected_parameters = {"x1": 2.0, "x2": 4.0}  # incremented values
        self.assertEqual(
            transformed_config.pruning_target_parameterization.parameters,
            expected_parameters,
        )

        # Confirm other fields are preserved
        self.assertEqual(transformed_config.objective.metric.name, "m1")
        self.assertTrue(transformed_config.objective.minimize)
        self.assertEqual(len(transformed_config.outcome_constraints), 1)
        self.assertEqual(transformed_config.outcome_constraints[0].metric.name, "m2")
        self.assertEqual(transformed_config.outcome_constraints[0].bound, 10.0)
