#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import MultiObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.multi_objective_torch import MultiObjectiveTorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import pareto_frontier_evaluator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_multi_objective,
    get_branin_experiment_with_multi_objective,
    get_branin_metric,
    get_multi_type_experiment,
)


PARETO_FRONTIER_EVALUATOR_PATH = (
    f"{MultiObjectiveBotorchModel.__module__}.pareto_frontier_evaluator"
)


# Prepare mock transforms
class t1(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        for param_name in new_ss.parameters:
            new_ss.parameters[param_name]._lower += 1.0
            new_ss.parameters[param_name]._upper += 1.0
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config + 1
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            for param_name in obsf.parameters:
                obsf.parameters[param_name] += 1
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means += 1
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            for param_name in obsf.parameters:
                obsf.parameters[param_name] -= 1
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means -= 1
        return x


class t2(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        for param_name in new_ss.parameters:
            new_ss.parameters[param_name]._lower = (
                new_ss.parameters[param_name]._lower ** 2
            )
            new_ss.parameters[param_name]._upper = (
                new_ss.parameters[param_name]._upper ** 2
            )
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config ** 2
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            for param_name in obsf.parameters:
                obsf.parameters[param_name] = obsf.parameters[param_name] ** 2
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = obsd.means ** 2
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            for param_name in obsf.parameters:
                obsf.parameters[param_name] = np.sqrt(obsf.parameters[param_name])
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = np.sqrt(obsd.means)
        return x


class MultiObjectiveTorchModelBridgeTest(TestCase):
    @patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(
        (
            f"{MultiObjectiveTorchModelBridge.__module__}."
            "MultiObjectiveTorchModelBridge._fit"
        ),
        autospec=True,
    )
    def test_transform_ref_point(self, _mock_fit, _mock_predict, _mock_unwrap):
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=False
        )
        metrics = exp.optimization_config.objective.metrics
        ref_point = {metrics[0].name: 0.0, metrics[1].name: 0.0}
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            ref_point=ref_point,
        )
        self.assertIsNone(modelbridge._transformed_ref_point)
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        exp.attach_data(get_branin_data_multi_objective(trial_indices=exp.trials))
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            ref_point=ref_point,
        )
        self.assertIsNotNone(modelbridge._transformed_ref_point)
        self.assertEqual(2, len(modelbridge._transformed_ref_point))

        mixed_objective_constraints_optimization_config = OptimizationConfig(
            objective=MultiObjective(
                metrics=[get_branin_metric(name="branin_b")], minimize=False
            ),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(name="branin_a"), op=ComparisonOp.LEQ, bound=1
                )
            ],
        )
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=mixed_objective_constraints_optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            ref_point={"branin_b": 0.0},
        )
        self.assertEqual({"branin_a", "branin_b"}, modelbridge._metric_names)
        self.assertEqual(["branin_b"], modelbridge._objective_metric_names)
        self.assertIsNotNone(modelbridge._transformed_ref_point)
        self.assertEqual(1, len(modelbridge._transformed_ref_point))

    def test_pareto_frontier(self):
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=False
        )
        ref_point = {"branin_a": 0.0, "branin_b": 0.0}
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        exp.attach_data(get_branin_data_multi_objective(trial_indices=exp.trials))
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            ref_point=ref_point,
        )
        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            modelbridge.model.frontier_evaluator = wrapped_frontier_evaluator
            observed_frontier_data = modelbridge.observed_pareto_frontier(
                ref_point=ref_point
            )
            wrapped_frontier_evaluator.assert_called_once()
            self.assertEqual(1, len(observed_frontier_data))

        with self.assertRaises(ValueError):
            modelbridge.predicted_pareto_frontier(
                ref_point=ref_point, observation_features=[]
            )

        observation_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 1.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 0.0}),
        ]
        predicted_frontier_data = modelbridge.predicted_pareto_frontier(
            ref_point=ref_point, observation_features=observation_features
        )
        self.assertTrue(len(predicted_frontier_data) <= 2)

    def test_multi_type_experiment(self):
        exp = get_multi_type_experiment()
        with self.assertRaises(NotImplementedError):
            MultiObjectiveTorchModelBridge(
                experiment=exp,
                search_space=exp.search_space,
                model=MultiObjectiveBotorchModel(),
                transforms=[],
                data=exp.fetch_data(),
                ref_point={"branin_b": 0.0},
            )
