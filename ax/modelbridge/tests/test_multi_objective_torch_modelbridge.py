#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.modelbridge.modelbridge_utils import (
    predicted_hypervolume,
    predicted_pareto_frontier,
    observed_hypervolume,
    observed_pareto_frontier,
)
from ax.modelbridge.multi_objective_torch import MultiObjectiveTorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import pareto_frontier_evaluator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_multi_objective,
    get_branin_experiment_with_multi_objective,
)


PARETO_FRONTIER_EVALUATOR_PATH = (
    f"{pareto_frontier_evaluator.__module__}.pareto_frontier_evaluator"
)
STUBS_PATH = get_branin_experiment_with_multi_objective.__module__


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
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    def test_pareto_frontier(self, _):
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        metrics_dict = exp.optimization_config.metrics
        objective_thresholds = [
            ObjectiveThreshold(
                metric=metrics_dict["branin_a"],
                bound=0.0,
                relative=False,
                op=ComparisonOp.GEQ,
            ),
            ObjectiveThreshold(
                metric=metrics_dict["branin_b"],
                bound=0.0,
                relative=False,
                op=ComparisonOp.GEQ,
            ),
        ]
        exp.optimization_config = exp.optimization_config.clone_with_args(
            objective_thresholds=objective_thresholds
        )
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            objective_thresholds=objective_thresholds,
        )
        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            modelbridge.model.frontier_evaluator = wrapped_frontier_evaluator
            observed_frontier_data = observed_pareto_frontier(
                modelbridge=modelbridge, objective_thresholds=objective_thresholds
            )
            wrapped_frontier_evaluator.assert_called_once()
            self.assertEqual(1, len(observed_frontier_data))

        with self.assertRaises(ValueError):
            predicted_pareto_frontier(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=[],
            )

        observation_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 1.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 0.0}),
        ]
        predicted_frontier_data = predicted_pareto_frontier(
            modelbridge=modelbridge,
            objective_thresholds=objective_thresholds,
            observation_features=observation_features,
        )
        self.assertTrue(len(predicted_frontier_data) <= 2)

    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    def test_hypervolume(self, _):
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        metrics_dict = exp.optimization_config.metrics
        objective_thresholds = [
            ObjectiveThreshold(
                metric=metrics_dict["branin_a"],
                bound=0.0,
                relative=False,
                op=ComparisonOp.GEQ,
            ),
            ObjectiveThreshold(
                metric=metrics_dict["branin_b"],
                bound=0.0,
                relative=False,
                op=ComparisonOp.GEQ,
            ),
        ]
        optimization_config = exp.optimization_config.clone_with_args(
            objective_thresholds=objective_thresholds
        )
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=optimization_config,
            transforms=[t1, t2],
            experiment=exp,
            data=exp.fetch_data(),
            objective_thresholds=objective_thresholds,
        )
        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            modelbridge.model.frontier_evaluator = wrapped_frontier_evaluator
            hv = observed_hypervolume(
                modelbridge=modelbridge, objective_thresholds=objective_thresholds
            )
            expected_hv = 25  # (5 - 0) * (5 - 0)
            wrapped_frontier_evaluator.assert_called_once()
            self.assertEqual(expected_hv, hv)

        with self.assertRaises(ValueError):
            predicted_hypervolume(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=[],
            )

        observation_features = [
            ObservationFeatures(parameters={"x1": 1.0, "x2": 2.0}),
            ObservationFeatures(parameters={"x1": 2.0, "x2": 1.0}),
        ]
        predicted_hv = predicted_hypervolume(
            modelbridge=modelbridge,
            objective_thresholds=objective_thresholds,
            observation_features=observation_features,
        )
        self.assertTrue(predicted_hv >= 0)
