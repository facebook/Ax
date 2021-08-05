#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch
from ax.core.observation import ObservationFeatures, ObservationData
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.modelbridge_utils import (
    get_pareto_frontier_and_transformed_configs,
    pareto_frontier,
    predicted_hypervolume,
    predicted_pareto_frontier,
    observed_hypervolume,
    observed_pareto_frontier,
)
from ax.modelbridge.multi_objective_torch import MultiObjectiveTorchModelBridge
from ax.modelbridge.registry import Cont_X_trans, Y_trans, ST_MTGP_trans
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    infer_objective_thresholds,
)
from ax.models.torch.botorch_moo_defaults import pareto_frontier_evaluator
from ax.service.utils.report_utils import exp_to_df
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_multi_objective,
    get_branin_experiment_with_multi_objective,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

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
            observed_frontier = observed_pareto_frontier(
                modelbridge=modelbridge, objective_thresholds=objective_thresholds
            )
            wrapped_frontier_evaluator.assert_called_once()
            self.assertIsNone(wrapped_frontier_evaluator.call_args[1]["X"])
            self.assertEqual(1, len(observed_frontier))
            self.assertEqual(observed_frontier[0].arm_name, "0_0")

        with self.assertRaises(ValueError):
            predicted_pareto_frontier(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=[],
            )

        predicted_frontier = predicted_pareto_frontier(
            modelbridge=modelbridge,
            objective_thresholds=objective_thresholds,
            observation_features=None,
        )
        self.assertEqual(predicted_frontier[0].arm_name, "0_0")

        observation_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 1.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 0.0}),
        ]
        observation_data = [
            ObservationData(
                metric_names=["branin_b", "branin_a"],
                means=np.array([1.0, 2.0]),
                covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            ObservationData(
                metric_names=["branin_a", "branin_b"],
                means=np.array([3.0, 4.0]),
                covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
        ]
        predicted_frontier = predicted_pareto_frontier(
            modelbridge=modelbridge,
            objective_thresholds=objective_thresholds,
            observation_features=observation_features,
        )
        self.assertTrue(len(predicted_frontier) <= 2)
        self.assertIsNone(predicted_frontier[0].arm_name, None)

        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            (
                observed_frontier,
                f,
                obj_w,
                obj_t,
            ) = get_pareto_frontier_and_transformed_configs(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
                observation_data=observation_data,
            )
            wrapped_frontier_evaluator.assert_called_once()
            self.assertTrue(
                torch.equal(
                    wrapped_frontier_evaluator.call_args[1]["X"],
                    torch.tensor([[1.0, 4.0], [4.0, 1.0]]),
                )
            )
            self.assertEqual(f.shape, (1, 2))
            self.assertTrue(torch.equal(obj_w, torch.tensor([1.0, 1.0])))
            self.assertTrue(torch.equal(obj_t, torch.tensor([0.0, 0.0])))
            observed_frontier2 = pareto_frontier(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
                observation_data=observation_data,
            )
            self.assertEqual(observed_frontier, observed_frontier2)

        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            (
                observed_frontier,
                f,
                obj_w,
                obj_t,
            ) = get_pareto_frontier_and_transformed_configs(
                modelbridge=modelbridge,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
                observation_data=observation_data,
                use_model_predictions=False,
            )
            wrapped_frontier_evaluator.assert_called_once()
            self.assertIsNone(wrapped_frontier_evaluator.call_args[1]["X"])
            true_Y = torch.tensor([[9.0, 4.0], [16.0, 25.0]])
            self.assertTrue(
                torch.equal(
                    wrapped_frontier_evaluator.call_args[1]["Y"],
                    true_Y,
                )
            )
            self.assertTrue(torch.equal(f, true_Y[1:, :]))

    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    def test_hypervolume(self, _, cuda=False):
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
            transforms=[],
            experiment=exp,
            data=exp.fetch_data(),
            torch_device=torch.device("cuda" if cuda else "cpu"),
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

    def test_hypervolume_cuda(self):
        if torch.cuda.is_available():
            self.test_hypervolume(cuda=True)

    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    def test_infer_objective_thresholds(self, _, cuda=False):
        # lightweight test
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True,
            with_batch=True,
            with_status_quo=True,
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        data = exp.fetch_data()
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=Cont_X_trans + Y_trans,
            torch_device=torch.device("cuda" if cuda else "cpu"),
            experiment=exp,
            data=data,
        )
        fixed_features = ObservationFeatures(parameters={"x1": 0.0})
        search_space = exp.search_space.clone()
        param_constraints = [
            ParameterConstraint(constraint_dict={"x1": 1.0}, bound=10.0)
        ]
        outcome_constraints = [
            OutcomeConstraint(
                metric=exp.metrics["branin_a"],
                op=ComparisonOp.GEQ,
                bound=-40.0,
                relative=False,
            )
        ]
        search_space.add_parameter_constraints(param_constraints)
        exp.optimization_config.outcome_constraints = outcome_constraints
        oc = exp.optimization_config.clone()
        oc.objective._objectives[0].minimize = True
        expected_base_gen_args = modelbridge._get_transformed_gen_args(
            search_space=search_space.clone(),
            optimization_config=oc,
            fixed_features=fixed_features,
        )
        with ExitStack() as es:
            mock_model_infer_obj_t = es.enter_context(
                patch(
                    "ax.modelbridge.multi_objective_torch.infer_objective_thresholds",
                    wraps=infer_objective_thresholds,
                )
            )
            mock_get_transformed_gen_args = es.enter_context(
                patch.object(
                    modelbridge,
                    "_get_transformed_gen_args",
                    wraps=modelbridge._get_transformed_gen_args,
                )
            )
            mock_get_transformed_model_gen_args = es.enter_context(
                patch.object(
                    modelbridge,
                    "_get_transformed_model_gen_args",
                    wraps=modelbridge._get_transformed_model_gen_args,
                )
            )
            mock_untransform_objective_thresholds = es.enter_context(
                patch.object(
                    modelbridge,
                    "untransform_objective_thresholds",
                    wraps=modelbridge.untransform_objective_thresholds,
                )
            )
            obj_thresholds = modelbridge.infer_objective_thresholds(
                search_space=search_space,
                optimization_config=oc,
                fixed_features=fixed_features,
            )
            expected_obj_weights = torch.tensor([-1.0, 1.0])
            ckwargs = mock_model_infer_obj_t.call_args[1]
            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], expected_obj_weights)
            )
            # check that transforms have been applied (at least UnitX)
            self.assertEqual(ckwargs["bounds"], [(0.0, 1.0), (0.0, 1.0)])
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], torch.tensor([[-1.0, 0.0]])))
            self.assertTrue(torch.equal(oc[1], torch.tensor([[45.0]])))
            lc = ckwargs["linear_constraints"]
            self.assertTrue(torch.equal(lc[0], torch.tensor([[15.0, 0.0]])))
            self.assertTrue(torch.equal(lc[1], torch.tensor([[15.0]])))
            self.assertEqual(ckwargs["fixed_features"], {0: 1.0 / 3.0})
            mock_get_transformed_gen_args.assert_called_once()
            mock_get_transformed_model_gen_args.assert_called_once_with(
                search_space=expected_base_gen_args.search_space,
                fixed_features=expected_base_gen_args.fixed_features,
                pending_observations=expected_base_gen_args.pending_observations,
                optimization_config=expected_base_gen_args.optimization_config,
            )
            mock_untransform_objective_thresholds.assert_called_once()
            ckwargs = mock_untransform_objective_thresholds.call_args[1]

            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], expected_obj_weights)
            )
            self.assertEqual(ckwargs["bounds"], [(0.0, 1.0), (0.0, 1.0)])
            self.assertEqual(ckwargs["fixed_features"], {0: 1.0 / 3.0})
        self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
        self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
        self.assertEqual(obj_thresholds[0].op, ComparisonOp.LEQ)
        self.assertEqual(obj_thresholds[1].op, ComparisonOp.GEQ)
        self.assertFalse(obj_thresholds[0].relative)
        self.assertFalse(obj_thresholds[1].relative)
        df = exp_to_df(exp)
        Y = np.stack([df.branin_a.values, df.branin_b.values]).T
        Y = torch.from_numpy(Y)
        Y[:, 0] *= -1
        pareto_Y = Y[is_non_dominated(Y)]
        nadir = pareto_Y.min(dim=0).values
        self.assertTrue(
            np.all(
                np.array([-obj_thresholds[0].bound, obj_thresholds[1].bound])
                < nadir.numpy()
            )
        )
        # test using MTGP
        sobol_generator = get_sobol(search_space=exp.search_space)
        sobol_run = sobol_generator.gen(n=5)
        trial = exp.new_batch_trial(optimize_for_power=True)
        trial.add_generator_run(sobol_run)
        trial.mark_running(no_runner_required=True).mark_completed()
        data = exp.fetch_data()
        modelbridge = MultiObjectiveTorchModelBridge(
            search_space=exp.search_space,
            model=MultiObjectiveBotorchModel(),
            optimization_config=exp.optimization_config,
            transforms=ST_MTGP_trans,
            experiment=exp,
            data=data,
        )
        fixed_features = ObservationFeatures(parameters={}, trial_index=1)
        expected_base_gen_args = modelbridge._get_transformed_gen_args(
            search_space=search_space.clone(),
            optimization_config=exp.optimization_config,
            fixed_features=fixed_features,
        )
        with self.assertRaises(ValueError):
            # Check that a ValueError is raised when MTGP is being used
            # and trial_index is not specified as a fixed features.
            # Note: this error is raised by StratifiedStandardizeY
            modelbridge.infer_objective_thresholds(
                search_space=search_space,
                optimization_config=exp.optimization_config,
            )
        with ExitStack() as es:
            mock_model_infer_obj_t = es.enter_context(
                patch(
                    "ax.modelbridge.multi_objective_torch.infer_objective_thresholds",
                    wraps=infer_objective_thresholds,
                )
            )
            mock_untransform_objective_thresholds = es.enter_context(
                patch.object(
                    modelbridge,
                    "untransform_objective_thresholds",
                    wraps=modelbridge.untransform_objective_thresholds,
                )
            )
            obj_thresholds = modelbridge.infer_objective_thresholds(
                search_space=search_space,
                optimization_config=exp.optimization_config,
                fixed_features=fixed_features,
            )
            ckwargs = mock_model_infer_obj_t.call_args[1]
            self.assertEqual(ckwargs["fixed_features"], {2: 1.0})
            mock_untransform_objective_thresholds.assert_called_once()
            ckwargs = mock_untransform_objective_thresholds.call_args[1]
            self.assertEqual(ckwargs["fixed_features"], {2: 1.0})
        self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
        self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
        self.assertEqual(obj_thresholds[0].op, ComparisonOp.GEQ)
        self.assertEqual(obj_thresholds[1].op, ComparisonOp.GEQ)
        self.assertFalse(obj_thresholds[0].relative)
        self.assertFalse(obj_thresholds[1].relative)
        df = exp_to_df(exp)
        trial_mask = df.trial_index == 1
        Y = np.stack([df.branin_a.values[trial_mask], df.branin_b.values[trial_mask]]).T
        Y = torch.from_numpy(Y)
        pareto_Y = Y[is_non_dominated(Y)]
        nadir = pareto_Y.min(dim=0).values
        self.assertTrue(
            np.all(
                np.array([obj_thresholds[0].bound, obj_thresholds[1].bound])
                < nadir.numpy()
            )
        )
