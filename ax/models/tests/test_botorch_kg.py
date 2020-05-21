#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from ax.models.torch.botorch_kg import (
    KnowledgeGradient,
    _get_objective,
    _instantiate_KG,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    LinearMCObjective,
    ScalarizedObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


class KnowledgeGradientTest(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.double

        self.Xs = [
            torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=self.dtype, device=self.device
            )
        ]
        self.Ys = [torch.tensor([[3.0], [4.0]], dtype=self.dtype, device=self.device)]
        self.Yvars = [
            torch.tensor([[0.0], [2.0]], dtype=self.dtype, device=self.device)
        ]
        self.bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.acq_options = {"num_fantasies": 30, "mc_samples": 30}
        self.objective_weights = torch.tensor(
            [1.0], dtype=self.dtype, device=self.device
        )
        self.optimizer_options = {
            "num_restarts": 12,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
        }
        self.optimize_acqf = "ax.models.torch.botorch_kg.optimize_acqf"

    def test_KnowledgeGradient(self):

        model = KnowledgeGradient()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            task_features=[],
            fidelity_features=[],
        )

        n = 2

        X_dummy = torch.rand(1, n, 4, dtype=self.dtype, device=self.device)
        acq_dummy = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        with mock.patch(self.optimize_acqf) as mock_optimize_acqf:
            mock_optimize_acqf.side_effect = [(X_dummy, acq_dummy)]
            Xgen, wgen, _, __ = model.gen(
                n=n,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": self.acq_options,
                    "optimizer_kwargs": self.optimizer_options,
                },
            )
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=self.dtype)))

            # called once, the best point call is not caught by mock
            mock_optimize_acqf.assert_called_once()

        ini_dummy = torch.rand(10, 32, 3, dtype=self.dtype, device=self.device)
        optimizer_options2 = {
            "num_restarts": 1,
            "raw_samples": 1,
            "maxiter": 5,
            "batch_limit": 1,
            "partial_restarts": 2,
        }
        with mock.patch(
            "ax.models.torch.botorch_kg.gen_one_shot_kg_initial_conditions",
            return_value=ini_dummy,
        ) as mock_warmstart_initialization:
            Xgen, wgen, _, __ = model.gen(
                n=n,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": self.acq_options,
                    "optimizer_kwargs": optimizer_options2,
                },
            )
            mock_warmstart_initialization.assert_called_once()

        obj = ScalarizedObjective(weights=self.objective_weights)
        dummy_acq = PosteriorMean(model=model.model, objective=obj)
        with mock.patch(
            "ax.models.torch.botorch_kg.PosteriorMean", return_value=dummy_acq
        ) as mock_posterior_mean:
            Xgen, wgen, _, __ = model.gen(
                n=n,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": self.acq_options,
                    "optimizer_kwargs": optimizer_options2,
                },
            )
            self.assertEqual(mock_posterior_mean.call_count, 2)

        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            bounds=self.bounds, objective_weights=self.objective_weights
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # test error message
        linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        with self.assertRaises(UnsupportedError):
            Xgen, wgen = model.gen(
                n=n,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=linear_constraints,
            )

    def test_KnowledgeGradient_multifidelity(self):
        model = KnowledgeGradient()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=[],
            feature_names=self.feature_names,
            metric_names=[],
            fidelity_features=[2],
        )

        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            target_fidelities={2: 5.0},
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # check error when no target fidelities are specified
        with self.assertRaises(RuntimeError):
            model.best_point(
                bounds=self.bounds, objective_weights=self.objective_weights
            )

        # check generation
        n = 2
        X_dummy = torch.zeros(1, n, 3, dtype=self.dtype, device=self.device)
        acq_dummy = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        dummy = (X_dummy, acq_dummy)
        with mock.patch(self.optimize_acqf, side_effect=[dummy]) as mock_optimize_acqf:
            Xgen, wgen, _, __ = model.gen(
                n=n,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": self.acq_options,
                    "optimizer_kwargs": self.optimizer_options,
                },
                target_fidelities={2: 5.0},
            )
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=self.dtype)))
            mock_optimize_acqf.assert_called()  # called twice, once for best_point

        # test error message
        linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        with self.assertRaises(UnsupportedError):
            xbest = model.best_point(
                bounds=self.bounds,
                linear_constraints=linear_constraints,
                objective_weights=self.objective_weights,
                target_fidelities={2: 1.0},
            )

    def test_KnowledgeGradient_helpers(self):

        model = KnowledgeGradient()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            task_features=[],
            fidelity_features=[],
        )

        # test _instantiate_KG
        objective = ScalarizedObjective(weights=self.objective_weights)
        X_dummy = torch.ones(1, 3, dtype=self.dtype, device=self.device)

        # test acquisition setting
        acq_function = _instantiate_KG(
            model=model.model, objective=objective, n_fantasies=10, qmc=True
        )
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsInstance(acq_function.objective, ScalarizedObjective)
        self.assertEqual(acq_function.num_fantasies, 10)

        acq_function = _instantiate_KG(
            model=model.model, objective=objective, n_fantasies=10, qmc=False
        )
        self.assertIsInstance(acq_function.sampler, IIDNormalSampler)

        acq_function = _instantiate_KG(
            model=model.model, objective=objective, qmc=False
        )
        self.assertIsNone(acq_function.inner_sampler)

        acq_function = _instantiate_KG(
            model=model.model, objective=objective, qmc=True, X_pending=X_dummy
        )
        self.assertIsNone(acq_function.inner_sampler)
        self.assertTrue(torch.equal(acq_function.X_pending, X_dummy))

        # test _get_obj()
        outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        objective_weights = torch.ones(1, dtype=self.dtype, device=self.device)
        self.assertIsInstance(
            _get_objective(
                model=model.model,
                outcome_constraints=outcome_constraints,
                objective_weights=objective_weights,
                X_observed=X_dummy,
            ),
            ConstrainedMCObjective,
        )

        # test _get_best_point_acqf
        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
        )
        self.assertIsInstance(acq_function, qSimpleRegret)
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsNone(non_fixed_idcs)

        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
            qmc=False,
        )
        self.assertIsInstance(acq_function.sampler, IIDNormalSampler)
        self.assertIsNone(non_fixed_idcs)

        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_dummy,
                target_fidelities={1: 1.0},
            )

        # multi-fidelity tests

        model = KnowledgeGradient()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=[],
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=[-1],
        )

        acq_function = _instantiate_KG(
            model=model.model,
            objective=objective,
            target_fidelities={2: 1.0},
            current_value=0,
        )
        self.assertIsInstance(acq_function, qMultiFidelityKnowledgeGradient)

        acq_function = _instantiate_KG(
            model=model.model,
            objective=LinearMCObjective(weights=self.objective_weights),
        )
        self.assertIsInstance(acq_function.inner_sampler, SobolQMCNormalSampler)

        # test error that target fidelity and fidelity weight indices must match
        with self.assertRaises(RuntimeError):
            _instantiate_KG(
                model=model.model,
                objective=objective,
                target_fidelities={1: 1.0},
                fidelity_weights={2: 1.0},
                current_value=0,
            )

        # test _get_best_point_acqf
        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
            target_fidelities={2: 1.0},
        )
        self.assertIsInstance(acq_function, FixedFeatureAcquisitionFunction)
        self.assertIsInstance(acq_function.acq_func.sampler, SobolQMCNormalSampler)
        self.assertEqual(non_fixed_idcs, [0, 1])

        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
            target_fidelities={2: 1.0},
            qmc=False,
        )
        self.assertIsInstance(acq_function, FixedFeatureAcquisitionFunction)
        self.assertIsInstance(acq_function.acq_func.sampler, IIDNormalSampler)
        self.assertEqual(non_fixed_idcs, [0, 1])

        # test error that fixed features are provided
        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_dummy,
                qmc=False,
            )

        # test error if fixed features are also fidelity features
        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_dummy,
                fixed_features={2: 2.0},
                target_fidelities={2: 1.0},
                qmc=False,
            )

        # TODO: Test subsetting multi-output model
