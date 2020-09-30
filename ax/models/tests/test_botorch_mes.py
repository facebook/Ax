#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from ax.models.torch.botorch_mes import MaxValueEntropySearch, _instantiate_MES
from ax.utils.common.testutils import TestCase
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.samplers import SobolQMCNormalSampler


class MaxValueEntropySearchTest(TestCase):
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
        self.acq_options = {"num_fantasies": 30, "candidate_size": 100}
        self.objective_weights = torch.tensor(
            [1.0], dtype=self.dtype, device=self.device
        )
        self.optimizer_options = {
            "num_restarts": 12,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
        }
        self.optimize_acqf = "ax.models.torch.botorch_mes.optimize_acqf"

    def test_MaxValueEntropySearch(self):

        model = MaxValueEntropySearch()
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

        # test model.gen()
        new_X_dummy = torch.rand(1, 1, 3, dtype=self.dtype, device=self.device)
        with mock.patch(self.optimize_acqf) as mock_optimize_acqf:
            mock_optimize_acqf.side_effect = [(new_X_dummy, None)]
            Xgen, wgen, _, __ = model.gen(
                n=1,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": self.acq_options,
                    "optimizer_kwargs": self.optimizer_options,
                },
            )
            self.assertTrue(torch.equal(Xgen, new_X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(1, dtype=self.dtype)))
            mock_optimize_acqf.assert_called_once()

        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=None,
            linear_constraints=None,
            model_gen_options={
                "acquisition_function_kwargs": self.acq_options,
                "optimizer_kwargs": self.optimizer_options,
            },
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # test error message in case of constraints
        linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        with self.assertRaises(UnsupportedError):
            Xgen, wgen, _, __ = model.gen(
                n=1,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                linear_constraints=linear_constraints,
            )

        # test error message in case of >1 objective weights
        objective_weights = torch.tensor(
            [1.0, 1.0], dtype=self.dtype, device=self.device
        )
        with self.assertRaises(UnsupportedError):
            Xgen, wgen, _, __ = model.gen(
                n=1, bounds=self.bounds, objective_weights=objective_weights
            )

        # test error message in best_point()
        with self.assertRaises(UnsupportedError):
            Xgen = model.best_point(
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                linear_constraints=linear_constraints,
            )

        with self.assertRaises(RuntimeError):
            Xgen = model.best_point(
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                target_fidelities={2: 1.0},
            )

    def test_MaxValueEntropySearch_MultiFidelity(self):
        model = MaxValueEntropySearch()
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

        # check error when target fidelity and fixed features have the same key
        with self.assertRaises(RuntimeError):
            model.best_point(
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                target_fidelities={2: 1.0},
                fixed_features={2: 1.0},
            )

        # check generation
        n = 1
        new_X_dummy = torch.rand(1, n, 3, dtype=self.dtype, device=self.device)
        with mock.patch(
            self.optimize_acqf, side_effect=[(new_X_dummy, None)]
        ) as mock_optimize_acqf:
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
                target_fidelities={2: 1.0},
            )
            self.assertTrue(torch.equal(Xgen, new_X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=self.dtype)))
            mock_optimize_acqf.assert_called()

    def test_instantiate_MES(self):

        model = MaxValueEntropySearch()
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

        # test acquisition setting
        X_dummy = torch.ones(1, 3, dtype=self.dtype, device=self.device)
        candidate_set = torch.rand(10, 3, dtype=self.dtype, device=self.device)
        acq_function = _instantiate_MES(model=model.model, candidate_set=candidate_set)

        self.assertIsInstance(acq_function, qMaxValueEntropy)
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsInstance(acq_function.fantasies_sampler, SobolQMCNormalSampler)
        self.assertEqual(acq_function.num_fantasies, 16)
        self.assertEqual(acq_function.num_mv_samples, 10)
        self.assertEqual(acq_function.use_gumbel, True)
        self.assertEqual(acq_function.maximize, True)

        acq_function = _instantiate_MES(
            model=model.model, candidate_set=candidate_set, X_pending=X_dummy
        )
        self.assertTrue(torch.equal(acq_function.X_pending, X_dummy))

        # multi-fidelity tests
        model = MaxValueEntropySearch()
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

        candidate_set = torch.rand(10, 3, dtype=self.dtype, device=self.device)
        acq_function = _instantiate_MES(
            model=model.model, candidate_set=candidate_set, target_fidelities={2: 1.0}
        )
        self.assertIsInstance(acq_function, qMultiFidelityMaxValueEntropy)
        self.assertEqual(acq_function.expand(self.Xs), self.Xs)

        # test error that target fidelity and fidelity weight indices must match
        with self.assertRaises(RuntimeError):
            _instantiate_MES(
                model=model.model,
                candidate_set=candidate_set,
                target_fidelities={1: 1.0},
                fidelity_weights={2: 1.0},
            )
