#!/usr/bin/env python3

import mock
import torch
from ax.models.torch.botorch_kg import (
    KnowledgeGradient,
    _get_objective,
    _instantiate_KG,
    _set_best_point_acq,
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
    def test_KnowledgeGradient(self):
        device = torch.device("cpu")
        dtype = torch.double

        Xs = [
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=dtype, device=device)
        ]
        Ys = [torch.tensor([[3.0], [4.0]], dtype=dtype, device=device)]
        Yvars = [torch.tensor([[0.0], [2.0]], dtype=dtype, device=device)]
        bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        task_features = []
        feature_names = ["x1", "x2", "x3"]

        acq_options = {"num_fantasies": 30, "mc_samples": 30}

        optimizer_options = {
            "num_restarts": 12,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
        }

        model = KnowledgeGradient()
        objective_weights = torch.tensor([1.0], dtype=dtype, device=device)
        model.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            fidelity_features=[],
        )

        n = 2

        X_dummy = torch.rand(1, n, 4, dtype=dtype, device=device)
        acq_dummy = torch.tensor(0.0, dtype=dtype, device=device)

        with mock.patch(
            "ax.models.torch.botorch_kg.optimize_acqf",
            return_value=(X_dummy, acq_dummy),
        ) as mock_optimize_acqf:
            Xgen, wgen, _ = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": acq_options,
                    "optimizer_kwargs": optimizer_options,
                },
            )
            self.assertTrue(torch.equal(Xgen, X_dummy[:, :2, :].cpu()))
            self.assertEqual(Xgen.size(), torch.Size([1, 2, 4]))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            mock_optimize_acqf.assert_called()  # called twice, once for best_point

        ini_dummy = torch.rand(10, 32, 3, dtype=dtype, device=device)
        optimizer_options2 = {
            "num_restarts": 10,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
            "partial_restarts": 2,
        }
        with mock.patch(
            "ax.models.torch.botorch_kg.gen_one_shot_kg_initial_conditions",
            return_value=ini_dummy,
        ) as mock_warmstart_initialization:
            Xgen, wgen, _ = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": acq_options,
                    "optimizer_kwargs": optimizer_options2,
                },
            )
            mock_warmstart_initialization.assert_called_once()

        obj = ScalarizedObjective(weights=objective_weights)
        dummy_acq = PosteriorMean(model=model.model, objective=obj)
        with mock.patch(
            "ax.models.torch.botorch_kg.PosteriorMean", return_value=dummy_acq
        ) as mock_posterior_mean:
            Xgen, wgen, _ = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                model_gen_options={
                    "acquisition_function_kwargs": acq_options,
                    "optimizer_kwargs": optimizer_options2,
                },
            )
            self.assertEqual(mock_posterior_mean.call_count, 2)

        X_dummy = torch.rand(3)
        acq_dummy = torch.tensor(0.0)
        # Check best point selection
        with mock.patch(
            "ax.models.torch.botorch_kg.optimize_acqf",
            return_value=(X_dummy, acq_dummy),
        ) as mock_optimize_acqf:
            xbest = model.best_point(bounds=bounds, objective_weights=objective_weights)
            self.assertTrue(torch.equal(xbest, X_dummy))
            mock_optimize_acqf.assert_called_once()

        X_dummy = torch.tensor([1.0, 2.0])
        acq_dummy = torch.tensor(0.0)
        model.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            fidelity_features=[-1],
        )
        # Check best point selection
        with mock.patch(
            "ax.models.torch.botorch_kg.optimize_acqf",
            return_value=(X_dummy, acq_dummy),
        ) as mock_optimize_acqf:
            xbest = model.best_point(bounds=bounds, objective_weights=objective_weights)
            self.assertTrue(torch.equal(xbest, torch.tensor([1.0, 2.0, 1.0])))
            mock_optimize_acqf.assert_called_once()

        X_dummy = torch.zeros(12, 1, 2, dtype=dtype, device=device)
        X_dummy2 = torch.zeros(1, 32, 3, dtype=dtype, device=device)
        acq_dummy = torch.tensor(0.0, dtype=dtype, device=device)
        dummy1 = (X_dummy, acq_dummy)
        dummy2 = (X_dummy2, acq_dummy)
        with mock.patch(
            "ax.models.torch.botorch_kg.optimize_acqf",
            side_effect=[dummy1, dummy2, dummy1, dummy2],
        ) as mock_optimize_acqf:
            mock_optimize_acqf(
                Xgen,
                wgen=model.gen(
                    n=n,
                    bounds=bounds,
                    objective_weights=objective_weights,
                    outcome_constraints=None,
                    linear_constraints=None,
                    model_gen_options={
                        "acquisition_function_kwargs": acq_options,
                        "optimizer_kwargs": optimizer_options,
                    },
                ),
            )

        # test error message
        linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        with self.assertRaises(UnsupportedError):
            Xgen, wgen, _ = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=linear_constraints,
            )

        with self.assertRaises(UnsupportedError):
            xbest = model.best_point(
                bounds=bounds,
                linear_constraints=linear_constraints,
                objective_weights=objective_weights,
            )

        # test _instantiate_KG
        objective = ScalarizedObjective(weights=objective_weights)
        X_dummy = torch.ones(1, 3, dtype=dtype, device=device)
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

        acq_function = _instantiate_KG(
            model=model.model,
            objective=objective,
            fidelity_features=[-1],
            current_value=0,
        )
        self.assertIsInstance(acq_function, qMultiFidelityKnowledgeGradient)

        acq_function = _instantiate_KG(
            model=model.model, objective=LinearMCObjective(weights=objective_weights)
        )
        self.assertIsInstance(acq_function.inner_sampler, SobolQMCNormalSampler)

        # test _get_obj()
        outcome_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        objective_weights = None
        self.assertIsInstance(
            _get_objective(
                model=model.model,
                outcome_constraints=outcome_constraints,
                objective_weights=objective_weights,
                X_observed=X_dummy,
            ),
            ConstrainedMCObjective,
        )
        # test _set_best_point_acq
        acq_function = _set_best_point_acq(
            model=model.model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
        )
        self.assertIsInstance(acq_function, qSimpleRegret)
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)

        acq_function = _set_best_point_acq(
            model=model.model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_dummy,
            qmc=False,
        )
        self.assertIsInstance(acq_function.sampler, IIDNormalSampler)

        objective_weights = torch.tensor([1.0], dtype=dtype, device=device)
        acq_function = _set_best_point_acq(
            model=model.model,
            objective_weights=objective_weights,
            X_observed=X_dummy,
            fidelity_features=[-1],
        )
        self.assertIsInstance(acq_function, FixedFeatureAcquisitionFunction)
