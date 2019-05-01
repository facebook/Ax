#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from itertools import chain
from unittest import mock

import torch
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_defaults import get_and_fit_model
from ax.utils.common.testutils import TestCase
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.utils import get_objective_weights_transform
from gpytorch.likelihoods import _GaussianLikelihoodBase


FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_model"


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


def _get_torch_test_data(dtype=torch.float, cuda=False, constant_noise=True):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=dtype, device=device)]
    Ys = [torch.tensor([[3.0], [4.0]], dtype=dtype, device=device)]
    Yvars = [torch.tensor([[0.0], [2.0]], dtype=dtype, device=device)]
    if constant_noise:
        Yvars[0].fill_(1.0)
    bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
    task_features = []
    feature_names = ["x1", "x2", "x3"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names


class BotorchModelTest(TestCase):
    def test_BotorchModel(self, dtype=torch.float, cuda=False):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        model = BotorchModel()
        # Test ModelListGP
        # make training data different for each output
        Xs2_diff = [Xs2[0] + 0.1]
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2_diff,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()
        # Check attributes
        self.assertTrue(torch.equal(model.Xs[0], Xs1[0]))
        self.assertTrue(torch.equal(model.Xs[1], Xs2_diff[0]))
        self.assertEqual(model.dtype, Xs1[0].dtype)
        self.assertEqual(model.device, Xs1[0].device)
        self.assertIsInstance(model.model, ModelListGP)

        # Check fitting
        model_list = model.model.models
        self.assertTrue(torch.equal(model_list[0].train_inputs[0], Xs1[0]))
        self.assertTrue(torch.equal(model_list[1].train_inputs[0], Xs2_diff[0]))
        self.assertTrue(torch.equal(model_list[0].train_targets, Ys1[0].view(-1)))
        self.assertTrue(torch.equal(model_list[1].train_targets, Ys2[0].view(-1)))
        self.assertIsInstance(model_list[0].likelihood, _GaussianLikelihoodBase)
        self.assertIsInstance(model_list[1].likelihood, _GaussianLikelihoodBase)

        # Test batched multi-output FixedNoiseGP
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()

        # Check attributes
        self.assertTrue(torch.equal(model.Xs[0], Xs1[0]))
        self.assertTrue(torch.equal(model.Xs[1], Xs2[0]))
        self.assertEqual(model.dtype, Xs1[0].dtype)
        self.assertEqual(model.device, Xs1[0].device)
        self.assertIsInstance(model.model, FixedNoiseGP)

        # Check fitting
        # train inputs should be `o x n x 1`
        self.assertTrue(
            torch.equal(
                model.model.train_inputs[0],
                Xs1[0].unsqueeze(0).expand(torch.Size([2]) + Xs1[0].shape),
            )
        )
        # train targets should be `o x n`
        self.assertTrue(
            torch.equal(
                model.model.train_targets, torch.cat(Ys1 + Ys2, dim=-1).permute(1, 0)
            )
        )
        self.assertIsInstance(model.model.likelihood, _GaussianLikelihoodBase)

        # Check infeasible cost can be computed on the model
        device = torch.device("cuda") if cuda else torch.device("cpu")
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        objective_transform = get_objective_weights_transform(objective_weights)
        infeasible_cost = torch.tensor(
            get_infeasible_cost(
                X=Xs1[0], model=model.model, objective=objective_transform
            )
        )
        expected_infeasible_cost = -1 * torch.min(
            objective_transform(
                model.model.posterior(Xs1[0]).mean
                - 6 * model.model.posterior(Xs1[0]).variance.sqrt()
            ).min(),
            torch.tensor(0.0, dtype=dtype, device=device),
        )
        self.assertTrue(torch.abs(infeasible_cost - expected_infeasible_cost) < 1e-5)

        # Check prediction
        X = torch.tensor([[6.0, 7.0, 8.0]], dtype=dtype, device=device)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([1, 2]))
        self.assertTrue(f_cov.shape == torch.Size([1, 2, 2]))

        # Check generation
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        outcome_constraints = (
            torch.tensor([[0.0, 1.0]], dtype=dtype, device=device),
            torch.tensor([[5.0]], dtype=dtype, device=device),
        )
        linear_constraints = (torch.tensor([[0.0, 1.0, 1.0]]), torch.tensor([[100.0]]))
        fixed_features = None
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device),
            torch.tensor([[2.0, 6.0, 8.0]], dtype=dtype, device=device),
        ]
        n = 3

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype, device=device)
        model_gen_options = {}
        # test sequential optimize
        with mock.patch(
            "ax.models.torch.botorch_defaults.sequential_optimize", return_value=X_dummy
        ) as mock_sequential_optimize:

            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options=model_gen_options,
                rounding_func=dummy_func,
            )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            self.assertEqual(
                mock_sequential_optimize.call_args_list[-1][1]["post_processing_func"],
                dummy_func,
            )

        # test joint optimize
        with mock.patch(
            "ax.models.torch.botorch_defaults.joint_optimize", return_value=X_dummy
        ) as mock_joint_optimize:
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=None,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"optimizer_kwargs": {"joint_optimization": True}},
            )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            mock_joint_optimize.assert_called_once()

        # Check best point selection
        xbest = model.best_point(bounds=bounds, objective_weights=objective_weights)
        xbest = model.best_point(
            bounds=bounds,
            objective_weights=objective_weights,
            fixed_features={0: 100.0},
        )
        self.assertIsNone(xbest)

        # Test cross-validation
        mean, variance = model.cross_validate(
            Xs_train=Xs1 + Xs2,
            Ys_train=Ys1 + Ys2,
            Yvars_train=Yvars1 + Yvars2,
            X_test=torch.tensor(
                [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], dtype=dtype, device=device
            ),
        )
        self.assertTrue(mean.shape == torch.Size([2, 2]))
        self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

        # Test cross-validation with refit_on_cv
        model.refit_on_cv = True
        mean, variance = model.cross_validate(
            Xs_train=Xs1 + Xs2,
            Ys_train=Ys1 + Ys2,
            Yvars_train=Yvars1 + Yvars2,
            X_test=torch.tensor(
                [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], dtype=dtype, device=device
            ),
        )
        self.assertTrue(mean.shape == torch.Size([2, 2]))
        self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

        # Test update
        model.refit_on_update = False
        model.update(Xs=Xs2 + Xs2, Ys=Ys2 + Ys2, Yvars=Yvars2 + Yvars2)
        self.assertTrue(torch.equal(model.Xs[0], torch.cat((Xs1[0], Xs2[0]))))
        self.assertTrue(torch.equal(model.Xs[1], torch.cat((Xs2[0], Xs2[0]))))
        self.assertTrue(torch.equal(model.Ys[0], torch.cat((Ys1[0], Ys2[0]))))
        self.assertTrue(torch.equal(model.Yvars[0], torch.cat((Yvars1[0], Yvars2[0]))))

        model.refit_on_update = True
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.update(Xs=Xs2 + Xs2, Ys=Ys2 + Ys2, Yvars=Yvars2 + Yvars2)

        # test unfit model CV and update
        unfit_model = BotorchModel()
        with self.assertRaises(RuntimeError):
            unfit_model.cross_validate(
                Xs_train=Xs1 + Xs2,
                Ys_train=Ys1 + Ys2,
                Yvars_train=Yvars1 + Yvars2,
                X_test=Xs1[0],
            )
        with self.assertRaises(RuntimeError):
            unfit_model.update(Xs=Xs1 + Xs2, Ys=Ys1 + Ys2, Yvars=Yvars1 + Yvars2)

        # Test loading state dict
        tkwargs = {"device": device, "dtype": dtype}
        true_state_dict = {
            "mean_module.constant": [[3.5004]],
            "covar_module.raw_outputscale": [2.2438],
            "covar_module.base_kernel.raw_lengthscale": [[[-0.9274, -0.9274, -0.9274]]],
            "covar_module.base_kernel.lengthscale_prior.concentration": 3.0,
            "covar_module.base_kernel.lengthscale_prior.rate": 6.0,
            "covar_module.outputscale_prior.concentration": 2.0,
            "covar_module.outputscale_prior.rate": 0.15,
        }
        true_state_dict = {
            key: torch.tensor(val, **tkwargs) for key, val in true_state_dict.items()
        }
        model = get_and_fit_model(
            Xs=Xs1, Ys=Ys1, Yvars=Yvars1, task_features=[], state_dict=true_state_dict
        )
        for k, v in chain(model.named_parameters(), model.named_buffers()):
            self.assertTrue(torch.equal(true_state_dict[k], v))

    def test_BotorchModel_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModel(cuda=True)

    def test_BotorchModel_double(self):
        self.test_BotorchModel(dtype=torch.double)

    def test_BotorchModel_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModel(dtype=torch.double, cuda=True)

    def test_BotorchModelOneOutcome(self):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        model = BotorchModel()
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1,
                Ys=Ys1,
                Yvars=Yvars1,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()
        X = torch.rand(2, 3, dtype=torch.float)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([2, 1]))
        self.assertTrue(f_cov.shape == torch.Size([2, 1, 1]))

    def test_BotorchModelConstraints(self):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _ = _get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        # make infeasible
        Xs2[0] = -1 * Xs2[0]
        objective_weights = torch.tensor(
            [-1.0, 1.0], dtype=torch.float, device=torch.device("cpu")
        )
        n = 3
        model = BotorchModel()
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()

        # because there are no feasible points:
        with self.assertRaises(ValueError):
            model.gen(n, bounds, objective_weights)
