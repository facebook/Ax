#!/usr/bin/env python3

from itertools import chain
from unittest import mock

import torch
from ae.lazarus.ae.models.torch.botorch import BotorchModel, _get_and_fit_model
from ae.lazarus.ae.models.torch.utils import MIN_OBSERVED_NOISE_LEVEL
from ae.lazarus.ae.utils.common.testutils import TestCase
from gpytorch.likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
from torch import Tensor

from .utils import (
    _get_model_test_state_dict,
    _get_model_test_state_dict_noiseless,
    _get_torch_test_data,
)


def dummy_func(X: Tensor) -> Tensor:
    return X


class BotorchModelTest(TestCase):
    def test_BotorchModelNoiseless(self, dtype=torch.float, cuda=False):
        Xs, Ys, Yvars, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=True
        )

        model = BotorchModel()
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch.fit_model"
        ) as _mock_fit_model:
            model.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()

        # Check attributes
        self.assertTrue(torch.equal(model.Xs[0], Xs[0]))
        self.assertEqual(model.dtype, Xs[0].dtype)
        self.assertEqual(model.device, Xs[0].device)
        self.assertEqual(len(model.models), 1)

        # Check fitting
        self.assertTrue(torch.equal(model.models[0].train_inputs[0], Xs[0]))
        self.assertTrue(torch.equal(model.models[0].train_targets, Ys[0].view(-1)))
        self.assertIsInstance(model.models[0].likelihood, _GaussianLikelihoodBase)

        # Check prediction
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor([[6.0, 7.0, 8.0]], dtype=dtype, device=device)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([1, 1]))
        self.assertTrue(f_cov.shape == torch.Size([1, 1, 1]))

        # Check validation on fit
        model_2 = BotorchModel()
        with self.assertRaises(ValueError):
            model_2.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=bounds,
                task_features=[0],
                feature_names=feature_names,
            )

        # Check generation
        # objective_weights has length 2 here to check that we correctly select
        # the non-zero objective weight when objective_weights contain one
        # non-zero weight.
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        outcome_constraints = None
        linear_constraints = None
        fixed_features = None
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device)
        ]
        model_gen_options = {}
        n = 3

        X_dummy = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype, device=device).repeat(
            n, 1
        )

        # test sequential optimize
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch._sequential_optimize",
            return_value=X_dummy,
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
                mock_sequential_optimize.call_args_list[-1][1]["rounding_func"],
                dummy_func,
            )

        # test joint optimize
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch._joint_optimize", return_value=X_dummy
        ) as mock_joint_optimize:
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"joint_optimization": True},
            )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            mock_joint_optimize.assert_called_once()

        # Check best point selection
        xbest = model.best_point(
            bounds=bounds,
            objective_weights=torch.tensor([1.0], device=device, dtype=dtype),
        )
        xbest = model.best_point(
            bounds=bounds,
            objective_weights=torch.tensor([1.0], device=device, dtype=dtype),
            fixed_features={0: 100.0},
        )
        self.assertIsNone(xbest)

        objective_weights = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
        with self.assertRaises(ValueError):
            model.gen(n, bounds, objective_weights)
        objective_weights = torch.tensor([1.0], dtype=dtype, device=device)
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device),
            torch.tensor([[2.0, 3.0, 4.0]], dtype=dtype, device=device),
        ]
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, pending_observations=pending_observations
            )
        linear_constraints = (
            torch.tensor([[0.0, 1.0, -1.0]], dtype=dtype, device=device),
            torch.tensor([[2.0]], dtype=dtype, device=device),
        )
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, linear_constraints=linear_constraints
            )
        outcome_constraints = (
            torch.tensor([[1.0]], dtype=dtype, device=device),
            torch.tensor([[2.0]], dtype=dtype, device=device),
        )
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, outcome_constraints=outcome_constraints
            )

        # Test cross-validation
        mean, variance = model.cross_validate(
            Xs_train=Xs,
            Ys_train=Ys,
            Yvars_train=Yvars,
            X_test=torch.tensor(
                [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], dtype=dtype, device=device
            ),
        )
        self.assertTrue(mean.shape == torch.Size([2, 1]))
        self.assertTrue(variance.shape == torch.Size([2, 1, 1]))

    def test_BotorchModelNoiseless_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModelNoiseless(cuda=True)

    def test_BotorchModelNoiseless_double(self):
        self.test_BotorchModelNoiseless(dtype=torch.double)

    def test_BotorchModelNoiseless_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModelNoiseless(dtype=torch.double, cuda=True)

    def test_BotorchModel(self, dtype=torch.float, cuda=False):
        Xs, Ys, Yvars, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=False
        )
        model = BotorchModel()
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch.fit_model"
        ) as _mock_fit_model:
            model.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=bounds,
                task_features=task_features,
                feature_names=feature_names,
            )
            _mock_fit_model.assert_called_once()

        # Check attributes
        self.assertTrue(torch.equal(model.Xs[0], Xs[0]))
        self.assertEqual(model.dtype, Xs[0].dtype)
        self.assertEqual(model.device, Xs[0].device)
        self.assertEqual(len(model.models), 1)

        # Check Fitting
        self.assertTrue(torch.equal(model.models[0].train_inputs[0], Xs[0]))
        self.assertTrue(torch.equal(model.models[0].train_targets, Ys[0].view(-1)))
        self.assertLess(
            torch.norm(
                model.models[0].likelihood.noise_covar.noise_model.train_targets
                - (Yvars[0].view(-1).clamp_min(MIN_OBSERVED_NOISE_LEVEL).log())
            ).item(),
            1e-5,
        )
        self.assertIsInstance(model.models[0].likelihood, _GaussianLikelihoodBase)

        # Check prediction
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor([[6.0, 7.0, 8.0]], dtype=dtype, device=device)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([1, 1]))
        self.assertTrue(f_cov.shape == torch.Size([1, 1, 1]))

        # Check validation on fit
        model_2 = BotorchModel()
        with self.assertRaises(ValueError):
            model_2.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=bounds,
                task_features=[0],
                feature_names=feature_names,
            )

        # Check generation
        objective_weights = torch.tensor([1.0], dtype=dtype, device=device)
        outcome_constraints = None
        linear_constraints = None
        fixed_features = None
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device)
        ]
        model_gen_options = {"eta_sim": 0.0}  # simple initialization heuristic
        n = 3

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype, device=device)

        # test sequential optimize
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch._sequential_optimize",
            return_value=X_dummy,
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
                mock_sequential_optimize.call_args_list[-1][1]["rounding_func"],
                dummy_func,
            )

        # test joint optimize
        with mock.patch(
            "ae.lazarus.ae.models.torch.botorch._joint_optimize", return_value=X_dummy
        ) as mock_joint_optimize:
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"joint_optimization": True},
            )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            mock_joint_optimize.assert_called_once()

        # Test errors for unsupported features
        objective_weights = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
        with self.assertRaises(ValueError):
            model.gen(n, bounds, objective_weights)
        objective_weights = torch.tensor([1.0], dtype=dtype, device=device)
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device),
            torch.tensor([[2.0, 3.0, 4.0]], dtype=dtype, device=device),
        ]
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, pending_observations=pending_observations
            )
        linear_constraints = (
            torch.tensor([[0.0, 1.0, -1.0]], dtype=dtype, device=device),
            torch.tensor([[2.0]], dtype=dtype, device=device),
        )
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, linear_constraints=linear_constraints
            )
        outcome_constraints = (
            torch.tensor([[1.0]], dtype=dtype, device=device),
            torch.tensor([[2.0]], dtype=dtype, device=device),
        )
        with self.assertRaises(ValueError):
            model.gen(
                n, bounds, objective_weights, outcome_constraints=outcome_constraints
            )

        # Test cross-validation
        mean, variance = model.cross_validate(
            Xs_train=Xs,
            Ys_train=Ys,
            Yvars_train=Yvars,
            X_test=torch.tensor(
                [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], dtype=dtype, device=device
            ),
        )
        self.assertTrue(mean.shape == torch.Size([2, 1]))
        self.assertTrue(variance.shape == torch.Size([2, 1, 1]))

    def test_BotorchModel_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModel(cuda=True)

    def test_BotorchModel_double(self):
        self.test_BotorchModel(dtype=torch.double)

    def test_BotorchModel_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModel(dtype=torch.double, cuda=True)

    def test_GetModelNoiseless(self, dtype=torch.float, cuda=False):
        # Test getting model with fitted parameters
        state_dict = _get_model_test_state_dict_noiseless(dtype=dtype, cuda=cuda)
        Xs, Ys, Yvars, bounds, task_feature, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=True
        )
        model = _get_and_fit_model(
            X=Xs[0], Y=Ys[0], Yvar=Yvars[0], state_dict=state_dict
        )
        for k, v in chain(model.named_parameters(), model.named_buffers()):
            self.assertTrue(torch.equal(state_dict[k], v))

    def test_GetModelNoiseless_cuda(self):
        if torch.cuda.is_available():
            self.test_GetModelNoiseless(cuda=True)

    def test_GetModelNoiseless_double(self):
        self.test_GetModelNoiseless(dtype=torch.double)

    def test_GetModelNoiseless_double_cuda(self):
        if torch.cuda.is_available():
            self.test_GetModelNoiseless(dtype=torch.double, cuda=True)

    def test_GetModel(self, dtype=torch.float, cuda=False):
        # Test getting model with fitted parameters
        state_dict = _get_model_test_state_dict(dtype=dtype, cuda=cuda)
        Xs, Ys, Yvars, _, _, _ = _get_torch_test_data(dtype=dtype, cuda=cuda)
        model = _get_and_fit_model(
            X=Xs[0], Y=Ys[0], Yvar=Yvars[0], state_dict=state_dict
        )
        for k, v in chain(model.named_parameters(), model.named_buffers()):
            self.assertTrue(torch.equal(state_dict[k], v))

    def test_GetModel_cuda(self):
        if torch.cuda.is_available():
            self.test_GetModel(cuda=True)

    def test_GetModel_double(self):
        self.test_GetModel(dtype=torch.double)

    def test_GetModel_double_cuda(self):
        if torch.cuda.is_available():
            self.test_GetModel(dtype=torch.double, cuda=True)
