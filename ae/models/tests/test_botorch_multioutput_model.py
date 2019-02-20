#!/usr/bin/env python3

from contextlib import ExitStack
from unittest import mock

import torch
from ae.lazarus.ae.models.torch.botorch_multioutput import BotorchMultiOutputModel
from ae.lazarus.ae.models.torch.utils import MIN_OBSERVED_NOISE_LEVEL
from ae.lazarus.ae.utils.common.testutils import TestCase
from botorch.models import MultiOutputGP
from gpytorch.likelihoods import _GaussianLikelihoodBase

from .utils import _get_torch_test_data


FIT_MODEL_MO_PATH = "ae.lazarus.ae.models.torch.botorch_multioutput.fit_model"
GEN_MO_PATH_PREFIX = "ae.lazarus.ae.models.torch.utils."


class BotorchMultiOutputModelTest(TestCase):
    def test_BotorchMultiOutputModelNoiseless(self, dtype=torch.float, cuda=False):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=True
        )
        Xs2, Ys2, Yvars2, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=True
        )

        model = BotorchMultiOutputModel()
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
        self.assertIsInstance(model.model, MultiOutputGP)

        # Check fitting
        model_list = model.model.models
        self.assertTrue(torch.equal(model_list[0].train_inputs[0], Xs1[0]))
        self.assertTrue(torch.equal(model_list[1].train_inputs[0], Xs2[0]))
        self.assertTrue(torch.equal(model_list[0].train_targets, Ys1[0].view(-1)))
        self.assertTrue(torch.equal(model_list[1].train_targets, Ys2[0].view(-1)))
        self.assertIsInstance(model_list[0].likelihood, _GaussianLikelihoodBase)
        self.assertIsInstance(model_list[1].likelihood, _GaussianLikelihoodBase)

        # Check prediction
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor([[6.0, 7.0, 8.0]], dtype=dtype, device=device)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([1, 2]))
        self.assertTrue(f_cov.shape == torch.Size([1, 2, 2]))

        # Check validation of task features on fit
        model_2 = BotorchMultiOutputModel()
        with self.assertRaises(ValueError):
            model_2.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=[0],
                feature_names=feature_names,
            )

        # Check generation
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        outcome_constraints = (
            torch.tensor([[0.0, 1.0]], dtype=dtype, device=device),
            torch.tensor([[5.0]], dtype=dtype, device=device),
        )
        linear_constraints = None
        fixed_features = None
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device)
        ]
        model_gen_options = {}
        n = 3

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype, device=device)
        val_dummy = torch.tensor([[1.0]], dtype=dtype, device=device)

        with ExitStack() as es:
            es.enter_context(
                mock.patch(
                    GEN_MO_PATH_PREFIX + "gen_candidates_scipy",
                    return_value=(X_dummy, val_dummy),
                )
            )
            es.enter_context(
                mock.patch(
                    GEN_MO_PATH_PREFIX + "_gen_batch_initial_conditions",
                    return_value=X_dummy,
                )
            )
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options=model_gen_options,
            )
        # note: gen() always returns CPU tensors
        self.assertTrue(torch.equal(Xgen, X_dummy.squeeze(0).repeat(n, 1).cpu()))
        self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))

        # Check best point selection
        xbest = model.best_point(
            bounds=bounds,
            objective_weights=torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        )
        xbest = model.best_point(
            bounds=bounds,
            objective_weights=torch.tensor([1.0, 0.0], device=device, dtype=dtype),
            fixed_features={0: 100.0},
        )
        self.assertIsNone(xbest)

        # Test errors for unsupported features
        with self.assertRaises(ValueError):
            model.gen(n, bounds, objective_weights=None)
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
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

    def test_BotorchMultiOutputModelNoiseless_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMultiOutputModelNoiseless(cuda=True)

    def test_BotorchMultiOutputModelNoiseless_double(self):
        self.test_BotorchMultiOutputModelNoiseless(dtype=torch.double)

    def test_BotorchMultiOutputModelNoiseless_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMultiOutputModelNoiseless(dtype=torch.double, cuda=True)

    def test_BotorchMultiOutputModel(self, dtype=torch.float, cuda=False):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=False
        )
        Xs2, Ys2, Yvars2, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, noiseless=False
        )
        model = BotorchMultiOutputModel()
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
        self.assertIsInstance(model.model, MultiOutputGP)

        # Check fitting
        model_list = model.model.models
        self.assertTrue(torch.equal(model_list[0].train_inputs[0], Xs1[0]))
        self.assertTrue(torch.equal(model_list[1].train_inputs[0], Xs2[0]))
        self.assertTrue(torch.equal(model_list[0].train_targets, Ys1[0].view(-1)))
        self.assertTrue(torch.equal(model_list[1].train_targets, Ys2[0].view(-1)))
        for mdl, Yvar in zip(model_list, Yvars1 + Yvars2):
            self.assertLess(
                torch.norm(
                    mdl.likelihood.noise_covar.noise_model.train_targets
                    - (Yvar.view(-1).clamp_min(MIN_OBSERVED_NOISE_LEVEL).log())
                ).item(),
                1e-5,
            )
        self.assertIsInstance(model_list[0].likelihood, _GaussianLikelihoodBase)
        self.assertIsInstance(model_list[1].likelihood, _GaussianLikelihoodBase)

        # Check prediction
        device = torch.device("cuda") if cuda else torch.device("cpu")
        X = torch.tensor([[6.0, 7.0, 8.0]], dtype=dtype, device=device)
        f_mean, f_cov = model.predict(X)
        self.assertTrue(f_mean.shape == torch.Size([1, 2]))
        self.assertTrue(f_cov.shape == torch.Size([1, 2, 2]))

        # Check validation on fit
        model_2 = BotorchMultiOutputModel()
        with self.assertRaises(ValueError):
            model_2.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=[0],
                feature_names=feature_names,
            )

        # Check generation
        objective_weights = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        outcome_constraints = (
            torch.tensor([[0.0, 1.0]], dtype=dtype, device=device),
            torch.tensor([[5.0]], dtype=dtype, device=device),
        )
        linear_constraints = None
        fixed_features = None
        pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device)
        ]
        n = 3

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype, device=device)
        val_dummy = torch.tensor([[1.0]], dtype=dtype, device=device)

        with ExitStack() as es:
            es.enter_context(
                mock.patch(
                    GEN_MO_PATH_PREFIX + "gen_candidates_scipy",
                    return_value=(X_dummy, val_dummy),
                )
            )
            es.enter_context(
                mock.patch(
                    GEN_MO_PATH_PREFIX + "_gen_batch_initial_conditions",
                    return_value=X_dummy,
                )
            )
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
            )
            # note: gen() always returns CPU tensors
            print(f"Xgen: {Xgen}")
            print(f"X_dummy: {X_dummy}")
            self.assertTrue(torch.equal(Xgen, X_dummy.squeeze(0).repeat(n, 1).cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))

        # Test errors for unsupported features
        objective_weights = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
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

        # validate that generating errors out correctly if not a block design
        model.Xs[0] += 1
        with self.assertRaises(NotImplementedError):
            model.gen(n, bounds, objective_weights)

    def test_BotorchMultiOutputModel_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMultiOutputModel(cuda=True)

    def test_BotorchMultiOutputModel_double(self):
        self.test_BotorchMultiOutputModel(dtype=torch.double)

    def test_BotorchMultiOutputModel_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMultiOutputModel(dtype=torch.double, cuda=True)

    # TODO: Implement getModel tests for MultiOutputModel
