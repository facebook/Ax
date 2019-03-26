#!/usr/bin/env python3

from itertools import chain
from unittest import mock

import torch
from ax.models.torch.botorch import BotorchModel, _get_and_fit_model
from ax.models.torch.utils import MIN_OBSERVED_NOISE_LEVEL
from ax.utils.common.testutils import TestCase
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
from botorch.models import MultiOutputGP
from botorch.utils import get_objective_weights_transform
from gpytorch.likelihoods import _GaussianLikelihoodBase


FIT_MODEL_MO_PATH = "ax.models.torch.botorch.fit_model"
GEN_MO_PATH_PREFIX = "ax.models.torch.utils."


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

        # Check validation on fit
        model_2 = BotorchModel()
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
            torch.tensor([[1.0, 3.0, 4.0]], dtype=dtype, device=device),
            torch.tensor([[2.0, 6.0, 8.0]], dtype=dtype, device=device),
        ]
        n = 3

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=dtype, device=device)
        model_gen_options = {}
        # test sequential optimize
        with mock.patch(
            "ax.models.torch.botorch.sequential_optimize", return_value=X_dummy
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
            "ax.models.torch.botorch.joint_optimize", return_value=X_dummy
        ) as mock_joint_optimize:
            Xgen, wgen = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=None,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"joint_optimization": True},
            )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
            self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))
            mock_joint_optimize.assert_called_once()

        # test passing acquisition function via model_gen_options
        acquisition_function = get_acquisition_function(
            acquisition_function_name="qUCB",
            model=model.model,
            objective=LinearMCObjective(objective_weights),
            X_observed=Xs1 + Xs2,
            beta=2.0,
        )
        with mock.patch(
            "ax.models.torch.botorch.sequential_optimize", return_value=X_dummy
        ) as mock_sequential_optimize:
            _, __ = model.gen(
                n=n,
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"acquisition_function": acquisition_function},
            )
            self.assertEqual(
                mock_sequential_optimize.call_args_list[-1][1]["acq_function"],
                acquisition_function,
            )

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

        # Test loading state dict
        tkwargs = {"device": device, "dtype": dtype}
        state_dict_keys = [
            "models.0.likelihood.noise_covar._noise_levels",
            "models.0.mean_module.constant",
            "models.0.covar_module.raw_outputscale",
            "models.0.covar_module.base_kernel.raw_lengthscale",
            "models.0.covar_module.base_kernel.lengthscale_prior.concentration",
            "models.0.covar_module.base_kernel.lengthscale_prior.rate",
            "models.0.covar_module.outputscale_prior.concentration",
            "models.0.covar_module.outputscale_prior.rate",
            "likelihood.likelihoods.0.noise_covar._noise_levels",
        ]
        state_dict_vals = [
            [0.0003],
            [[3.4969]],
            [1.0109],
            [[[-0.9313, -0.9313, -0.9313]]],
            3.0,
            6.0,
            2.0,
            0.1500,
            [0.0003],
        ]
        state_dict_vals = [torch.tensor(val, **tkwargs) for val in state_dict_vals]
        true_state_dict = dict(zip(state_dict_keys, state_dict_vals))
        model = _get_and_fit_model(
            Xs=Xs1, Ys=Ys1, Yvars=Yvars1, state_dict=true_state_dict
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

    def test_BotorchModelHetNoise(self, dtype=torch.float, cuda=False):
        Xs1, Ys1, Yvars1, bounds, task_features, feature_names = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=False
        )
        Xs2, Ys2, Yvars2, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=False
        )

        model = BotorchModel()
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

        self.assertLess(
            torch.norm(
                model.model.likelihood.likelihoods[
                    0
                ].noise_covar.noise_model.train_targets
                - (Yvars1[0].view(-1).clamp_min(MIN_OBSERVED_NOISE_LEVEL).log())
            ).item(),
            1e-5,
        )

        # Test state dict loading
        nm = "models.0.likelihood.noise_covar.noise_model"
        lk = "likelihood.likelihoods.0.noise_covar.noise_model"
        state_dict_keys = [
            f"{nm}.likelihood.noise_covar.raw_noise",
            f"{nm}.likelihood.noise_covar.noise_prior.a",
            f"{nm}.likelihood.noise_covar.noise_prior.b",
            f"{nm}.likelihood.noise_covar.noise_prior.sigma",
            f"{nm}.likelihood.noise_covar.noise_prior.tails.loc",
            f"{nm}.likelihood.noise_covar.noise_prior.tails.scale",
            f"{nm}.mean_module.constant",
            f"{nm}.covar_module.raw_outputscale",
            f"{nm}.covar_module.base_kernel.raw_lengthscale",
            f"{nm}.covar_module.base_kernel.lengthscale_prior.concentration",
            f"{nm}.covar_module.base_kernel.lengthscale_prior.rate",
            f"{nm}.covar_module.outputscale_prior.concentration",
            f"{nm}.covar_module.outputscale_prior.rate",
            "models.0.mean_module.constant",
            "models.0.covar_module.raw_outputscale",
            "models.0.covar_module.base_kernel.raw_lengthscale",
            "models.0.covar_module.base_kernel.lengthscale_prior.concentration",
            "models.0.covar_module.base_kernel.lengthscale_prior.rate",
            "models.0.covar_module.outputscale_prior.concentration",
            "models.0.covar_module.outputscale_prior.rate",
            f"{lk}.likelihood.noise_covar.raw_noise",
            f"{lk}.likelihood.noise_covar.noise_prior.a",
            f"{lk}.likelihood.noise_covar.noise_prior.b",
            f"{lk}.likelihood.noise_covar.noise_prior.sigma",
            f"{lk}.likelihood.noise_covar.noise_prior.tails.loc",
            f"{lk}.likelihood.noise_covar.noise_prior.tails.scale",
            f"{lk}.mean_module.constant",
            f"{lk}.covar_module.raw_outputscale",
            f"{lk}.covar_module.base_kernel.raw_lengthscale",
            f"{lk}.covar_module.base_kernel.lengthscale_prior.concentration",
            f"{lk}.covar_module.base_kernel.lengthscale_prior.rate",
            f"{lk}.covar_module.outputscale_prior.concentration",
            f"{lk}.covar_module.outputscale_prior.rate",
        ]
        state_dict_vals = [
            [[0.0]],
            [-3.0],
            [5.0],
            [0.5],
            [0.0],
            [0.5],
            [[-7.654_725_551_605_225]],
            [6.679_621_219_635_01],
            [
                [
                    [
                        -0.927_428_603_172_302_2,
                        -0.927_428_603_172_302_2,
                        -0.927_428_603_172_302_2,
                    ]
                ]
            ],
            3.0,
            6.0,
            2.0,
            0.150_000_005_960_464_48,
            [[3.500_487_565_994_262_7]],
            [0.955_063_641_071_319_6],
            [
                [
                    [
                        -0.929_637_432_098_388_7,
                        -0.929_637_432_098_388_7,
                        -0.929_637_432_098_388_7,
                    ]
                ]
            ],
            3.0,
            6.0,
            2.0,
            0.150_000_005_960_464_48,
            [[0.0]],
            [-3.0],
            [5.0],
            [0.5],
            [0.0],
            [0.5],
            [[-7.654_725_551_605_225]],
            [6.679_621_219_635_01],
            [
                [
                    [
                        -0.927_428_603_172_302_2,
                        -0.927_428_603_172_302_2,
                        -0.927_428_603_172_302_2,
                    ]
                ]
            ],
            3.0,
            6.0,
            2.0,
            0.150_000_005_960_464_48,
        ]
        device = torch.device("cuda") if cuda else torch.device("cpu")
        tkwargs = {"device": device, "dtype": dtype}
        state_dict_vals = [torch.tensor(val, **tkwargs) for val in state_dict_vals]
        true_state_dict = dict(zip(state_dict_keys, state_dict_vals))
        model = _get_and_fit_model(
            Xs=Xs1, Ys=Ys1, Yvars=Yvars1, state_dict=true_state_dict
        )
        for k, v in chain(model.named_parameters(), model.named_buffers()):
            self.assertTrue(torch.equal(true_state_dict[k], v))

    def test_BotorchModelHetNoise_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModelHetNoise(cuda=True)

    def test_BotorchModelConstantNoise_double(self):
        self.test_BotorchModelHetNoise(dtype=torch.double)

    def test_BotorchModelConstantNoise_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchModelHetNoise(dtype=torch.double, cuda=True)

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
