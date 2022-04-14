#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from abc import ABC
from contextlib import ExitStack
from itertools import product
from math import sqrt
from typing import Dict, Type
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError
from ax.models.torch.fully_bayesian import (
    FullyBayesianBotorchModel,
    FullyBayesianMOOBotorchModel,
    matern_kernel,
    rbf_kernel,
    single_task_pyro_model,
)
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models import ModelListGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.transforms.input import Warp
from botorch.optim.optimize import optimize_acqf
from botorch.utils import get_objective_weights_transform
from gpytorch.likelihoods import _GaussianLikelihoodBase

RUN_INFERENCE_PATH = "ax.models.torch.fully_bayesian.run_inference"
NUTS_PATH = "pyro.infer.mcmc.NUTS"
MCMC_PATH = "pyro.infer.mcmc.MCMC"

logger = get_logger(__name__)


def _get_dummy_mcmc_samples(
    num_samples: int,
    num_outputs: int,
    dtype: torch.dtype,
    device: torch.device,
    perturb_sd: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    tkwargs = {"dtype": dtype, "device": device}
    dummy_sample_list = []
    for i in range(num_outputs):
        dummy_samples = {
            # use real MAP values with tiny perturbations
            # so that the generation code below has feasible in-sample
            # points
            "lengthscale": (i + 1) * torch.tensor([[1 / 3, 1 / 3, 1 / 3]], **tkwargs)
            + perturb_sd * torch.randn(num_samples, 1, 3, **tkwargs),
            "outputscale": (i + 1) * torch.tensor(2.3436, **tkwargs)
            + perturb_sd * torch.randn(num_samples, **tkwargs),
            "mean": (i + 1) * torch.tensor([3.5000], **tkwargs)
            + perturb_sd * torch.randn(num_samples, **tkwargs),
        }
        dummy_samples["kernel_tausq"] = (i + 1) * torch.tensor(0.5, **tkwargs)
        dummy_samples["_kernel_inv_length_sq"] = (
            1.0 / dummy_samples["lengthscale"].sqrt()
        )
        dummy_sample_list.append(dummy_samples)
    return dummy_sample_list


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


try:
    from pyro.infer.mcmc import MCMC, NUTS

    class BaseFullyBayesianBotorchModelTest(ABC):
        model_cls: Type[FullyBayesianBotorchModel]

        def test_FullyBayesianBotorchModel(self, dtype=torch.float, cuda=False):
            # test deprecation warning
            warnings.resetwarnings()  # this is necessary for building in mode/opt
            warnings.simplefilter("always", append=True)
            with warnings.catch_warnings(record=True) as ws:
                self.model_cls(use_saas=True)
                self.assertTrue(
                    any(issubclass(w.category, DeprecationWarning) for w in ws)
                )
                msg = "Passing `use_saas` is no longer supported"
                self.assertTrue(any(msg in str(w.message) for w in ws))
            Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
                dtype=dtype, cuda=cuda, constant_noise=True
            )
            Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
                dtype=dtype, cuda=cuda, constant_noise=True
            )
            Yvars_inferred_noise = [
                torch.full_like(Yvars1[0], float("nan")),
                torch.full_like(Yvars2[0], float("nan")),
            ]
            # make input different for each output
            Xs2_diff = [Xs2[0] + 0.1]
            Xs = Xs1 + Xs2_diff
            Ys = Ys1 + Ys2

            options = product([True, False], [True, False], ["matern", "rbf"])
            for inferred_noise, use_input_warping, gp_kernel in options:
                Yvars = Yvars_inferred_noise if inferred_noise else Yvars1 + Yvars2
                model = self.model_cls(
                    use_input_warping=use_input_warping,
                    thinning=1,
                    num_samples=4,
                    disable_progbar=True,
                    max_tree_depth=1,
                    gp_kernel=gp_kernel,
                    verbose=True,
                )
                if use_input_warping:
                    self.assertTrue(model.use_input_warping)
                # Test ModelListGP
                # make training data different for each output
                tkwargs = {"dtype": dtype, "device": Xs1[0].device}
                dummy_samples_list = _get_dummy_mcmc_samples(
                    num_samples=4, num_outputs=2, **tkwargs
                )
                for dummy_samples in dummy_samples_list:
                    if use_input_warping:
                        dummy_samples["c0"] = (
                            torch.rand(4, 1, Xs1[0].shape[-1], **tkwargs) * 0.5 + 0.1
                        )
                        dummy_samples["c1"] = (
                            torch.rand(4, 1, Xs1[0].shape[-1], **tkwargs) * 0.5 + 0.1
                        )
                    if inferred_noise:
                        dummy_samples["noise"] = torch.rand(4, 1, **tkwargs).clamp_min(
                            MIN_INFERRED_NOISE_LEVEL
                        )

                with mock.patch(
                    RUN_INFERENCE_PATH,
                    side_effect=dummy_samples_list,
                ) as _mock_fit_model:
                    model.fit(
                        Xs=Xs,
                        Ys=Ys,
                        Yvars=Yvars,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns,
                            bounds=bounds,
                            task_features=tfs,
                        ),
                        metric_names=mns,
                    )
                    self.assertEqual(_mock_fit_model.call_count, 2)
                    for i, call in enumerate(_mock_fit_model.call_args_list):
                        _, ckwargs = call
                        X = Xs[i]
                        Y = Ys[i]
                        Yvar = Yvars[i]
                        self.assertIs(ckwargs["pyro_model"], single_task_pyro_model)

                        self.assertTrue(torch.equal(ckwargs["X"], X))
                        self.assertTrue(torch.equal(ckwargs["Y"], Y))
                        if inferred_noise:
                            self.assertTrue(torch.isnan(ckwargs["Yvar"]).all())
                        else:
                            self.assertTrue(torch.equal(ckwargs["Yvar"], Yvar))
                        self.assertEqual(ckwargs["num_samples"], 4)
                        self.assertEqual(ckwargs["warmup_steps"], 512)
                        self.assertEqual(ckwargs["max_tree_depth"], 1)
                        self.assertTrue(ckwargs["disable_progbar"])
                        self.assertEqual(
                            ckwargs["use_input_warping"], use_input_warping
                        )
                        self.assertEqual(ckwargs["gp_kernel"], gp_kernel)
                        self.assertTrue(ckwargs["verbose"])

                        # Check attributes
                        self.assertTrue(torch.equal(model.Xs[i], Xs[i]))
                        self.assertEqual(model.dtype, Xs[i].dtype)
                        self.assertEqual(model.device, Xs[i].device)
                        self.assertIsInstance(model.model, ModelListGP)

                        # Check fitting
                        # Note each model in the model list is a batched model, where
                        # the batch dim corresponds to the MCMC samples
                        model_list = model.model.models
                        # Put model in `eval` mode to transform the train inputs.
                        m = model_list[i].eval()
                        # check mcmc samples
                        dummy_samples = dummy_samples_list[i]
                        expected_train_inputs = Xs[i].expand(4, *Xs[i].shape)
                        if use_input_warping:
                            # train inputs should be warped inputs
                            expected_train_inputs = m.input_transform(
                                expected_train_inputs
                            )
                        self.assertTrue(
                            torch.equal(
                                m.train_inputs[0],
                                expected_train_inputs,
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                m.train_targets,
                                Ys[i].view(1, -1).expand(4, Ys[i].numel()),
                            )
                        )
                        expected_noise = (
                            dummy_samples["noise"].view(m.likelihood.noise.shape)
                            if inferred_noise
                            else Yvars[i].view(1, -1).expand(4, Yvars[i].numel())
                        )
                        self.assertTrue(
                            torch.allclose(
                                m.likelihood.noise.detach(),
                                expected_noise,
                            )
                        )
                        self.assertIsInstance(m.likelihood, _GaussianLikelihoodBase)
                        self.assertTrue(
                            torch.allclose(
                                m.covar_module.base_kernel.lengthscale.detach(),
                                dummy_samples["lengthscale"].view(
                                    m.covar_module.base_kernel.lengthscale.shape
                                ),
                                rtol=1e-4,
                                atol=1e-6,
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                m.covar_module.outputscale.detach(),
                                dummy_samples["outputscale"].view(
                                    m.covar_module.outputscale.shape
                                ),
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                m.mean_module.constant.detach(),
                                dummy_samples["mean"].view(
                                    m.mean_module.constant.shape
                                ),
                            )
                        )
                        if use_input_warping:
                            self.assertTrue(hasattr(m, "input_transform"))
                            self.assertIsInstance(m.input_transform, Warp)
                            self.assertTrue(
                                torch.equal(
                                    m.input_transform.concentration0,
                                    dummy_samples_list[i]["c0"],
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    m.input_transform.concentration1,
                                    dummy_samples_list[i]["c1"],
                                )
                            )
                        else:
                            self.assertFalse(hasattr(m, "input_transform"))
                # test that multi-task is not implemented
                (
                    Xs_mt,
                    Ys_mt,
                    Yvars_mt,
                    bounds_mt,
                    tfs_mt,
                    fns_mt,
                    mns_mt,
                ) = get_torch_test_data(
                    dtype=dtype, cuda=cuda, constant_noise=True, task_features=[2]
                )
                with mock.patch(
                    RUN_INFERENCE_PATH,
                    side_effect=dummy_samples_list,
                ) as _mock_fit_model, self.assertRaises(NotImplementedError):
                    model.fit(
                        Xs=Xs_mt,
                        Ys=Ys_mt,
                        Yvars=Yvars_mt,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns_mt,
                            bounds=bounds_mt,
                            task_features=tfs_mt,
                        ),
                        metric_names=mns_mt,
                    )
                with mock.patch(
                    RUN_INFERENCE_PATH,
                    side_effect=dummy_samples_list,
                ) as _mock_fit_model, self.assertRaises(NotImplementedError):
                    model.fit(
                        Xs=Xs1 + Xs2,
                        Ys=Ys1 + Ys2,
                        Yvars=Yvars1 + Yvars2,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns,
                            bounds=bounds,
                            fidelity_features=[0],
                        ),
                        metric_names=mns,
                    )
                # fit model with same inputs (otherwise X_observed will be None)
                model = self.model_cls(
                    use_input_warping=use_input_warping,
                    thinning=1,
                    num_samples=4,
                    disable_progbar=True,
                    max_tree_depth=1,
                    gp_kernel=gp_kernel,
                )
                Yvars = Yvars1 + Yvars2
                dummy_samples_list = _get_dummy_mcmc_samples(
                    num_samples=4, num_outputs=2, **tkwargs
                )
                with mock.patch(
                    RUN_INFERENCE_PATH,
                    side_effect=dummy_samples_list,
                ) as _mock_fit_model:
                    model.fit(
                        Xs=Xs1 + Xs2,
                        Ys=Ys1 + Ys2,
                        Yvars=Yvars,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns,
                            bounds=bounds,
                            task_features=tfs,
                        ),
                        metric_names=mns,
                    )

                # Check the hyperparameters and shapes
                self.assertEqual(len(model.model.models), 2)
                m1, m2 = model.model.models[0], model.model.models[1]
                # Mean
                self.assertEqual(m1.mean_module.constant.shape, (4, 1))
                self.assertFalse(
                    torch.isclose(
                        m1.mean_module.constant, m2.mean_module.constant
                    ).any()
                )
                # Outputscales
                self.assertEqual(m1.covar_module.outputscale.shape, (4,))
                self.assertFalse(
                    torch.isclose(
                        m1.covar_module.outputscale, m2.covar_module.outputscale
                    ).any()
                )
                # Lengthscales
                self.assertEqual(
                    m1.covar_module.base_kernel.lengthscale.shape, (4, 1, 3)
                )
                self.assertFalse(
                    torch.isclose(
                        m1.covar_module.base_kernel.lengthscale,
                        m2.covar_module.base_kernel.lengthscale,
                    ).any()
                )

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
                self.assertTrue(
                    torch.abs(infeasible_cost - expected_infeasible_cost) < 1e-5
                )

                # Check prediction
                X = torch.tensor([[6.0, 7.0, 8.0]], **tkwargs)
                f_mean, f_cov = model.predict(X)
                self.assertTrue(f_mean.shape == torch.Size([1, 2]))
                self.assertTrue(f_cov.shape == torch.Size([1, 2, 2]))

                # Check generation
                objective_weights = torch.tensor(
                    [1.0, 0.0]
                    if self.model_cls is FullyBayesianBotorchModel
                    else [1.0, 1.0],
                    **tkwargs,
                )
                outcome_constraints = (
                    torch.tensor([[0.0, 1.0]], **tkwargs),
                    torch.tensor([[5.0]], **tkwargs),
                )
                gen_kwargs = (
                    {"objective_thresholds": torch.zeros(2, **tkwargs)}
                    if self.model_cls is FullyBayesianMOOBotorchModel
                    else {}
                )
                linear_constraints = (
                    torch.tensor([[0.0, 1.0, 1.0]]),
                    torch.tensor([[100.0]]),
                )
                fixed_features = None
                pending_observations = [
                    torch.tensor([[1.0, 3.0, 4.0]], **tkwargs),
                    torch.tensor([[2.0, 6.0, 8.0]], **tkwargs),
                ]
                n = 3

                X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
                acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
                model_gen_options = {
                    Keys.OPTIMIZER_KWARGS: {"maxiter": 1},
                    Keys.ACQF_KWARGS: {"mc_samples": 3},
                }
                # test sequential optimize with constraints
                with mock.patch(
                    "ax.models.torch.botorch_defaults.optimize_acqf",
                    return_value=(X_dummy, acqfv_dummy),
                ) as _:
                    Xgen, wgen, gen_metadata, cand_metadata = model.gen(
                        n=n,
                        bounds=bounds,
                        objective_weights=objective_weights,
                        outcome_constraints=outcome_constraints,
                        linear_constraints=linear_constraints,
                        fixed_features=fixed_features,
                        pending_observations=pending_observations,
                        model_gen_options=model_gen_options,
                        rounding_func=dummy_func,
                        **gen_kwargs,
                    )
                    # note: gen() always returns CPU tensors
                    self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
                    self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))

                # actually test optimization for 1 step without constraints
                with mock.patch(
                    "ax.models.torch.botorch_defaults.optimize_acqf",
                    wraps=optimize_acqf,
                    return_value=(X_dummy, acqfv_dummy),
                ) as _:
                    Xgen, wgen, gen_metadata, cand_metadata = model.gen(
                        n=n,
                        bounds=bounds,
                        objective_weights=objective_weights,
                        outcome_constraints=outcome_constraints,
                        fixed_features=fixed_features,
                        pending_observations=pending_observations,
                        model_gen_options=model_gen_options,
                        **gen_kwargs,
                    )
                    # note: gen() always returns CPU tensors
                    self.assertTrue(torch.equal(Xgen, X_dummy.cpu()))
                    self.assertTrue(torch.equal(wgen, torch.ones(n, dtype=dtype)))

                # Check best point selection
                xbest = model.best_point(
                    bounds=bounds, objective_weights=objective_weights
                )
                xbest = model.best_point(
                    bounds=bounds,
                    objective_weights=objective_weights,
                    fixed_features={0: 100.0},
                )
                self.assertIsNone(xbest)

                # Test cross-validation
                mean, variance = model.cross_validate(
                    Xs_train=Xs1 + Xs2,
                    Ys_train=Ys,
                    Yvars_train=Yvars,
                    X_test=torch.tensor(
                        [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], dtype=dtype, device=device
                    ),
                )
                self.assertTrue(mean.shape == torch.Size([2, 2]))
                self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

                # Test cross-validation with refit_on_cv
                model.refit_on_cv = True
                with mock.patch(
                    RUN_INFERENCE_PATH,
                    side_effect=dummy_samples_list,
                ) as _mock_fit_model:
                    mean, variance = model.cross_validate(
                        Xs_train=Xs1 + Xs2,
                        Ys_train=Ys,
                        Yvars_train=Yvars,
                        X_test=torch.tensor(
                            [[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]],
                            dtype=dtype,
                            device=device,
                        ),
                    )
                    self.assertTrue(mean.shape == torch.Size([2, 2]))
                    self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

                # Test update
                model.refit_on_update = False
                model.update(Xs=Xs2 + Xs2, Ys=Ys2 + Ys2, Yvars=Yvars2 + Yvars2)

                # Test feature_importances
                importances = model.feature_importances()
                self.assertEqual(importances.shape, torch.Size([2, 1, 3]))

                # When calling update directly, the data is completely overwritten.
                self.assertTrue(torch.equal(model.Xs[0], Xs2[0]))
                self.assertTrue(torch.equal(model.Xs[1], Xs2[0]))
                self.assertTrue(torch.equal(model.Ys[0], Ys2[0]))
                self.assertTrue(torch.equal(model.Yvars[0], Yvars2[0]))

                model.refit_on_update = True
                with mock.patch(
                    RUN_INFERENCE_PATH, side_effect=dummy_samples_list
                ) as _mock_fit_model:
                    model.update(Xs=Xs2 + Xs2, Ys=Ys2 + Ys2, Yvars=Yvars2 + Yvars2)

                # test unfit model CV, update, and feature_importances
                unfit_model = self.model_cls()
                with self.assertRaises(RuntimeError):
                    unfit_model.cross_validate(
                        Xs_train=Xs1 + Xs2,
                        Ys_train=Ys1 + Ys2,
                        Yvars_train=Yvars1 + Yvars2,
                        X_test=Xs1[0],
                    )
                with self.assertRaises(RuntimeError):
                    unfit_model.update(
                        Xs=Xs1 + Xs2, Ys=Ys1 + Ys2, Yvars=Yvars1 + Yvars2
                    )
                with self.assertRaises(RuntimeError):
                    unfit_model.feature_importances()

        def test_saasbo_sample(self):
            for use_input_warping, gp_kernel in product(
                [False, True], ["rbf", "matern"]
            ):
                with torch.random.fork_rng():
                    torch.manual_seed(0)
                    X = torch.randn(3, 2)
                    Y = torch.randn(3, 1)
                    Yvar = torch.randn(3, 1)
                    kernel = NUTS(single_task_pyro_model, max_tree_depth=1)
                    mcmc = MCMC(kernel, warmup_steps=0, num_samples=1)
                    mcmc.run(
                        X,
                        Y,
                        Yvar,
                        use_input_warping=use_input_warping,
                        gp_kernel=gp_kernel,
                    )
                    samples = mcmc.get_samples()
                    self.assertTrue("kernel_tausq" in samples)
                    self.assertTrue("_kernel_inv_length_sq" in samples)
                    self.assertTrue("lengthscale" not in samples)
                    if use_input_warping:
                        self.assertIn("c0", samples)
                        self.assertIn("c1", samples)
                    else:
                        self.assertNotIn("c0", samples)
                        self.assertNotIn("c1", samples)

        def test_gp_kernels(self):
            torch.manual_seed(0)
            X = torch.randn(3, 2)
            Y = torch.randn(3, 1)
            Yvar = torch.randn(3, 1)
            kernel = NUTS(single_task_pyro_model, max_tree_depth=1)
            with self.assertRaises(ValueError):
                mcmc = MCMC(kernel, warmup_steps=0, num_samples=1)
                mcmc.run(
                    X,
                    Y,
                    Yvar,
                    gp_kernel="some_kernel_we_dont_support",
                )

        def test_FullyBayesianBotorchModel_cuda(self):
            if torch.cuda.is_available():
                self.test_FullyBayesianBotorchModel(cuda=True)

        def test_FullyBayesianBotorchModel_double(self):
            self.test_FullyBayesianBotorchModel(dtype=torch.double)

        def test_FullyBayesianBotorchModel_double_cuda(self):
            if torch.cuda.is_available():
                self.test_FullyBayesianBotorchModel(dtype=torch.double, cuda=True)

        def test_FullyBayesianBotorchModelConstraints(self):
            Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
                dtype=torch.float, cuda=False, constant_noise=True
            )
            Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
                dtype=torch.float, cuda=False, constant_noise=True
            )
            # make infeasible
            Xs2[0] = -1 * Xs2[0]
            objective_weights = torch.tensor(
                [-1.0, 1.0], dtype=torch.float, device=torch.device("cpu")
            )
            n = 3
            model = self.model_cls(
                num_samples=4,
                thinning=1,
                disable_progbar=True,
                max_tree_depth=1,
            )
            dummy_samples = _get_dummy_mcmc_samples(
                num_samples=4, num_outputs=2, dtype=torch.float, device=Xs1[0].device
            )
            with mock.patch(
                RUN_INFERENCE_PATH, side_effect=dummy_samples
            ) as _mock_fit_model:
                model.fit(
                    Xs=Xs1 + Xs2,
                    Ys=Ys1 + Ys2,
                    Yvars=Yvars1 + Yvars2,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=fns,
                        bounds=bounds,
                        task_features=tfs,
                    ),
                    metric_names=mns,
                )
                self.assertEqual(_mock_fit_model.call_count, 2)

            # because there are no feasible points:
            with self.assertRaises(ValueError):
                model.gen(n, bounds, objective_weights)

        def test_FullyBayesianBotorchModelPyro(self, cuda=False):
            for dtype in (torch.float, torch.double):
                Xs1, Ys1, raw_Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
                    dtype=dtype, cuda=cuda, constant_noise=True
                )
                Xs2, Ys2, raw_Yvars2, _, _, _, _ = get_torch_test_data(
                    dtype=dtype, cuda=cuda, constant_noise=True
                )
                options = product([False, True], [False, True], ["rbf", "matern"])
                for inferred_noise, use_input_warping, gp_kernel in options:
                    model = self.model_cls(
                        num_samples=4,
                        warmup_steps=0,
                        thinning=1,
                        use_input_warping=use_input_warping,
                        disable_progbar=True,
                        max_tree_depth=1,
                        gp_kernel=gp_kernel,
                        verbose=True,
                    )
                    if inferred_noise:
                        Yvars1 = [torch.full_like(raw_Yvars1[0], float("nan"))]
                        Yvars2 = [torch.full_like(raw_Yvars2[0], float("nan"))]
                    else:
                        Yvars1 = raw_Yvars1
                        Yvars2 = raw_Yvars2

                    dummy_samples = _get_dummy_mcmc_samples(
                        num_samples=4,
                        num_outputs=2,
                        dtype=dtype,
                        device=Xs1[0].device,
                    )
                    with ExitStack() as es:
                        _mock_fit_model = es.enter_context(
                            mock.patch(RUN_INFERENCE_PATH, side_effect=dummy_samples)
                        )
                        model.fit(
                            Xs=Xs1 + Xs2,
                            Ys=Ys1 + Ys2,
                            Yvars=Yvars1 + Yvars2,
                            search_space_digest=SearchSpaceDigest(
                                feature_names=fns,
                                bounds=bounds,
                                task_features=tfs,
                            ),
                            metric_names=mns,
                        )
                        # check run_inference arguments
                        self.assertEqual(_mock_fit_model.call_count, 2)
                        _, ckwargs = _mock_fit_model.call_args
                        self.assertIs(ckwargs["pyro_model"], single_task_pyro_model)

                        self.assertTrue(torch.equal(ckwargs["X"], Xs1[0]))
                        self.assertTrue(torch.equal(ckwargs["Y"], Ys1[0]))
                        if inferred_noise:
                            self.assertTrue(torch.isnan(ckwargs["Yvar"]).all())
                        else:
                            self.assertTrue(torch.equal(ckwargs["Yvar"], Yvars1[0]))
                        self.assertEqual(ckwargs["num_samples"], 4)
                        self.assertEqual(ckwargs["warmup_steps"], 0)
                        self.assertEqual(ckwargs["max_tree_depth"], 1)
                        self.assertTrue(ckwargs["disable_progbar"])
                        self.assertEqual(
                            ckwargs["use_input_warping"], use_input_warping
                        )
                        self.assertEqual(ckwargs["gp_kernel"], gp_kernel)
                        self.assertTrue(ckwargs["verbose"])

                    with ExitStack() as es:
                        _mock_mcmc = es.enter_context(mock.patch(MCMC_PATH))
                        _mock_mcmc.return_value.get_samples.side_effect = dummy_samples
                        _mock_nuts = es.enter_context(mock.patch(NUTS_PATH))
                        model.fit(
                            Xs=Xs1 + Xs2,
                            Ys=Ys1 + Ys2,
                            Yvars=Yvars1 + Yvars2,
                            search_space_digest=SearchSpaceDigest(
                                feature_names=fns,
                                bounds=bounds,
                                task_features=tfs,
                            ),
                            metric_names=mns,
                        )
                        # check MCMC.__init__ arguments
                        self.assertEqual(_mock_mcmc.call_count, 2)
                        _, ckwargs = _mock_mcmc.call_args
                        self.assertEqual(ckwargs["num_samples"], 4)
                        self.assertEqual(ckwargs["warmup_steps"], 0)
                        self.assertTrue(ckwargs["disable_progbar"])
                        # check NUTS.__init__ arguments
                        _mock_nuts.assert_called_with(
                            single_task_pyro_model,
                            jit_compile=True,
                            full_mass=True,
                            ignore_jit_warnings=True,
                            max_tree_depth=1,
                        )
                    # now actually run pyro
                    if not use_input_warping:
                        # input warping is quite slow, so we omit it for
                        # testing purposes
                        model.fit(
                            Xs=Xs1 + Xs2,
                            Ys=Ys1 + Ys2,
                            Yvars=Yvars1 + Yvars2,
                            search_space_digest=SearchSpaceDigest(
                                feature_names=fns,
                                bounds=bounds,
                                task_features=tfs,
                            ),
                            metric_names=mns,
                        )

                        for m, X, Y, Yvar in zip(
                            model.model.models, Xs1 + Xs2, Ys1 + Ys2, Yvars1 + Yvars2
                        ):
                            self.assertTrue(
                                torch.equal(
                                    m.train_inputs[0],
                                    X.expand(4, *X.shape),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    m.train_targets,
                                    Y.view(1, -1).expand(4, Y.numel()),
                                )
                            )
                            # check shapes of sampled parameters
                            if not inferred_noise:
                                self.assertTrue(
                                    torch.allclose(
                                        m.likelihood.noise.detach(),
                                        Yvar.view(1, -1).expand(4, Yvar.numel()),
                                    )
                                )
                            else:
                                self.assertEqual(
                                    m.likelihood.noise.shape, torch.Size([4, 1])
                                )

                            self.assertEqual(
                                m.covar_module.base_kernel.lengthscale.shape,
                                torch.Size([4, 1, X.shape[-1]]),
                            )
                            self.assertEqual(
                                m.covar_module.outputscale.shape, torch.Size([4])
                            )
                            self.assertEqual(
                                m.mean_module.constant.shape,
                                torch.Size([4, 1]),
                            )
                            if use_input_warping:
                                self.assertTrue(hasattr(m, "input_transform"))
                                self.assertIsInstance(m.input_transform, Warp)
                                self.assertEqual(
                                    m.input_transform.concentration0.shape,
                                    torch.Size([4, 1, 3]),
                                )
                                self.assertEqual(
                                    m.input_transform.concentration1.shape,
                                    torch.Size([4, 1, 3]),
                                )
                            else:
                                self.assertFalse(hasattr(m, "input_transform"))

        def test_FullyBayesianBotorchModelPyro_cuda(self):
            if torch.cuda.is_available():
                self.test_FullyBayesianBotorchModelPyro(cuda=True)

    class SingleObjectiveFullyBayesianBotorchModelTest(
        TestCase, BaseFullyBayesianBotorchModelTest
    ):
        model_cls = FullyBayesianBotorchModel

        def test_FullyBayesianBotorchModelOneOutcome(self):
            Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
                dtype=torch.float, cuda=False, constant_noise=True
            )
            for use_input_warping, gp_kernel in product(
                [True, False], ["rbf", "matern"]
            ):
                model = self.model_cls(
                    use_input_warping=use_input_warping,
                    num_samples=4,
                    thinning=1,
                    disable_progbar=True,
                    max_tree_depth=1,
                    gp_kernel=gp_kernel,
                )
                dummy_samples = _get_dummy_mcmc_samples(
                    num_samples=4,
                    num_outputs=1,
                    dtype=torch.float,
                    device=Xs1[0].device,
                )
                with mock.patch(
                    RUN_INFERENCE_PATH, side_effect=dummy_samples
                ) as _mock_fit_model:
                    model.fit(
                        Xs=Xs1,
                        Ys=Ys1,
                        Yvars=Yvars1,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns,
                            bounds=bounds,
                            task_features=tfs,
                        ),
                        metric_names=mns[0],
                    )
                    _mock_fit_model.assert_called_once()
                X = torch.rand(2, 3, dtype=torch.float)
                f_mean, f_cov = model.predict(X)
                self.assertTrue(f_mean.shape == torch.Size([2, 1]))
                self.assertTrue(f_cov.shape == torch.Size([2, 1, 1]))
                model_list = model.model.models
                self.assertTrue(len(model_list) == 1)
                if use_input_warping:
                    self.assertTrue(hasattr(model_list[0], "input_transform"))
                    self.assertIsInstance(model_list[0].input_transform, Warp)
                else:
                    self.assertFalse(hasattr(model_list[0], "input_transform"))

    class FullyBayesianMOOBotorchModelTest(TestCase, BaseFullyBayesianBotorchModelTest):
        model_cls = FullyBayesianMOOBotorchModel


except ImportError:
    logger.info("pyro not found. Skipping fully bayesian tests.")
    pass


class TestKernels(TestCase):
    def test_matern_kernel(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2
        # test matern 1/2
        res = matern_kernel(a, b, nu=0.5, lengthscale=lengthscale)
        actual = (
            torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float)
            .div_(-lengthscale)
            .exp()
        )
        self.assertLess(torch.norm(res - actual), 1e-3)
        # matern test 3/2
        res = matern_kernel(a, b, nu=1.5, lengthscale=lengthscale)
        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(
            sqrt(3) / lengthscale
        )
        actual = (dist + 1).mul(torch.exp(-dist))
        self.assertLess(torch.norm(res - actual), 1e-3)
        # matern test 5/2
        res = matern_kernel(a, b, nu=2.5, lengthscale=lengthscale)
        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(
            sqrt(5) / lengthscale
        )
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        self.assertLess(torch.norm(res - actual), 1e-3)

        # test k(x,x) with no gradients
        res = matern_kernel(b, b, nu=0.5, lengthscale=2.0)
        actual = (
            torch.tensor([[0, 2], [2, 0]], dtype=torch.float).div_(-lengthscale).exp()
        )
        self.assertLess(torch.norm(res - actual), 1e-3)

        # test unsupported nu
        with self.assertRaises(AxError):
            matern_kernel(b, b, nu=0.0, lengthscale=2.0)

    def test_rbf_kernel(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2
        # test rbf
        res = rbf_kernel(a, b, lengthscale=lengthscale)
        actual = (
            torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float)
            .pow_(2.0)
            .mul_(-0.5 / (lengthscale ** 2))
            .exp()
        )
        self.assertLess(torch.norm(res - actual), 1e-3)
