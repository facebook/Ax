#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest import mock

import torch
from ax.models.torch.botorch_defaults import (
    _get_acquisition_func,
    _get_customized_covar_module,
    _get_model,
    get_and_fit_model,
    get_warping_transform,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.penalized import PenalizedMCObjective
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.module import Module
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.prior import Prior


class BotorchDefaultsTest(TestCase):
    def test_get_model(self) -> None:
        x = torch.rand(2, 2)
        y = torch.rand(2, 1)
        var = torch.rand(2, 1)
        partial_var = torch.tensor([0, float("nan")]).unsqueeze(-1)
        unknown_var = torch.tensor([float("nan"), float("nan")]).unsqueeze(-1)
        model = _get_model(x, y, unknown_var, None)
        self.assertIsInstance(model, SingleTaskGP)

        model = _get_model(X=x, Y=y, Yvar=var)
        self.assertIsInstance(model, FixedNoiseGP)
        self.assertEqual(
            # pyre-ignore
            model.covar_module.base_kernel.lengthscale_prior.concentration,
            3.0,
        )
        # pyre-ignore
        self.assertEqual(model.covar_module.base_kernel.lengthscale_prior.rate, 6.0)
        model = _get_model(X=x, Y=y, Yvar=unknown_var, task_feature=1)
        self.assertTrue(type(model) == MultiTaskGP)  # Don't accept subclasses.
        model = _get_model(X=x, Y=y, Yvar=var, task_feature=1)
        self.assertIsInstance(model, FixedNoiseMultiTaskGP)
        model = _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1)
        self.assertIsInstance(model, FixedNoiseMultiTaskGP)
        model = _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, rank=1)
        self.assertEqual(model._rank, 1)
        with self.assertRaises(ValueError):
            model = _get_model(X=x, Y=y, Yvar=partial_var, task_feature=None)
        model = _get_model(X=x, Y=y, Yvar=var, fidelity_features=[-1])
        self.assertTrue(isinstance(model, SingleTaskMultiFidelityGP))
        with self.assertRaises(NotImplementedError):
            _get_model(X=x, Y=y, Yvar=var, task_feature=1, fidelity_features=[-1])
        # test fixed prior
        kwargs = {
            "prior": {
                "type": LKJCovariancePrior,
                "sd_prior": GammaPrior(2.0, 0.44),
                "eta": 0.6,
            }
        }
        # pyre-fixme[6]: For 5th param expected `Optional[List[int]]` but got
        #  `Dict[str, Union[Type[LKJCovariancePrior], float, GammaPrior]]`.
        # pyre-fixme[6]: For 5th param expected `bool` but got `Dict[str,
        #  Union[Type[LKJCovariancePrior], float, GammaPrior]]`.
        model = _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs)
        self.assertIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior,
            LKJCovariancePrior,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior.sd_prior.concentration,
            2.0,
        )
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `IndexKernelPrior`.
        self.assertEqual(model.task_covar_module.IndexKernelPrior.sd_prior.rate, 0.44)
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior.correlation_prior.eta,
            0.6,
        )

        kwargs2 = {"prior": {"type": LKJCovariancePrior}}
        model = _get_model(
            X=x,
            Y=y,
            Yvar=partial_var.clone(),
            task_feature=1,
            # pyre-fixme[6]: For 5th param expected `Optional[List[int]]` but got
            #  `Dict[str, Type[LKJCovariancePrior]]`.
            # pyre-fixme[6]: For 5th param expected `bool` but got `Dict[str,
            #  Type[LKJCovariancePrior]]`.
            **kwargs2,
        )
        self.assertIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior,
            LKJCovariancePrior,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior.sd_prior.concentration,
            1.0,
        )
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `IndexKernelPrior`.
        self.assertEqual(model.task_covar_module.IndexKernelPrior.sd_prior.rate, 0.15)
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
            #  attribute `IndexKernelPrior`.
            model.task_covar_module.IndexKernelPrior.correlation_prior.eta,
            0.5,
        )
        kwargs3 = {
            "prior": {
                "type": LKJCovariancePrior,
                "sd_prior": GammaPrior(2.0, 0.44),
                "eta": "hi",
            }
        }
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 5th param expected `Optional[List[int]]` but got
            #  `Dict[str, Union[Type[LKJCovariancePrior], GammaPrior, str]]`.
            # pyre-fixme[6]: For 5th param expected `bool` but got `Dict[str,
            #  Union[Type[LKJCovariancePrior], GammaPrior, str]]`.
            _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs3)

        kwargs5 = {
            "prior": {"type": Prior, "sd_prior": GammaPrior(2.0, 0.44), "eta": 0.5}
        }
        with self.assertRaises(NotImplementedError):
            # pyre-fixme[6]: For 5th param expected `Optional[List[int]]` but got
            #  `Dict[str, Union[Type[Prior], float, GammaPrior]]`.
            # pyre-fixme[6]: For 5th param expected `bool` but got `Dict[str,
            #  Union[Type[Prior], float, GammaPrior]]`.
            _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs5)
        # test passing customized prior
        kwargs6 = {
            "prior": {
                "covar_module_prior": {"lengthscale_prior": GammaPrior(12.0, 2.0)},
                "type": LKJCovariancePrior,
            }
        }
        model = _get_model(X=x, Y=y, Yvar=var, **deepcopy(kwargs6))  # pyre-ignore
        self.assertIsInstance(model, FixedNoiseGP)
        self.assertEqual(
            # pyre-ignore
            model.covar_module.base_kernel.lengthscale_prior.concentration,
            12.0,
        )
        # pyre-ignore
        self.assertEqual(model.covar_module.base_kernel.lengthscale_prior.rate, 2.0)
        model = _get_model(
            X=x,
            Y=y,
            Yvar=unknown_var,
            task_feature=1,
            **deepcopy(kwargs6),  # pyre-ignore
        )
        self.assertTrue(type(model) == MultiTaskGP)
        self.assertEqual(
            model.covar_module.base_kernel.lengthscale_prior.concentration, 12.0
        )
        self.assertEqual(model.covar_module.base_kernel.lengthscale_prior.rate, 2.0)
        self.assertIsInstance(
            model.task_covar_module.IndexKernelPrior,
            LKJCovariancePrior,
        )
        model = _get_model(
            X=x, Y=y, Yvar=var, task_feature=1, **deepcopy(kwargs6)  # pyre-ignore
        )
        self.assertIsInstance(model, FixedNoiseMultiTaskGP)
        self.assertEqual(
            # pyre-ignore
            model.covar_module.base_kernel.lengthscale_prior.concentration,
            12.0,
        )
        # pyre-ignore
        self.assertEqual(model.covar_module.base_kernel.lengthscale_prior.rate, 2.0)
        self.assertIsInstance(
            # pyre-ignore
            model.task_covar_module.IndexKernelPrior,
            LKJCovariancePrior,
        )
        # test passing customized prior
        kwargs7 = {
            "prior": {
                "covar_module_prior": {"lengthscale_prior": GammaPrior(12.0, 2.0)},
            }
        }
        covar_module = MaternKernel(
            nu=2.5,
            ard_num_dims=2,
            lengthscale_prior=GammaPrior(6.0, 6.0),
        )
        model = _get_model(
            X=x, Y=y, Yvar=var, covar_module=covar_module, **kwargs7  # pyre-ignore
        )
        self.assertIsInstance(model, FixedNoiseGP)
        self.assertEqual(covar_module, model.covar_module)

    @mock.patch("ax.models.torch.botorch_defaults._get_model", wraps=_get_model)
    @fast_botorch_optimize
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_task_feature(self, get_model_mock):
        x = [torch.zeros(2, 2)]
        y = [torch.zeros(2, 1)]
        yvars = [torch.ones(2, 1)]
        get_and_fit_model(
            Xs=x,
            Ys=y,
            Yvars=yvars,
            task_features=[1],
            fidelity_features=[],
            metric_names=["L2NormMetric"],
            state_dict=None,
            refit_model=False,
        )
        # Check that task feature was correctly passed to _get_model
        self.assertEqual(get_model_mock.mock_calls[0][2]["task_feature"], 1)

        # check error on multiple task features
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[0, 1],
                fidelity_features=[],
                metric_names=["L2NormMetric"],
                state_dict=None,
                refit_model=False,
            )

        # check error on multiple fidelity features
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[],
                fidelity_features=[-1, -2],
                metric_names=["L2NormMetric"],
                state_dict=None,
                refit_model=False,
            )

        # check error on botch task and fidelity feature
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[1],
                fidelity_features=[-1],
                metric_names=["L2NormMetric"],
                state_dict=None,
                refit_model=False,
            )

    @mock.patch("ax.models.torch.botorch_defaults._get_model", wraps=_get_model)
    @fast_botorch_optimize
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_pass_customized_prior(self, get_model_mock):
        x = [torch.zeros(2, 2)]
        y = [torch.zeros(2, 1)]
        yvars = [torch.ones(2, 1)]
        kwarg = {
            "prior": {
                "covar_module_prior": {
                    "lengthscale_prior": GammaPrior(12.0, 2.0),
                    "outputscale_prior": GammaPrior(2.0, 12.0),
                },
            }
        }
        model = get_and_fit_model(
            Xs=x,
            Ys=y,
            Yvars=yvars,
            task_features=[],
            fidelity_features=[],
            metric_names=["L2NormMetric"],
            state_dict=None,
            refit_model=False,
            **kwarg,  # pyre-ignore
        )
        self.assertTrue(type(model) == FixedNoiseGP)
        self.assertEqual(
            # pyre-ignore
            model.covar_module.base_kernel.lengthscale_prior.concentration,
            12.0,
        )
        self.assertEqual(model.covar_module.base_kernel.lengthscale_prior.rate, 2.0)
        # pyre-ignore
        self.assertEqual(model.covar_module.outputscale_prior.concentration, 2.0)
        self.assertEqual(model.covar_module.outputscale_prior.rate, 12.0)

        model = get_and_fit_model(
            Xs=x + x,
            Ys=y + y,
            Yvars=yvars + yvars,
            task_features=[1],
            fidelity_features=[],
            metric_names=["L2NormMetric", "L2NormMetric2"],
            state_dict=None,
            refit_model=False,
            **kwarg,  # pyre-ignore
        )
        for m in model.models:  # pyre-ignore
            self.assertTrue(type(m) == FixedNoiseMultiTaskGP)
            self.assertEqual(
                m.covar_module.base_kernel.lengthscale_prior.concentration,
                12.0,
            )
            self.assertEqual(m.covar_module.base_kernel.lengthscale_prior.rate, 2.0)
            self.assertEqual(m.covar_module.outputscale_prior.concentration, 2.0)
            self.assertEqual(m.covar_module.outputscale_prior.rate, 12.0)

    def test_get_acquisition_func(self) -> None:
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 1)
        unknown_var = torch.tensor([float("nan"), float("nan")]).unsqueeze(-1)
        model = _get_model(x, y, unknown_var, None)
        objective_weights = torch.tensor([1.0])
        outcome_constraints = (
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[5.0]]),
        )
        X_observed = torch.zeros(2, 2)
        with self.assertRaises(ValueError) as cm:
            _get_acquisition_func(
                model=model,
                acquisition_function_name="qNEI",
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                constrained_mc_objective=None,
            )
        self.assertEqual(
            "constrained_mc_objective cannot be set to None "
            "when applying outcome constraints.",
            str(cm.exception),
        )
        with self.assertRaises(RuntimeError):
            _get_acquisition_func(
                model=model,
                acquisition_function_name="qNEI",
                objective_weights=objective_weights,
                mc_objective=PenalizedMCObjective,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
            )

    def test_get_customized_covar_module(self) -> None:
        ard_num_dims = 3
        batch_shape = torch.Size([2])
        covar_module = _get_customized_covar_module(
            covar_module_prior_dict={},
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            task_feature=None,
        )
        self.assertIsInstance(covar_module, Module)
        self.assertIsInstance(covar_module, ScaleKernel)
        self.assertIsInstance(covar_module.outputscale_prior, GammaPrior)
        self.assertEqual(covar_module.outputscale_prior.concentration, 2.0)
        self.assertEqual(covar_module.outputscale_prior.rate, 0.15)
        self.assertIsInstance(covar_module.base_kernel, MaternKernel)
        self.assertIsInstance(covar_module.base_kernel.lengthscale_prior, GammaPrior)
        self.assertEqual(covar_module.base_kernel.lengthscale_prior.concentration, 3.0)
        self.assertEqual(covar_module.base_kernel.lengthscale_prior.rate, 6.0)
        self.assertEqual(covar_module.base_kernel.ard_num_dims, ard_num_dims)
        self.assertEqual(covar_module.base_kernel.batch_shape, batch_shape)

        covar_module = _get_customized_covar_module(
            covar_module_prior_dict={
                "lengthscale_prior": GammaPrior(12.0, 2.0),
                "outputscale_prior": GammaPrior(2.0, 12.0),
            },
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            task_feature=3,
        )
        self.assertIsInstance(covar_module, Module)
        self.assertIsInstance(covar_module, ScaleKernel)
        self.assertIsInstance(covar_module.outputscale_prior, GammaPrior)
        self.assertEqual(covar_module.outputscale_prior.concentration, 2.0)
        self.assertEqual(covar_module.outputscale_prior.rate, 12.0)
        self.assertIsInstance(covar_module.base_kernel, MaternKernel)
        self.assertIsInstance(covar_module.base_kernel.lengthscale_prior, GammaPrior)
        self.assertEqual(covar_module.base_kernel.lengthscale_prior.concentration, 12.0)
        self.assertEqual(covar_module.base_kernel.lengthscale_prior.rate, 2.0)
        self.assertEqual(covar_module.base_kernel.ard_num_dims, ard_num_dims - 1)
        self.assertEqual(covar_module.base_kernel.batch_shape, batch_shape)

    def test_get_warping_transform(self) -> None:
        warp_tf = get_warping_transform(d=4)
        self.assertIsInstance(warp_tf, Warp)
        self.assertEqual(warp_tf.indices.tolist(), list(range(4)))
        warp_tf = get_warping_transform(d=4, task_feature=2)
        self.assertEqual(warp_tf.indices.tolist(), [0, 1, 3])
        warp_tf = get_warping_transform(d=4, batch_shape=torch.Size([2]))
        self.assertIsInstance(warp_tf, Warp)
        self.assertEqual(warp_tf.indices.tolist(), list(range(4)))
        self.assertEqual(warp_tf.batch_shape, torch.Size([2]))
