#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from ax.models.torch.botorch_defaults import (
    _get_model,
    get_and_fit_model,
    get_warping_transform,
)
from ax.utils.common.testutils import TestCase
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.prior import Prior


class BotorchDefaultsTest(TestCase):
    def test_get_model(self):
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 1)
        var = torch.zeros(2, 1)
        partial_var = torch.tensor([0, float("nan")]).unsqueeze(-1)
        unknown_var = torch.tensor([float("nan"), float("nan")]).unsqueeze(-1)
        model = _get_model(x, y, unknown_var, None)
        self.assertIsInstance(model, SingleTaskGP)

        model = _get_model(X=x, Y=y, Yvar=var)
        self.assertIsInstance(model, FixedNoiseGP)
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
        model = _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs)
        self.assertIsInstance(
            model.task_covar_module.IndexKernelPrior, LKJCovariancePrior
        )
        self.assertEqual(
            model.task_covar_module.IndexKernelPrior.sd_prior.concentration, 2.0
        )
        self.assertEqual(model.task_covar_module.IndexKernelPrior.sd_prior.rate, 0.44)
        self.assertEqual(
            model.task_covar_module.IndexKernelPrior.correlation_prior.eta, 0.6
        )

        kwargs2 = {"prior": {"type": LKJCovariancePrior}}
        model = _get_model(
            X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs2
        )
        self.assertIsInstance(
            model.task_covar_module.IndexKernelPrior, LKJCovariancePrior
        )
        self.assertEqual(
            model.task_covar_module.IndexKernelPrior.sd_prior.concentration, 1.0
        )
        self.assertEqual(model.task_covar_module.IndexKernelPrior.sd_prior.rate, 0.15)
        self.assertEqual(
            model.task_covar_module.IndexKernelPrior.correlation_prior.eta, 0.5
        )
        kwargs3 = {
            "prior": {
                "type": LKJCovariancePrior,
                "sd_prior": GammaPrior(2.0, 0.44),
                "eta": "hi",
            }
        }
        with self.assertRaises(ValueError):
            _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs3)

        kwargs5 = {
            "prior": {"type": Prior, "sd_prior": GammaPrior(2.0, 0.44), "eta": 0.5}
        }
        with self.assertRaises(NotImplementedError):
            _get_model(X=x, Y=y, Yvar=partial_var.clone(), task_feature=1, **kwargs5)

    @mock.patch("ax.models.torch.botorch_defaults._get_model", wraps=_get_model)
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

    def test_get_warping_transform(self):
        warp_tf = get_warping_transform(d=4)
        self.assertIsInstance(warp_tf, Warp)
        self.assertEqual(warp_tf.indices.tolist(), list(range(4)))
        warp_tf = get_warping_transform(d=4, task_feature=2)
        self.assertEqual(warp_tf.indices.tolist(), [0, 1, 3])
        warp_tf = get_warping_transform(d=4, batch_shape=torch.Size([2]))
        self.assertIsInstance(warp_tf, Warp)
        self.assertEqual(warp_tf.indices.tolist(), list(range(4)))
        self.assertEqual(warp_tf.batch_shape, torch.Size([2]))
