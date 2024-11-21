#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.input_constructors.covar_modules import (
    covar_module_argparse,
)
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.utils.common.testutils import TestCase
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors import GammaPrior


class DummyKernel(Kernel):
    pass


class CovarModuleArgparseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        X = torch.randn((10, 10))
        Y = torch.randn((10, 2))
        self.dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=[f"x{i}" for i in range(10)],
            outcome_names=["y1", "y2"],
        )

    def test_notImplemented(self) -> None:
        with self.assertRaises(NotImplementedError) as e:
            covar_module_argparse[
                type(None)
            ]  # passing `None` produces a different error
            self.assertTrue("Could not find signature for" in str(e))

    def test_register(self) -> None:
        @covar_module_argparse.register(DummyKernel)
        def _argparse(kernel: DummyKernel) -> None:
            pass

        self.assertEqual(covar_module_argparse[DummyKernel], _argparse)

    def test_fallback(self) -> None:
        with patch.dict(covar_module_argparse.funcs, {}):

            @covar_module_argparse.register(Kernel)
            def _argparse(covar_module_class: Kernel) -> None:
                pass

            self.assertEqual(covar_module_argparse[Kernel], _argparse)

    def test_argparse_kernel(self) -> None:
        covar_module_kwargs = covar_module_argparse(
            Kernel,
            botorch_model_class=SingleTaskGP,
            dataset=self.dataset,
        )

        self.assertEqual(covar_module_kwargs, {})

        covar_module_kwargs = covar_module_argparse(
            Kernel,
            botorch_model_class=SingleTaskGP,
            dataset=self.dataset,
            ard_num_dims=19,
            batch_shape=torch.Size([10]),
        )

        self.assertEqual(
            covar_module_kwargs, {"ard_num_dims": 19, "batch_shape": torch.Size([10])}
        )

    def test_argparse_scalematern_kernel(self) -> None:
        covar_module_kwargs_defaults = [
            {
                "ard_num_dims": 10,
                "lengthscale_prior_concentration": 6.0,
                "lengthscale_prior_rate": 3.0,
                "outputscale_prior_concentration": 2.0,
                "outputscale_prior_rate": 0.15,
                "batch_shape": torch.Size([2]),
            },
            {
                "ard_num_dims": 9,
                "lengthscale_prior_concentration": 6.0,
                "lengthscale_prior_rate": 3.0,
                "outputscale_prior_concentration": 2.0,
                "outputscale_prior_rate": 0.15,
                "batch_shape": torch.Size([]),
            },
        ]

        for i, botorch_model_class in enumerate([SingleTaskGP, MultiTaskGP]):
            covar_module_kwargs = covar_module_argparse(
                ScaleMaternKernel,
                botorch_model_class=botorch_model_class,
                dataset=self.dataset,
                lengthscale_prior=GammaPrior(6.0, 3.0),
                outputscale_prior=GammaPrior(2, 0.15),
            )

            covar_module_kwargs["lengthscale_prior_concentration"] = (
                covar_module_kwargs["lengthscale_prior"].concentration.item()
            )
            covar_module_kwargs["lengthscale_prior_rate"] = covar_module_kwargs[
                "lengthscale_prior"
            ].rate.item()

            covar_module_kwargs["outputscale_prior_concentration"] = (
                covar_module_kwargs["outputscale_prior"].concentration.item()
            )
            covar_module_kwargs["outputscale_prior_rate"] = covar_module_kwargs[
                "outputscale_prior"
            ].rate.item()

            covar_module_kwargs.pop("lengthscale_prior")
            covar_module_kwargs.pop("outputscale_prior")

            for key in covar_module_kwargs.keys():
                self.assertAlmostEqual(
                    covar_module_kwargs[key],
                    covar_module_kwargs_defaults[i][key],
                    places=4,
                )

        X = torch.randn((10, 10))
        Y = torch.randn((10, 1))
        dataset = SupervisedDataset(
            X=X, Y=Y, feature_names=[f"x{i}" for i in range(10)], outcome_names=["y"]
        )
        covar_module_kwargs = covar_module_argparse(
            ScaleMaternKernel,
            botorch_model_class=SingleTaskGP,
            dataset=dataset,
            lengthscale_prior=GammaPrior(6.0, 3.0),
            outputscale_prior=GammaPrior(2, 0.15),
        )

        self.assertEqual(covar_module_kwargs["batch_shape"], torch.Size([]))
