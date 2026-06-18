#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest.mock import patch

import torch
from ax.exceptions.core import UserInputError
from ax.generators.torch.botorch_modular.input_constructors.covar_modules import (
    covar_module_argparse,
)
from ax.generators.torch.botorch_modular.kernels import (
    DefaultMaternKernel,
    DefaultRBFKernel,
    ScaleMaternKernel,
    ScaleRBFLinearKernel,
)
from ax.utils.common.testutils import TestCase
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.linear_kernel import LinearKernel
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
                "active_dims": None,
            },
            {
                "ard_num_dims": 9,
                "lengthscale_prior_concentration": 6.0,
                "lengthscale_prior_rate": 3.0,
                "outputscale_prior_concentration": 2.0,
                "outputscale_prior_rate": 0.15,
                "batch_shape": torch.Size([]),
                "active_dims": None,
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

    def test_argparse_scale_rbf_linear_kernel(self) -> None:
        # SingleTaskGP infers ard_num_dims from the 10 features; the
        # multi-output dataset (Y has 2 columns) yields a batch shape of [2].
        # MultiTaskGP drops the task feature, so ard_num_dims is 9 and the
        # batch shape is empty.
        expected = [
            {"ard_num_dims": 10, "batch_shape": torch.Size([2])},
            {"ard_num_dims": 9, "batch_shape": torch.Size([])},
        ]
        for i, botorch_model_class in enumerate([SingleTaskGP, MultiTaskGP]):
            covar_module_kwargs = covar_module_argparse(
                ScaleRBFLinearKernel,
                botorch_model_class=botorch_model_class,
                dataset=self.dataset,
                lengthscale_prior=GammaPrior(6.0, 3.0),
                outputscale_prior=GammaPrior(2.0, 0.15),
                variance_prior=GammaPrior(1.0, 1.0),
            )
            self.assertEqual(
                covar_module_kwargs["ard_num_dims"], expected[i]["ard_num_dims"]
            )
            self.assertEqual(
                covar_module_kwargs["batch_shape"], expected[i]["batch_shape"]
            )
            # No active_dims requested, so none is set.
            self.assertIsNone(covar_module_kwargs["active_dims"])
            # Priors are passed straight through.
            self.assertAlmostEqual(
                covar_module_kwargs["lengthscale_prior"].concentration.item(),
                6.0,
                places=4,
            )
            self.assertAlmostEqual(
                covar_module_kwargs["outputscale_prior"].rate.item(), 0.15, places=4
            )
            self.assertAlmostEqual(
                covar_module_kwargs["variance_prior"].concentration.item(),
                1.0,
                places=4,
            )
            # The resulting kwargs can construct the kernel.
            kernel = ScaleRBFLinearKernel(**covar_module_kwargs)
            self.assertIsInstance(kernel, ScaleRBFLinearKernel)

    def test_argparse_scale_rbf_linear_kernel_active_dims(self) -> None:
        # Explicit active_dims is passed through and normalized; ard_num_dims is
        # set to the number of active dims.
        covar_module_kwargs = covar_module_argparse(
            ScaleRBFLinearKernel,
            botorch_model_class=SingleTaskGP,
            dataset=self.dataset,
            active_dims=[0, 2, 4],
        )
        self.assertEqual(covar_module_kwargs["active_dims"], [0, 2, 4])
        self.assertEqual(covar_module_kwargs["ard_num_dims"], 3)
        kernel = ScaleRBFLinearKernel(**covar_module_kwargs)
        self.assertIsInstance(kernel, ScaleRBFLinearKernel)

    def test_argparse_scale_rbf_linear_kernel_remove_task_features(self) -> None:
        # A SingleTaskGP on a MultiTaskDataset with remove_task_features=True
        # excludes the task feature from the kernel: active_dims drops the task
        # column and ard_num_dims shrinks accordingly.
        n, d = 10, 5
        task_feature_index = d - 1
        X = torch.cat([torch.randn((n, d - 1)), torch.randint(0, 2, (n, 1))], dim=-1)
        Y = torch.randn((n, 1))
        joint_dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=[f"x{j}" for j in range(d - 1)] + ["task"],
            outcome_names=["y"],
        )
        dataset = MultiTaskDataset.from_joint_dataset(
            dataset=joint_dataset,
            task_feature_index=task_feature_index,
            target_task_value=1,
        )
        covar_module_kwargs = covar_module_argparse(
            ScaleRBFLinearKernel,
            botorch_model_class=SingleTaskGP,
            dataset=dataset,
            remove_task_features=True,
        )
        # The task feature (last column, index d - 1) is excluded.
        self.assertEqual(covar_module_kwargs["active_dims"], list(range(d - 1)))
        self.assertEqual(covar_module_kwargs["ard_num_dims"], d - 1)
        kernel = ScaleRBFLinearKernel(**covar_module_kwargs)
        self.assertIsInstance(kernel, ScaleRBFLinearKernel)

    def test_argparse_default(self) -> None:
        for kernel_class in (DefaultRBFKernel, DefaultMaternKernel):
            with self.assertRaisesRegex(UserInputError, "Only one of"):
                covar_module_argparse(
                    kernel_class,
                    botorch_model_class=SingleTaskGP,
                    dataset=self.dataset,
                    inactive_features=["x1"],
                    active_dims=[0],
                )
            # Test with inactive features.
            covar_module_kwargs = covar_module_argparse(
                kernel_class,
                botorch_model_class=SingleTaskGP,
                dataset=self.dataset,
                inactive_features=["x9"],
            )
            expected = {
                "active_dims": list(range(9)),
                "batch_shape": torch.Size([2]),  # For 2 outputs.
                "ard_num_dims": 9,
                "lengthscale_prior": None,
            }
            self.assertEqual(covar_module_kwargs, expected)
            # Test other inputs.
            covar_module_kwargs = covar_module_argparse(
                kernel_class,
                botorch_model_class=SingleTaskGP,
                dataset=self.dataset,
                active_dims=[-3, -2],
                ard_num_dims=1,
                batch_shape=torch.Size([]),
            )
            expected = {
                "active_dims": [7, 8],
                "batch_shape": torch.Size([]),
                "ard_num_dims": 1,
                "lengthscale_prior": None,
            }
            self.assertEqual(covar_module_kwargs, expected)

    def test_argparse_linear(self) -> None:
        # Test other inputs.
        covar_module_kwargs = covar_module_argparse(
            LinearKernel,
            botorch_model_class=SingleTaskGP,
            dataset=self.dataset,
            active_dims=[-3, -2],
            ard_num_dims=1,
            batch_shape=torch.Size([]),
            variance_prior=GammaPrior(2, 0.15),
        )
        expected = {
            "active_dims": [7, 8],
            "batch_shape": torch.Size([]),
            "ard_num_dims": 1,
        }
        for k, v in expected.items():
            self.assertEqual(covar_module_kwargs[k], v)
        self.assertIsInstance(covar_module_kwargs["variance_prior"], GammaPrior)
        self.assertAlmostEqual(
            covar_module_kwargs["variance_prior"].concentration.item(), 2
        )
        self.assertAlmostEqual(covar_module_kwargs["variance_prior"].rate.item(), 0.15)
