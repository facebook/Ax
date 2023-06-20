#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.utils.common.testutils import TestCase
from gpytorch.kernels import MaternKernel
from gpytorch.priors import GammaPrior


class KernelsTest(TestCase):
    def test_scalematern_kernel(self) -> None:
        covar = ScaleMaternKernel(
            ard_num_dims=10,
            lengthscale_prior=GammaPrior(6.0, 3.0),
            outputscale_prior=GammaPrior(2.0, 0.15),
            batch_shape=torch.Size([2]),
        )
        self.assertTrue(isinstance(covar.base_kernel, MaternKernel))
        self.assertTrue(isinstance(covar.base_kernel, MaternKernel))
        self.assertEqual(covar.base_kernel.ard_num_dims, 10)
        self.assertEqual(
            covar.base_kernel.lengthscale_prior.rate, 3.0  # pyre-ignore[16]
        )
        self.assertEqual(
            covar.base_kernel.lengthscale_prior.concentration, 6.0  # pyre-ignore[16]
        )
        self.assertEqual(covar.outputscale_prior.rate, 0.15)  # pyre-ignore[16]
        self.assertEqual(covar.outputscale_prior.concentration, 2.0)  # pyre-ignore[16]
        self.assertEqual(covar.base_kernel.batch_shape[0], 2)
