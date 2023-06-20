#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import Prior


class ScaleMaternKernel(ScaleKernel):
    def __init__(
        self,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        lengthscale_prior: Optional[Prior] = None,
        outputscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        outputscale_constraint: Optional[Interval] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            ard_num_dims: The number of lengthscales.
            batch_shape: The batch shape.
            lengthscale_prior: The prior over the lengthscale parameter.
            outputscale_prior: The prior over the scaling parameter.
            lengthscale_constraint: Optionally provide a lengthscale constraint.
            outputscale_constraint: Optionally provide a output scale constraint.

        Returns: None
        """
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            lengthscale_constraint=lengthscale_constraint,
            lengthscale_prior=lengthscale_prior,
            batch_shape=batch_shape,
        )
        super().__init__(
            base_kernel=base_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
            **kwargs,
        )
