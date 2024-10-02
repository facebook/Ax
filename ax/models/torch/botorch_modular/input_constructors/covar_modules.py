#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.models import MultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import Prior


covar_module_argparse = Dispatcher(
    name="covar_module_argparse", encoder=_argparse_type_encoder
)


@covar_module_argparse.register(Kernel)
def _covar_module_argparse_base(
    covar_module_class: type[Kernel],
    botorch_model_class: type[Model] | None = None,
    dataset: SupervisedDataset | None = None,
    covar_module_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Extract the covar module kwargs form the given arguments.

    NOTE: Since `covar_module_options` is how the user would typically pass in these
    options, it takes precedence over other arguments. E.g., if both `ard_num_dims`
    and `covar_module_options["ard_num_dims"]` are provided, this will use
    `ard_num_dims` from `covar_module_options`.

    Args:
        covar_module_class: Covariance module class.
        botorch_model_class: ``Model`` class to be used as the underlying
            BoTorch model.
        dataset: Dataset containing feature matrix and the response.
        covar_module_options: An optional dictionary of covariance module options.
            This may include overrides for the above options. For example, when
            covar_module_class is MaternKernel this dictionary might include
            {
                "nu": 2.5, # the smoothness parameter
                "ard_num_dims": 3, # the num. of lengthscales per input dimension
                "batch_shape": torch.Size([2]), # the num. of lengthscales per batch,
                    # e.g., metric
                "lengthscale_prior": GammaPrior(6.0, 3.0), # prior for the lengthscale
                    # parameter
            }
            See `gpytorch/kernels/matern_kernel.py` for more options.

    Returns:
        A dictionary with covar module kwargs.
    """
    covar_module_options = covar_module_options or {}
    return {**kwargs, **covar_module_options}


@covar_module_argparse.register(ScaleMaternKernel)
def _covar_module_argparse_scale_matern(
    covar_module_class: type[ScaleMaternKernel],
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    ard_num_dims: int | _DefaultType = DEFAULT,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
    lengthscale_prior: Prior | None = None,
    outputscale_prior: Prior | None = None,
    covar_module_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract the base covar module kwargs form the given arguments.

    NOTE: This setup does not allow for setting multi-dimensional priors,
        with different priors over lengthscales.

    Args:
        covar_module_class: Covariance module class.
        botorch_model_class: Model class to be used as the underlying
            BoTorch model.
        dataset: Dataset containing feature matrix and the response.
        ard_num_dims: Number of lengthscales per feature.
        batch_shape: The number of lengthscales per batch.
        lengthscale_prior: Lenthscale prior.
        outputscale_prior: Outputscale prior.
        covar_module_options: Covariance module kwargs.

    Returns:
        A dictionary with covar module kwargs.
    """

    if issubclass(botorch_model_class, MultiTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = dataset.X.shape[-1] - 1

        if batch_shape is DEFAULT:
            batch_shape = torch.Size([])

    if issubclass(botorch_model_class, SingleTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = dataset.X.shape[-1]

        if (batch_shape is DEFAULT) and (dataset.Y.shape[-1:] == torch.Size([1])):
            batch_shape = torch.Size([])
        elif batch_shape is DEFAULT:
            batch_shape = dataset.Y.shape[-1:]

    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        dataset=dataset,
        botorch_model_class=botorch_model_class,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        batch_shape=batch_shape,
        covar_module_options=covar_module_options,
        **kwargs,
    )
