#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger
from math import log
from typing import Any

import torch
from ax.exceptions.core import UserInputError
from ax.generators.torch.botorch_modular.kernels import (
    DefaultMaternKernel,
    DefaultRBFKernel,
    ScaleMaternKernel,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.models import MultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.utils.gpytorch_modules import SQRT2, SQRT3
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.transforms import normalize_indices
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.priors.torch_priors import LogNormalPrior, Prior
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)

covar_module_argparse = Dispatcher(
    name="covar_module_argparse", encoder=_argparse_type_encoder
)


@covar_module_argparse.register(Kernel)
def _covar_module_argparse_base(
    covar_module_class: type[Kernel],
    botorch_model_class: type[Model] | None = None,
    dataset: SupervisedDataset | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Extract the covar module kwargs form the given arguments.

    Args:
        covar_module_class: Covariance module class.
        botorch_model_class: ``Model`` class to be used as the underlying
            BoTorch model.
        dataset: Dataset containing feature matrix and the response.

    Returns:
        A dictionary with covar module kwargs.
    """
    kwargs.pop("remove_task_features", None)
    return {**kwargs}


@covar_module_argparse.register(ScaleMaternKernel)
def _covar_module_argparse_scale_matern(
    covar_module_class: type[ScaleMaternKernel],
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    ard_num_dims: int | _DefaultType = DEFAULT,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
    lengthscale_prior: Prior | None = None,
    outputscale_prior: Prior | None = None,
    remove_task_features: bool = False,
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
        remove_task_features: A boolean indicating whether to remove the task
            features (e.g. when using a SingleTask model on a MultiTaskDataset).

    Returns:
        A dictionary with covar module kwargs.
    """
    ard_num_dims, batch_shape, active_dims = _get_default_ard_num_dims_and_batch_shape(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        botorch_model_class=botorch_model_class,
        dataset=dataset,
        remove_task_features=remove_task_features,
    )
    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        dataset=dataset,
        botorch_model_class=botorch_model_class,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        batch_shape=batch_shape,
        active_dims=active_dims,
        **kwargs,
    )


@covar_module_argparse.register(LinearKernel)
def _covar_module_argparse_linear(
    covar_module_class: type[LinearKernel],
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    ard_num_dims: int | _DefaultType = DEFAULT,
    active_dims: Sequence[int] | None = None,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
    variance_prior: Prior | None = None,
    remove_task_features: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract the covar module kwargs for a LinearKernle from the given arguments.

    Args:
        covar_module_class: Covariance module class.
        botorch_model_class: Model class to be used as the underlying
            BoTorch model.
        dataset: Dataset containing feature matrix and the response.
        ard_num_dims: Number of lengthscales per feature.
        active_dims: The active dimensions of the kernel.
        batch_shape: The number of lengthscales per batch.
        variance_prior: Variance prior.
        remove_task_features: A boolean indicating whether to exclude the task
            feature in the kernel.

    Returns:
        A dictionary with covar module kwargs.
    """
    ard_num_dims, batch_shape, active_dims = _get_default_ard_num_dims_and_batch_shape(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        botorch_model_class=botorch_model_class,
        dataset=dataset,
        remove_task_features=remove_task_features,
        active_dims=active_dims,
    )
    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        dataset=dataset,
        botorch_model_class=botorch_model_class,
        ard_num_dims=ard_num_dims,
        variance_prior=variance_prior,
        batch_shape=batch_shape,
        active_dims=active_dims,
        **kwargs,
    )


@covar_module_argparse.register((DefaultRBFKernel, DefaultMaternKernel))
def _covar_module_argparse_default_rbf(
    covar_module_class: type[DefaultRBFKernel] | type[DefaultMaternKernel],
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    ard_num_dims: int | _DefaultType = DEFAULT,
    inactive_features: Sequence[str] | None = None,
    active_dims: Sequence[int] | None = None,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
    remove_task_features: bool = False,
    lengthscale_prior_dict: dict[str, tuple[float, float]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Constructs inputs for ``DefaultRBFKernel``.

    The key feature of this helper is the support for ``inactive_features`` input,
    which is used to determine the ``active_dims`` input for the kernel by removing
    the dimensions that correspond to the given feature names in the dataset.

    Args:
        lengthscale_prior_dict: A dictionary of lengthscale prior parameters.
            The keys are the parameter names and the values are the loc and scale
            parameters for GPyTorch LogNormal prior for the given parameter.
            For parameters that are not found in this dictionary, the default
            values are used.
            This allow us to specify specific priors for individual parameters,
            while using the default priors for the rest.
    """
    if inactive_features is not None and active_dims is not None:
        raise UserInputError(
            "Only one of `active_dims` or `inactive_features` inputs can be provided. "
            f"Got {active_dims=}, {inactive_features=}."
        )
    d = len(dataset.feature_names)
    if inactive_features is not None:
        all_dims = set(range(d))
        inactive_dims = [dataset.feature_names.index(ft) for ft in inactive_features]
        active_dims = list(all_dims.difference(inactive_dims))
    ard_num_dims, batch_shape, active_dims = _get_default_ard_num_dims_and_batch_shape(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        botorch_model_class=botorch_model_class,
        dataset=dataset,
        remove_task_features=remove_task_features,
        active_dims=active_dims,
    )
    if ard_num_dims is DEFAULT:
        ard_num_dims = len(active_dims) if active_dims is not None else d
    if lengthscale_prior_dict:
        default_loc = SQRT2 + log(assert_is_instance(ard_num_dims, int)) * 0.5
        default_scale = SQRT3
        loc_scale = [
            lengthscale_prior_dict.get(ft, (default_loc, default_scale))
            for ft in dataset.feature_names
        ]
        loc, scale = zip(*loc_scale)
        lengthscale_prior = LogNormalPrior(
            loc=torch.tensor(loc), scale=torch.tensor(scale)
        )
    else:
        lengthscale_prior = None
    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        dataset=dataset,
        botorch_model_class=botorch_model_class,
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        active_dims=active_dims,
        lengthscale_prior=lengthscale_prior,
        **kwargs,
    )


def _get_default_batch_shape(
    dataset: SupervisedDataset,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
) -> torch.Size:
    if (batch_shape is DEFAULT) and (dataset.Y.shape[-1:] == torch.Size([1])):
        return torch.Size([])
    if batch_shape is DEFAULT:
        return dataset.Y.shape[-1:]
    return assert_is_instance(batch_shape, torch.Size)


def _get_default_ard_num_dims_and_batch_shape(
    ard_num_dims: int | _DefaultType,
    batch_shape: torch.Size | _DefaultType,
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    remove_task_features: bool,
    active_dims: Sequence[int] | None = None,
) -> tuple[int | _DefaultType, torch.Size | _DefaultType, list[int] | None]:
    """Helper method for constructing shared inputs across kernels.

    Args:
        ard_num_dims: The number of lengthscales.
        batch_shape: The batch shape of the kernel.
        botorch_model_class: The BoTorch model class.
        dataset: The dataset.
        remove_task_features: A boolean indicating whether to exclude the task
            feature in the kernel.
        active_dims: The active dimensions of the kernel.

    Returns:
        A tuple of (ard_num_dims, batch_shape, active_dims).
    """
    if active_dims is not None:
        active_dims = none_throws(
            normalize_indices(indices=list(active_dims), d=len(dataset.feature_names))
        )
        num_active_dims = len(active_dims)
    else:
        num_active_dims = dataset.X.shape[-1]
    if issubclass(botorch_model_class, MultiTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = num_active_dims - 1
        if batch_shape is DEFAULT:
            batch_shape = torch.Size([])
    if issubclass(botorch_model_class, SingleTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = num_active_dims
            if remove_task_features:
                if isinstance(dataset, MultiTaskDataset):
                    if active_dims is None:
                        active_dims = list(range(dataset.X.shape[-1]))
                    logger.debug(
                        "Excluding task feature from covar_module.", stacklevel=6
                    )
                    task_feature_index = (
                        -1
                        if dataset.task_feature_index is None
                        else dataset.task_feature_index
                    )
                    normalized_task_idx = none_throws(
                        normalize_indices(
                            indices=[task_feature_index],
                            d=dataset.X.shape[-1],
                        )
                    )[0]
                    ard_num_dims -= 1
                    active_dims = [i for i in active_dims if i != normalized_task_idx]
        batch_shape = _get_default_batch_shape(dataset=dataset, batch_shape=batch_shape)

    return ard_num_dims, batch_shape, active_dims
