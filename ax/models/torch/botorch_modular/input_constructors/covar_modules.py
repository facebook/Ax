#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger
from typing import Any

import torch
from ax.exceptions.core import UserInputError
from ax.models.torch.botorch_modular.kernels import DefaultRBFKernel, ScaleMaternKernel
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.models import MultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.transforms import normalize_indices
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import Prior
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
    active_dims = None

    if issubclass(botorch_model_class, MultiTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = dataset.X.shape[-1] - 1

        if batch_shape is DEFAULT:
            batch_shape = torch.Size([])

    if issubclass(botorch_model_class, SingleTaskGP):
        if ard_num_dims is DEFAULT:
            ard_num_dims = dataset.X.shape[-1]
            if remove_task_features:
                if isinstance(dataset, MultiTaskDataset):
                    logger.debug(
                        "Excluding task feature from covar_module.", stacklevel=6
                    )
                    normalized_task_idx = none_throws(
                        normalize_indices(
                            indices=[none_throws(dataset.task_feature_index)],
                            d=dataset.X.shape[-1],
                        )
                    )[0]
                    ard_num_dims -= 1
                    active_dims = [
                        i
                        for i in range(dataset.X.shape[-1])
                        if i != normalized_task_idx
                    ]
        batch_shape = _get_default_batch_shape(dataset=dataset, batch_shape=batch_shape)

    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        botorch_model_class=botorch_model_class,
        dataset=dataset,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        batch_shape=batch_shape,
        active_dims=active_dims,
        **kwargs,
    )


@covar_module_argparse.register(DefaultRBFKernel)
def _covar_module_argparse_default_rbf(
    covar_module_class: type[DefaultRBFKernel],
    botorch_model_class: type[Model],
    dataset: SupervisedDataset,
    ard_num_dims: int | _DefaultType = DEFAULT,
    inactive_features: Sequence[str] | None = None,
    active_dims: Sequence[int] | None = None,
    batch_shape: torch.Size | _DefaultType = DEFAULT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Constructs inputs for ``DefaultRBFKernel``.

    The key feature of this helper is the support for ``inactive_features`` input,
    which is used to determine the ``active_dims`` input for the kernel by removing
    the dimensions that correspond to the given feature names in the dataset.
    """
    if inactive_features is not None and active_dims is not None:
        raise UserInputError(
            "Only one of `active_dims` or `inactive_features` inputs can be provided. "
            f"Got {active_dims=}, {inactive_features=}."
        )
    d = len(dataset.feature_names)
    if active_dims is not None:
        active_dims = none_throws(normalize_indices(indices=list(active_dims), d=d))
    if inactive_features is not None:
        all_dims = set(range(d))
        inactive_dims = [dataset.feature_names.index(ft) for ft in inactive_features]
        active_dims = list(all_dims.difference(inactive_dims))
    batch_shape = _get_default_batch_shape(dataset=dataset, batch_shape=batch_shape)
    if ard_num_dims is DEFAULT:
        ard_num_dims = len(active_dims) if active_dims is not None else d
    return _covar_module_argparse_base(
        covar_module_class=covar_module_class,
        dataset=dataset,
        botorch_model_class=botorch_model_class,
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        active_dims=active_dims,
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
