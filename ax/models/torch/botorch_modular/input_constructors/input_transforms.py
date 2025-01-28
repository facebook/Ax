#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.utils import normalize_indices
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.models.transforms.input import (
    InputPerturbation,
    InputTransform,
    Normalize,
    Warp,
)
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.dispatcher import Dispatcher


input_transform_argparse = Dispatcher(
    name="input_transform_argparse", encoder=_argparse_type_encoder
)


def _set_default_bounds(
    search_space_digest: SearchSpaceDigest,
    input_transform_options: dict[str, Any],
    d: int,
    torch_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> None:
    """Set default bounds in input_transform_options, in-place.

    Args:
        search_space_digest: Search space digest.
        input_transform_options: Input transform kwargs.
        d: The dimension of the input space.
        torch_device: The device on which the input transform will be used.
        torch_dtype: The dtype on which the input transform will be used.
    """
    bounds = torch.as_tensor(
        search_space_digest.bounds,
        dtype=torch_dtype,
        device=torch_device,
    ).T

    if (
        ("bounds" not in input_transform_options)
        and (bounds.shape[-1] < d)
        and (len(search_space_digest.task_features) == 0)
    ):
        raise NotImplementedError(
            "Normalization bounds should be specified explicitly if there"
            " are task features outside the search space."
        )

    input_transform_options.setdefault("bounds", bounds)


@input_transform_argparse.register(InputTransform)
def _input_transform_argparse_base(
    input_transform_class: type[InputTransform],
    dataset: SupervisedDataset | None = None,
    search_space_digest: SearchSpaceDigest | None = None,
    input_transform_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract the input transform kwargs from the given arguments.

    Args:
        input_transform_class: Input transform class.
        dataset: Dataset containing feature matrix and the response.
        search_space_digest: Search space digest.
        input_transform_options: An optional dictionary of input transform options.
            This may include overrides for the above options. For example, when
            `input_transform_class` is Normalize this dictionary might include
            {
                "d": 2, # the dimension of the input space
            }
            See `botorch.models.transforms.input.py` for more options.

    Returns:
        A dictionary with input transform kwargs.
    """
    return input_transform_options or {}


@input_transform_argparse.register(Warp)
def _input_transform_argparse_warp(
    input_transform_class: type[Warp],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    input_transform_options: dict[str, Any] | None = None,
    torch_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Extract the base input transform kwargs form the given arguments.

    Args:
        input_transform_class: Input transform class.
        dataset: Dataset containing feature matrix and the response.
        search_space_digest: Search space digest.
        input_transform_options: Input transform kwargs.
        torch_device: The device on which the input transform will be used.
        torch_dtype: The dtype on which the input transform will be used.

    Returns:
        A dictionary with input transform kwargs.
    """
    input_transform_options = input_transform_options or {}
    d = len(dataset.feature_names)
    indices = list(range(d))
    task_features = normalize_indices(search_space_digest.task_features, d=d)

    for task_feature in sorted(task_features, reverse=True):
        del indices[task_feature]

    input_transform_options.setdefault("d", d)
    input_transform_options.setdefault("indices", indices)
    _set_default_bounds(
        search_space_digest=search_space_digest,
        input_transform_options=input_transform_options,
        d=d,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )
    return input_transform_options


@input_transform_argparse.register(Normalize)
def _input_transform_argparse_normalize(
    input_transform_class: type[Normalize],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    input_transform_options: dict[str, Any] | None = None,
    torch_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """
    Extract the base input transform kwargs form the given arguments.
    NOTE: This input constructor doesn't support the case when there are
    task features that are not part of search space digest. In that case,
    the bounds need to be specified manually as a part of input_transform_options.

    Args:
        input_transform_class: Input transform class.
        dataset: Dataset containing feature matrix and the response.
        search_space_digest: Search space digest.
        input_transform_options: Input transform kwargs.
        torch_device: The device on which the input transform will be used.
        torch_dtype: The dtype on which the input transform will be used.

    Returns:
        A dictionary with input transform kwargs.
    """
    input_transform_options = input_transform_options or {}

    d = input_transform_options.get("d", len(search_space_digest.feature_names))
    input_transform_options["d"] = d

    indices = list(range(d))
    # having indices set to None means that we don't remove task features
    if ("indices" in input_transform_options) and (
        input_transform_options["indices"] is None
    ):
        input_transform_options["indices"] = indices
    else:
        task_features = normalize_indices(search_space_digest.task_features, d=d)
        for task_feature in sorted(task_features, reverse=True):
            del indices[task_feature]

    if ("indices" in input_transform_options) or (len(indices) < d):
        input_transform_options.setdefault("indices", indices)

    _set_default_bounds(
        search_space_digest=search_space_digest,
        input_transform_options=input_transform_options,
        d=d,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )

    return input_transform_options


@input_transform_argparse.register(InputPerturbation)
def _input_transform_argparse_input_perturbation(
    input_transform_class: type[InputPerturbation],
    search_space_digest: SearchSpaceDigest,
    dataset: SupervisedDataset | None = None,
    input_transform_options: dict[str, Any] | None = None,
    torch_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Extract the base input transform kwargs form the given arguments.

    Args:
        input_transform_class: Input transform class.
        dataset: Dataset containing feature matrix and the response.
        search_space_digest: Search space digest.
        input_transform_options: Input transform kwargs.

    Returns:
        A dictionary with input transform kwargs.
    """

    input_transform_options = input_transform_options or {}

    robust_digest = search_space_digest.robust_digest

    if robust_digest is None:
        raise ValueError("Robust search space digest must be provided.")

    if len(robust_digest.environmental_variables) > 0:
        # TODO[T131759269]: support env variables.
        raise NotImplementedError(
            "Environmental variable support is not yet implemented."
        )
    if robust_digest.sample_param_perturbations is None:
        raise ValueError("Robust digest needs to sample parameter perturbations.")

    samples = torch.as_tensor(
        robust_digest.sample_param_perturbations(),
        dtype=torch_dtype,
        device=torch_device,
    )

    input_transform_options.setdefault("perturbation_set", samples)

    input_transform_options.setdefault("multiplicative", robust_digest.multiplicative)

    return input_transform_options
