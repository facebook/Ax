#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
from ax.exceptions.core import AxError
from gpytorch.constraints import Interval
from gpytorch.kernels import PeriodicKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import Prior


class ScaleMaternKernel(ScaleKernel):
    def __init__(
        self,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        lengthscale_prior: Prior | None = None,
        outputscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        outputscale_constraint: Interval | None = None,
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


class TemporalKernel(ScaleKernel):
    """A product kernel of a periodic kernel and a Matern kernel.

    The periodic kernel computes the similarity between temporal
    features such as the time of day.

    The Matern kernel computes the similarity between the tunable
    parameters.
    """

    def __init__(
        self,
        dim: int,
        temporal_features: list[int],
        matern_ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        lengthscale_prior: Prior | None = None,
        temporal_lengthscale_prior: Prior | None = None,
        period_length_prior: Prior | None = None,
        fixed_period_length: float | None = None,
        outputscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        outputscale_constraint: Interval | None = None,
        temporal_lengthscale_constraint: Interval | None = None,
        period_length_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            dim: The input dimension.
            temporal_features: The features to pass to the periodic kernel.
            matern_ard_num_dims: The number of lengthscales. This must be
                equal to the total number of parameters (excluding
                temporal parameters)
            batch_shape: The batch shape.
            lengthscale_prior: The prior over the lengthscale parameters.
            temporal_lengthscale_prior: The prior over the lengthscale parameters
                for the periodic kernel.
            period_length_prior: The prior over the period length.
            fixed_period_length: A fixed period length for the periodic kernel.
                If provided, the period length will not be tuned with the other
                hyperparameters.
            outputscale_prior: The prior over the scaling parameter.
            lengthscale_constraint: Optionally provide a lengthscale constraint.
            outputscale_constraint: Optionally provide a output scale constraint.
            temporal_lengthscale_constraint: Optionally provide a
            lengthscale constraint for the periodic kernel.period_length_constraint:
                Optionally provide a constraint for the period length.

        """
        if len(temporal_features) == 0:
            raise AxError(
                "The temporal kernel should only be used if there "
                "are temporal features."
            )
        if fixed_period_length is not None and (
            period_length_prior is not None or period_length_constraint is not None
        ):
            raise ValueError(
                "If `fixed_period_length` is provided, then `period_length_prior` "
                "and `period_length_constraint` are not used."
            )
        non_temporal_dims = sorted(set(range(dim)) - set(temporal_features))
        matern_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=matern_ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            active_dims=non_temporal_dims,
            batch_shape=batch_shape,
            lengthscale_constraint=lengthscale_constraint,
        )
        periodic_kernel = PeriodicKernel(
            ard_num_dims=len(temporal_features),
            active_dims=temporal_features,
            lengthscale_prior=temporal_lengthscale_prior,
            period_length_prior=period_length_prior,
            lengthscale_constraint=temporal_lengthscale_constraint,
            period_length_constraint=period_length_constraint,
            batch_shape=batch_shape,
        )
        if fixed_period_length is not None:
            periodic_kernel.raw_period_length.requires_grad_(False)
            periodic_kernel.period_length = fixed_period_length
        super().__init__(
            base_kernel=matern_kernel * periodic_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
            **kwargs,
        )
