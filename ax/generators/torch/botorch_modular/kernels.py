#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
References

.. [Papenmeier2025hd]
    L. Papenmeier, M. Poloczek, and L. Nardi. Understanding High-Dimensional Bayesian
        Optimization. Arxiv, 2025.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log, sqrt
from typing import Any

import torch
from ax.exceptions.core import AxError
from botorch.models.utils.gpytorch_modules import SQRT2, SQRT3
from gpytorch.constraints import Interval
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import PeriodicKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import LogNormalPrior, Prior


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


def default_loc_and_scale_for_lognormal_lengthscale_prior(
    ard_num_dims: int | None,
) -> tuple[float, float]:
    """Get the default location and scale for the lengthscale prior.

    Args:
        ard_num_dims: The number of ARD dimensions.

    Returns:
        A tuple of the location and scale for the lengthscale prior.
    """
    ard_num_dims = ard_num_dims or 1
    loc = SQRT2 + log(ard_num_dims) * 0.5
    scale = SQRT3
    return loc, scale


def get_lengthscale_prior_and_initial_value(
    ard_num_dims: int | None, mle: bool
) -> tuple[LogNormalPrior | None, float]:
    if ard_num_dims is None:
        ard_num_dims = 1
    if mle:
        # initial value comes from [Papenmeier2025hd]_.
        return None, sqrt(ard_num_dims) / 10
    loc, scale = default_loc_and_scale_for_lognormal_lengthscale_prior(
        ard_num_dims=ard_num_dims
    )
    lengthscale_prior = LogNormalPrior(loc=loc, scale=scale)
    return lengthscale_prior, lengthscale_prior.mode.item()


class DefaultRBFKernel(RBFKernel):
    """A simple wrapper around RBF kernel that integrates the dim-scaled
    prior commonly used by default in BoTorch models, if mle=False.

    This supports easy configuration of the kernel with options like
    `active_dims`, while avoiding the need to explicitly specify the
    prior for the RBF kernel.
    """

    def __init__(
        self,
        ard_num_dims: int | None,
        active_dims: Sequence[int] | None = None,
        batch_shape: torch.Size | None = None,
        mle: bool = False,
        lengthscale_prior: LogNormalPrior | None = None,
    ) -> None:
        """Initialize Matern kernel with dimension-scaling prior or MLE.

        Args:
            ard_num_dims: The number of ARD dimensions. None signifies that an
                isotropic kernel should be used
            active_dims: The active input dimensions.
            batch_shape: The batch shape for the kernel.
            mle: A boolean indicating whether to use MLE (no priors) or a dimension
                scaling prior.
            lengthscale_prior: The lengthscale prior. If None, a default prior is used.
        """
        if lengthscale_prior is not None:
            initial_value = lengthscale_prior.mode[0].item()
        else:
            lengthscale_prior, initial_value = get_lengthscale_prior_and_initial_value(
                ard_num_dims=ard_num_dims, mle=mle
            )
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=initial_value
            ),
            # pyre-ignore[6] GPyTorch type is unnecessarily restrictive.
            active_dims=active_dims,
        )


class DefaultMaternKernel(MaternKernel):
    """A simple wrapper arounda Matern kernel that integrates the dim-scaled
    prior commonly used by default in BoTorch models (if mle is false).

    This supports easy configuration of the kernel with options like
    `active_dims`, while avoiding the need to explicitly specify the
    prior for the RBF kernel.
    """

    def __init__(
        self,
        ard_num_dims: int | None,
        active_dims: Sequence[int] | None = None,
        batch_shape: torch.Size | None = None,
        nu: float = 2.5,
        mle: bool = False,
        lengthscale_prior: LogNormalPrior | None = None,
    ) -> None:
        if lengthscale_prior is not None:
            initial_value = lengthscale_prior.mode[0].item()
        else:
            lengthscale_prior, initial_value = get_lengthscale_prior_and_initial_value(
                ard_num_dims=ard_num_dims, mle=mle
            )
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=initial_value
            ),
            active_dims=active_dims,
            nu=nu,
        )
