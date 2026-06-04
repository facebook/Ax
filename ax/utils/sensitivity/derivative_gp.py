# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
from typing import Any

import torch
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.models.exact_gp import ExactGP
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from pyre_extensions import assert_is_instance
from torch import Tensor


def _get_prediction_strategy(gp: Model) -> DefaultPredictionStrategy:
    """Get the prediction strategy from the GP model."""
    return assert_is_instance(
        assert_is_instance(gp, ExactGP).prediction_strategy, DefaultPredictionStrategy
    )


def get_KXX_inv(gp: Model) -> Tensor:
    r"""Get the inverse matrix of K(X,X).
    Args:
        gp: Botorch model.
    Returns:
        The inverse of K(X,X).
    """
    L_inv_upper: Tensor = _get_prediction_strategy(gp).covar_cache.detach()
    return L_inv_upper @ L_inv_upper.transpose(0, 1)


def _get_train_inputs(gp: Model) -> Tensor:
    """Get the first element of gp.train_inputs, typed as Tensor."""
    train_inputs: Any = gp.train_inputs
    return train_inputs[0]


def get_KxX_dx(gp: Model, x: Tensor, kernel_type: str = "rbf") -> Tensor:
    """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.
    Args:
        gp: Botorch model.
        x: (n x D) Test points.
        kernel_type: Takes "rbf" or "matern"
    Returns:
        Tensor (n x D) The derivative of the kernel K(x,X) w.r.t. x.
    """
    X = _get_train_inputs(gp)
    D = X.shape[1]
    N = X.shape[0]
    n = x.shape[0]
    covar_module = assert_is_instance(gp.covar_module, Kernel)
    if isinstance(covar_module, ScaleKernel):
        lengthscale = covar_module.base_kernel.lengthscale.detach()
        sigma_f = covar_module.outputscale.detach()
    else:
        lengthscale = covar_module.lengthscale.detach()
        sigma_f = 1.0
    if kernel_type == "rbf":
        # pyre-fixme[16]: Tensor, linear opearator mix is tricky to fix.
        K_xX = covar_module(x, X).evaluate()
        part1 = -torch.eye(D, device=x.device, dtype=x.dtype) / lengthscale**2
        part2 = x.view(n, 1, D) - X.view(1, N, D)
        return part1 @ (part2 * K_xX.view(n, N, 1)).transpose(1, 2)
    # Else, we have a Matern kernel
    mean = x.reshape(-1, x.size(-1)).mean(0)[(None,) * (x.dim() - 1)]
    x1_ = (x - mean).div(lengthscale)
    x2_ = (X - mean).div(lengthscale)
    distance: Tensor = covar_module.covar_dist(x1_, x2_)
    exp_component = torch.exp(-math.sqrt(5.0) * distance)
    constant_component = (-5.0 / 3.0) * distance - (5.0 * math.sqrt(5.0) / 3.0) * (
        distance.pow(2)
    )
    part1 = torch.eye(D, device=lengthscale.device) / lengthscale
    part2 = (x1_.view(n, 1, D) - x2_.view(1, N, D)) / distance.unsqueeze(2)
    total_k = sigma_f * constant_component * exp_component
    total = part1 @ (part2 * total_k.view(n, N, 1)).transpose(1, 2)
    return total


def get_Kxx_dx2(gp: Model, kernel_type: str = "rbf") -> Tensor:
    r"""Computes the analytic second derivative of the kernel w.r.t. the training data
    Args:
        gp: Botorch model.
        kernel_type: Takes "rbf" or "matern"
    Returns:
        Tensor (n x D x D) The second derivative of the kernel w.r.t. the training data.
    """
    X = _get_train_inputs(gp)
    D = X.shape[1]
    covar_module = assert_is_instance(gp.covar_module, Kernel)
    if isinstance(covar_module, ScaleKernel):
        lengthscale = covar_module.base_kernel.lengthscale.detach()
        sigma_f = covar_module.outputscale.detach()
    else:
        lengthscale = covar_module.lengthscale.detach()
        sigma_f = 1.0
    res = (torch.eye(D, device=lengthscale.device) / lengthscale**2) * sigma_f
    if kernel_type == "rbf":
        return res
    return res * (5 / 3)


def posterior_derivative(
    gp: Model, x: Tensor, kernel_type: str = "rbf"
) -> MultivariateNormal:
    r"""Computes the posterior of the derivative of the GP w.r.t. the given test
    points x.
    This follows the derivation used by GIBO in Sarah Muller, Alexander
    von Rohr, Sebastian Trimpe. "Local policy search with Bayesian optimization",
    Advances in Neural Information Processing Systems 34, NeurIPS 2021.
    Args:
        gp: Botorch model
        x: (n x D) Test points.
        kernel_type: Takes "rbf" or "matern"
    Returns:
        A Botorch Posterior.
    """
    if gp.prediction_strategy is None:
        gp.posterior(x)  # Call this to update prediction strategy of GPyTorch.
    if kernel_type not in ["rbf", "matern"]:
        raise ValueError("only matern and rbf kernels are supported")
    K_xX_dx = get_KxX_dx(gp, x, kernel_type=kernel_type)
    Kxx_dx2 = get_Kxx_dx2(gp, kernel_type=kernel_type)
    train_inputs = _get_train_inputs(gp)
    mean_module: Any = gp.mean_module
    train_targets: Any = gp.train_targets
    mean_d = K_xX_dx @ get_KXX_inv(gp) @ (train_targets - mean_module(train_inputs))
    variance_d = Kxx_dx2 - K_xX_dx @ get_KXX_inv(gp) @ K_xX_dx.transpose(1, 2)
    variance_d = variance_d.clamp_min(1e-9)
    try:
        return MultivariateNormal(mean_d, variance_d)
    except RuntimeError:
        variance_d_diag = torch.diagonal(variance_d, offset=0, dim1=1, dim2=2)
        variance_d_new = torch.diag_embed(variance_d_diag)
        return MultivariateNormal(mean_d, variance_d_new)
