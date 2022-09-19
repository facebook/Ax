# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


def get_KXX_inv(gp: Model) -> Tensor:
    r"""Get the inverse matrix of K(X,X).
    Args:
        gp: Botorch model.
    Returns:
        The inverse of K(X,X).
    """
    L_inv_upper = gp.prediction_strategy.covar_cache.detach()  # pyre-ignore
    return L_inv_upper @ L_inv_upper.transpose(0, 1)


def get_KxX_dx(gp: Model, x: Tensor, kernel_type: str = "rbf") -> Tensor:
    """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.
    Args:
        gp: Botorch model.
        x: (n x D) Test points.
        kernel_type: Takes "rbf" or "matern_l1" or "matern_l2"
    Returns:
        Tensor (n x D) The derivative of the kernel K(x,X) w.r.t. x.
    """
    X = gp.train_inputs[0]  # pyre-ignore
    D = X.shape[1]
    N = X.shape[0]
    n = x.shape[0]
    lengthscale = gp.covar_module.base_kernel.lengthscale.detach()  # pyre-ignore
    if kernel_type == "rbf":
        K_xX = gp.covar_module(x, X).evaluate()  # pyre-ignore
        part1 = -torch.eye(D, device=x.device, dtype=x.dtype) / lengthscale**2
        part2 = x.view(n, 1, D) - X.view(1, N, D)
        return part1 @ (part2 * K_xX.view(n, N, 1)).transpose(1, 2)
    # Else, we have a Matern kernel, either L1 or L2
    mean = x.reshape(-1, x.size(-1)).mean(0)[(None,) * (x.dim() - 1)]
    x1_ = (x - mean).div(lengthscale)
    x2_ = (X - mean).div(lengthscale)
    matern_norml2 = kernel_type == "matern_l2"
    distance = gp.covar_module.covar_dist(  # pyre-ignore
        x1_, x2_, square_dist=matern_norml2
    )
    exp_component = torch.exp(-math.sqrt(5.0) * distance)  # pyre-ignore
    constant_component = (-5.0 / 3.0) * distance - (5.0 * math.sqrt(5.0) / 3.0) * (
        distance**2
    )
    sigma_f = gp.covar_module.outputscale.detach()  # pyre-ignore
    if matern_norml2:
        part1 = torch.eye(D, device=lengthscale.device) / lengthscale**2
        part2 = 2 * (x.view(n, 1, D) - X.view(1, N, D))
    else:
        part1 = torch.eye(D, device=lengthscale.device) / lengthscale
        part2 = (x1_.view(n, 1, D) - x2_.view(1, N, D)) / distance.unsqueeze(2)
    total_k = sigma_f * constant_component * exp_component
    total = part1 @ (part2 * total_k.view(n, N, 1)).transpose(1, 2)
    return total


def get_Kxx_dx2(gp: Model, kernel_type: str = "rbf") -> Tensor:
    r"""Computes the analytic second derivative of the kernel w.r.t. the training data
    Args:
        gp: Botorch model.
        kernel_type: Takes "rbf" or "matern_l1" or "matern_l2"
    Returns:
        Tensor (n x D x D) The second derivative of the kernel w.r.t. the training data.
    """
    X = gp.train_inputs[0]  # pyre-ignore
    D = X.shape[1]
    lengthscale = gp.covar_module.base_kernel.lengthscale.detach()  # pyre-ignore
    if kernel_type == "rbf":
        sigma_f = gp.covar_module.outputscale.detach()  # pyre-ignore
        return (torch.eye(D, device=lengthscale.device) / lengthscale**2) * sigma_f
    if kernel_type == "matern_l2":
        return torch.zeros(D, D, device=lengthscale.device)
    warnings.warn("second derivative of Matern undefined when x1==x2")
    return torch.eye(D, device=lengthscale.device) * 1e10


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
        kernel_type: Takes "rbf" or "matern_l1" or "matern_l2"
    Returns:
        A Botorch Posterior.
    """
    if gp.prediction_strategy is None:
        gp.posterior(x)  # Call this to update prediction strategy of GPyTorch.
    if kernel_type not in ["rbf", "matern_l1", "matern_l2"]:
        raise ValueError("only matern and rbf kernels are supported")
    K_xX_dx = get_KxX_dx(gp, x, kernel_type=kernel_type)
    Kxx_dx2 = get_Kxx_dx2(gp, kernel_type=kernel_type)
    mean_d = K_xX_dx @ get_KXX_inv(gp) @ gp.train_targets
    variance_d = Kxx_dx2 - K_xX_dx @ get_KXX_inv(gp) @ K_xX_dx.transpose(1, 2)
    variance_d = variance_d.clamp_min(1e-9)
    try:
        return MultivariateNormal(mean_d, variance_d)
    except RuntimeError:
        variance_d_diag = torch.diagonal(variance_d, offset=0, dim1=1, dim2=2)
        variance_d_new = torch.diag_embed(variance_d_diag)
        return MultivariateNormal(mean_d, variance_d_new)
