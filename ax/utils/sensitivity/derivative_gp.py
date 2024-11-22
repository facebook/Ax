# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math

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
    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `covar_cache`.
    L_inv_upper = gp.prediction_strategy.covar_cache.detach()
    return L_inv_upper @ L_inv_upper.transpose(0, 1)


def get_KxX_dx(gp: Model, x: Tensor, kernel_type: str = "rbf") -> Tensor:
    """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.
    Args:
        gp: Botorch model.
        x: (n x D) Test points.
        kernel_type: Takes "rbf" or "matern"
    Returns:
        Tensor (n x D) The derivative of the kernel K(x,X) w.r.t. x.
    """
    # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, Any, ...
    X = gp.train_inputs[0]
    D = X.shape[1]
    N = X.shape[0]
    n = x.shape[0]
    if hasattr(gp.covar_module, "outputscale"):
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `base_kernel`.
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach()
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `outputscale`.
        sigma_f = gp.covar_module.outputscale.detach()
    else:
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `lengthscale`.
        lengthscale = gp.covar_module.lengthscale.detach()
        sigma_f = 1.0
    if kernel_type == "rbf":
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        K_xX = gp.covar_module(x, X).evaluate()
        part1 = -torch.eye(D, device=x.device, dtype=x.dtype) / lengthscale**2
        part2 = x.view(n, 1, D) - X.view(1, N, D)
        return part1 @ (part2 * K_xX.view(n, N, 1)).transpose(1, 2)
    # Else, we have a Matern kernel
    mean = x.reshape(-1, x.size(-1)).mean(0)[(None,) * (x.dim() - 1)]
    x1_ = (x - mean).div(lengthscale)
    x2_ = (X - mean).div(lengthscale)
    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `covar_dist`.
    distance = gp.covar_module.covar_dist(x1_, x2_)
    exp_component = torch.exp(-math.sqrt(5.0) * distance)  # pyre-ignore
    constant_component = (-5.0 / 3.0) * distance - (5.0 * math.sqrt(5.0) / 3.0) * (
        distance**2
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
    # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, Any, ...
    X = gp.train_inputs[0]
    D = X.shape[1]
    if hasattr(gp.covar_module, "outputscale"):
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `base_kernel`.
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach()
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `outputscale`.
        sigma_f = gp.covar_module.outputscale.detach()
    else:
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `lengthscale`.
        lengthscale = gp.covar_module.lengthscale.detach()
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
    mean_d = (
        K_xX_dx
        @ get_KXX_inv(gp)
        # pyre-fixme[29]: `Union[(self: TensorBase, other: Union[bool, complex,
        #  float, int, Tensor]) -> Tensor, Tensor, Module]` is not a function.
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        @ (gp.train_targets - gp.mean_module(gp.train_inputs[0]))
    )
    variance_d = Kxx_dx2 - K_xX_dx @ get_KXX_inv(gp) @ K_xX_dx.transpose(1, 2)
    variance_d = variance_d.clamp_min(1e-9)
    try:
        return MultivariateNormal(mean_d, variance_d)
    except RuntimeError:
        variance_d_diag = torch.diagonal(variance_d, offset=0, dim1=1, dim2=2)
        variance_d_new = torch.diag_embed(variance_d_diag)
        return MultivariateNormal(mean_d, variance_d_new)
