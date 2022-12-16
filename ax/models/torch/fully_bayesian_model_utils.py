# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import pyro  # @manual=fbsource//third-party/pypi/pyro-ppl:pyro-ppl
import torch
from ax.models.torch.botorch_defaults import _get_model
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch import Tensor


def _get_rbf_kernel(num_samples: int, dim: int) -> ScaleKernel:
    return ScaleKernel(
        base_kernel=RBFKernel(ard_num_dims=dim, batch_shape=torch.Size([num_samples])),
        batch_shape=torch.Size([num_samples]),
    )


def _get_single_task_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "matern",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )

    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def pyro_sample_outputscale(
    concentration: float = 2.0,
    rate: float = 0.15,
    **tkwargs: Any,
) -> Tensor:

    return pyro.sample(
        "outputscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`
        pyro.distributions.Gamma(
            torch.tensor(concentration, **tkwargs),
            torch.tensor(rate, **tkwargs),
        ),
    )


def pyro_sample_mean(**tkwargs: Any) -> Tensor:

    return pyro.sample(
        "mean",
        # pyre-fixme[16]: Module `distributions` has no attribute `Normal`.
        pyro.distributions.Normal(
            torch.tensor(0.0, **tkwargs),
            torch.tensor(1.0, **tkwargs),
        ),
    )


def pyro_sample_noise(**tkwargs: Any) -> Tensor:

    # this prefers small noise but has heavy tails
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.Gamma(
            torch.tensor(0.9, **tkwargs),
            torch.tensor(10.0, **tkwargs),
        ),
    )


def pyro_sample_saas_lengthscales(
    dim: int,
    alpha: float = 0.1,
    **tkwargs: Any,
) -> Tensor:

    tausq = pyro.sample(
        "kernel_tausq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
    )
    inv_length_sq = pyro.sample(
        "_kernel_inv_length_sq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
    )
    inv_length_sq = pyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
    lengthscale = pyro.deterministic(
        "lengthscale",
        (1.0 / inv_length_sq).sqrt(),  # pyre-ignore [16]
    )
    return lengthscale


def pyro_sample_input_warping(
    dim: int,
    **tkwargs: Any,
) -> Tuple[Tensor, Tensor]:

    c0 = pyro.sample(
        "c0",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    c1 = pyro.sample(
        "c1",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    return c0, c1


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
def load_mcmc_samples_to_model(model: GPyTorchModel, mcmc_samples: Dict) -> None:
    """Load MCMC samples into GPyTorchModel."""
    if "noise" in mcmc_samples:
        model.likelihood.noise_covar.noise = (
            mcmc_samples["noise"]
            .detach()
            .clone()
            .view(model.likelihood.noise_covar.noise.shape)
            .clamp_min(MIN_INFERRED_NOISE_LEVEL)
        )
    model.covar_module.base_kernel.lengthscale = (
        mcmc_samples["lengthscale"]
        .detach()
        .clone()
        .view(model.covar_module.base_kernel.lengthscale.shape)  # pyre-ignore
    )
    model.covar_module.outputscale = (  # pyre-ignore
        mcmc_samples["outputscale"]
        .detach()
        .clone()
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `outputscale`.
        .view(model.covar_module.outputscale.shape)
    )
    model.mean_module.constant.data = (
        mcmc_samples["mean"]
        .detach()
        .clone()
        .view(model.mean_module.constant.shape)  # pyre-ignore
    )
    if "c0" in mcmc_samples:
        model.input_transform._set_concentration(  # pyre-ignore
            i=0,
            value=mcmc_samples["c0"]
            .detach()
            .clone()
            .view(model.input_transform.concentration0.shape),  # pyre-ignore
        )
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `_set_concentration`.
        model.input_transform._set_concentration(
            i=1,
            value=mcmc_samples["c1"]
            .detach()
            .clone()
            .view(model.input_transform.concentration1.shape),  # pyre-ignore
        )
