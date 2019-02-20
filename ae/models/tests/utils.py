#!/usr/bin/env python3

import torch


def _get_torch_test_data(dtype=torch.float, cuda=False, noiseless=False):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=dtype, device=device)]
    Ys = [torch.tensor([[3.0], [4.0]], dtype=dtype, device=device)]
    Yvars = [torch.tensor([[0.0], [2.0]], dtype=dtype, device=device)]
    if noiseless:
        Yvars[0].fill_(0.0)
    bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
    task_features = []
    feature_names = ["x1", "x2", "x3"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names


def _get_model_test_state_dict_noiseless(dtype=torch.float, cuda=False):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    tkwargs = {"device": device, "dtype": dtype}
    cm = "covar_module."
    np = "likelihood.noise_covar.noise_prior"
    state_dict = {
        "likelihood.noise_covar.raw_noise": torch.tensor([[1.0]], **tkwargs),
        "mean_module.constant": torch.tensor([[2.0]], **tkwargs),
        f"{cm}raw_outputscale": torch.tensor([1.0], **tkwargs),
        f"{cm}base_kernel.raw_lengthscale": torch.tensor(
            [[[1.0, 2.0, 3.0]]], **tkwargs
        ),
        f"{cm}base_kernel.lengthscale_prior.concentration": torch.tensor(
            1.0, **tkwargs
        ),
        f"{cm}base_kernel.lengthscale_prior.rate": torch.tensor(2.0, **tkwargs),
        f"{cm}outputscale_prior.concentration": torch.tensor(3.0, **tkwargs),
        f"{cm}outputscale_prior.rate": torch.tensor(4.0, **tkwargs),
        f"{np}.concentration": torch.tensor(0.1, **tkwargs),
        f"{np}.rate": torch.tensor(0.01, **tkwargs),
    }
    return state_dict


def _get_model_test_state_dict(dtype=torch.float, cuda=False):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    tkwargs = {"device": device, "dtype": dtype}

    nm = "likelihood.noise_covar.noise_model"
    np = "likelihood.noise_covar.noise_prior"
    bk = "covar_module.base_kernel"
    state_dict = {
        f"{nm}.likelihood.noise_covar.raw_noise": torch.tensor([[0.0]], **tkwargs),
        f"{nm}.{np}.a": torch.tensor([-3.0], **tkwargs),
        f"{nm}.{np}.b": torch.tensor([5.0], **tkwargs),
        f"{nm}.{np}.sigma": torch.tensor([0.25], **tkwargs),
        f"{nm}.{np}.tails.loc": torch.tensor([0.0], **tkwargs),
        f"{nm}.{np}.tails.scale": torch.tensor([0.25], **tkwargs),
        f"{nm}.mean_module.constant": torch.tensor([[-1.0]], **tkwargs),
        f"{nm}.covar_module.raw_outputscale": torch.tensor([0.05], **tkwargs),
        f"{nm}.{bk}.raw_lengthscale": torch.tensor([[[-1.0, -2.0, -3.0]]], **tkwargs),
        f"{nm}.{bk}.lengthscale_prior.concentration": torch.tensor(2.0, **tkwargs),
        f"{nm}.{bk}.lengthscale_prior.rate": torch.tensor(5.0, **tkwargs),
        f"{nm}.covar_module.outputscale_prior.concentration": torch.tensor(
            1.1, **tkwargs
        ),
        f"{nm}.covar_module.outputscale_prior.rate": torch.tensor(0.05, **tkwargs),
        f"mean_module.constant": torch.tensor([[2.0]], **tkwargs),
        f"covar_module.raw_outputscale": torch.tensor([0.4], **tkwargs),
        f"{bk}.raw_lengthscale": torch.tensor([[[-1.0, -2.0, -3.0]]], **tkwargs),
        f"{bk}.lengthscale_prior.concentration": torch.tensor(2.0, **tkwargs),
        f"{bk}.lengthscale_prior.rate": torch.tensor(5.0, **tkwargs),
        f"covar_module.outputscale_prior.concentration": torch.tensor(1.1, **tkwargs),
        f"covar_module.outputscale_prior.rate": torch.tensor(0.05, **tkwargs),
    }
    return state_dict
