#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional

import torch
from ax.models.torch.botorch_defaults import _get_model
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelList
from botorch.models.deterministic import GenericDeterministicModel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from torch import Tensor


def get_and_fit_model_list_det(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    det_metric_names: List[str],
    det_metric_funcs: Dict[str, Callable[[Tensor], Tensor]],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    **kwargs: Any,
) -> ModelList:
    r"""Instantiates and fits a botorch ModelList using the given data.

    Args:
        Xs: List of X data, one tensor per outcome.
        Ys: List of Y data, one tensor per outcome.
        Yvars: List of observed variance of Ys.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.
        metric_names: Names of each outcome Y in Ys.
        det_metric_names: Names of the deterministic outcomes
        det_metric_funcs: Dict of deterministic metric function callables
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        refit_model: Flag for refitting model.

    Returns:
        A fitted ModelListGPyTorchModel.
    """

    if len(fidelity_features) > 0 or len(task_features) > 0:
        raise NotImplementedError(
            "Currently do not support fidelity_features or task_features!"
        )
    if any(m not in metric_names for m in det_metric_names):
        raise ValueError("All deterministic metric names must be objective names.")

    models = []
    for i, metric in enumerate(metric_names):
        if metric in det_metric_names:
            models.append(GenericDeterministicModel(det_metric_funcs[metric]))
        else:
            # use single task GP for each metric except for the deterministic metrics
            models.append(
                _get_model(
                    X=Xs[i],
                    Y=Ys[i],
                    Yvar=Yvars[i],
                    use_input_warping=use_input_warping,
                    **kwargs,
                )
            )

    model = ModelList(*models)
    model.to(Xs[0])

    if state_dict is not None:
        model.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        if use_loocv_pseudo_likelihood:
            mll_cls = LeaveOneOutPseudoLikelihood
        else:
            mll_cls = ExactMarginalLogLikelihood
        for metric, single_model in zip(metric_names, model.models):
            # No GP fitting for the deterministic metrics
            if metric not in det_metric_names:
                mll = mll_cls(single_model.likelihood, single_model)
                mll = fit_gpytorch_mll(mll, bounds=bounds)
    return model


def L1_norm_func(X: Tensor, init_point: Tensor) -> Tensor:
    r"""L1_norm takes in a a `batch_shape x n x d`-dim input tensor `X`
    to a `batch_shape x n x 1`-dimensional L1 norm tensor. To be used
    for constructing a GenericDeterministicModel.
    """
    return torch.norm((X - init_point), p=1, dim=-1, keepdim=True)
