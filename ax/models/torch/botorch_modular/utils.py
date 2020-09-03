#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inspect import isclass
from typing import Dict, List, Optional, Type

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def choose_model_class(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
) -> Type[Model]:
    """Chooses a BoTorch `Model` using the given data.

    Args:
        Xs: List of tensors, each representing observation features for a
            given outcome.
        Ys: List of tensors, each representing observation data for a given
            outcome, where outcomes are in the same order as in Xs.
        Yvars: List of tensors, each representing observation noise for a
            given outcome, where outcomes are in the same order as in Xs.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A BoTorch `Model` class.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support `task_features`!")
    if len(fidelity_features) > 1:
        raise NotImplementedError("Currently support only a single fidelity parameter!")

    # NOTE: We currently do not support `task_features`. This code block will only
    # be relevant once we support `task_features`.
    # if len(task_features) > 1:
    #     raise NotImplementedError(
    #         f"This model only supports 1 task feature (got {task_features})"
    #     )
    # elif len(task_features) == 1:
    #     task_feature = task_features[0]
    # else:
    #     task_feature = None
    task_feature = None

    # NOTE: In the current setup, `task_feature = None` always.
    if task_feature is None:
        Yvars_cat = torch.cat(Yvars).clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
        is_nan = torch.isnan(Yvars_cat)
        any_nan_Yvar = torch.any(is_nan)
        all_nan_Yvar = torch.all(is_nan)
        if any_nan_Yvar and not all_nan_Yvar:
            raise ValueError(
                "Mix of known and unknown variances indicates valuation function "
                "errors. Variances should all be specified, or none should be."
            )
        if len(fidelity_features or []) > 0:
            return SingleTaskMultiFidelityGP
        elif all_nan_Yvar:
            return SingleTaskGP
        return FixedNoiseGP
    # TODO: Replace ValueError with `ModelListGP`.
    # raise ValueError("Unexpected training data format. Cannot choose `Model`.")


def choose_mll_class(
    model_class: Type[Model],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit: bool = True,
) -> Type[MarginalLogLikelihood]:
    r"""Chooses a BoTorch `MarginalLogLikelihood` class using the given `Model` class.

    Args:
        model_class: BoTorch `Model` class.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        refit: Flag for refitting model.

    Returns:
        A `MarginalLogLikelihood` class.
    """
    # NOTE: We currently do not support `ModelListGP`. This code block will only
    # be relevant once we support `ModelListGP`.
    if (state_dict is None or refit) and issubclass(model_class, ModelListGP):
        return SumMarginalLogLikelihood
    return ExactMarginalLogLikelihood


def choose_botorch_acqf_class() -> Type[AcquisitionFunction]:
    r"""Chooses a BoTorch `AcquisitionFunction` class."""
    # NOTE: In the future, this dispatch function could leverage any
    # of the attributes of `BoTorchModel` or kwargs passed to
    # `BoTorchModel.gen` to intelligently select acquisition function.
    return qNoisyExpectedImprovement


def supports_batched_multioutput(model_class: Type[Model]) -> bool:
    """Whether a given BoTorch `Model` class supports batched
    multioutput."""
    if not isclass(model_class):  # pragma: no cover
        raise ValueError(
            f"Expected `Type[Model]`, got: {model_class} "
            f"(type: {type(model_class)})."
        )
    return issubclass(model_class, BatchedMultiOutputGPyTorchModel)


def validate_data_format(
    Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor], metric_names: List[str]
) -> None:
    """Validates that Xs, Ys, Yvars, and metric names all have equal lengths."""
    if len({len(Xs), len(Ys), len(Yvars), len(metric_names)}) > 1:
        raise ValueError(
            "Lengths of Xs, Ys, Yvars, and metric_names must match. Your "
            f"inputs have lengths {len(Xs)}, {len(Ys)}, {len(Yvars)}, and "
            f"{len(metric_names)}, respectively."
        )
