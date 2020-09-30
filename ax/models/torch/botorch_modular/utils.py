#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inspect import isclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from ax.core.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.utils.containers import TrainingData
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def choose_model_class(
    Yvars: List[Tensor], task_features: List[int], fidelity_features: List[int]
) -> Type[Model]:
    """Chooses a BoTorch `Model` using the given data (currently just Yvars)
    and its properties (information about task and fidelity features).

    Args:
        Yvars: List of tensors, each representing observation noise for a
            given outcome, where outcomes are in the same order as in Xs.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A BoTorch `Model` class.
    """
    if len(fidelity_features) > 1:
        raise NotImplementedError(
            f"Only a single fidelity feature supported (got: {fidelity_features})."
        )
    if len(task_features) > 1:
        raise NotImplementedError(
            f"Only a single task feature supported (got: {task_features})."
        )
    if task_features and fidelity_features:
        raise NotImplementedError(
            "Multi-task multi-fidelity optimization not yet supported."
        )

    Yvars_cat = torch.cat(Yvars).clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
    is_nan = torch.isnan(Yvars_cat)
    all_nan_Yvar = torch.all(is_nan)
    if torch.any(is_nan) and not all_nan_Yvar:
        raise ValueError(
            "Mix of known and unknown variances indicates valuation function "
            "errors. Variances should all be specified, or none should be."
        )

    # Multi-task cases (when `task_features` specified).
    if task_features and all_nan_Yvar:
        return MultiTaskGP  # Unknown observation noise.
    elif task_features:
        return FixedNoiseMultiTaskGP  # Known observation noise.

    # Single-task multi-fidelity cases.
    if fidelity_features and all_nan_Yvar:
        return SingleTaskMultiFidelityGP  # Unknown observation noise.
    elif fidelity_features:
        return FixedNoiseMultiFidelityGP  # Known observation noise.

    # Single-task single-fidelity cases.
    elif all_nan_Yvar:  # Unknown observation noise.
        return SingleTaskGP
    return FixedNoiseGP  # Known observation noise.


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


def construct_training_data(
    Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor], model_class: Type[Model]
) -> TrainingData:
    """Construct a `TrainingData` object based on sizes of Xs, Ys, and Yvars, and
    the type of model, for which the training data is intended.

    NOTE: All four outputs are organized as lists over outcomes. E.g. if there are two
    outcomes, 'x' and 'y', the Xs are formatted like so: `[Xs_x_ndarray, Xs_y_ndarray]`.
    We specifically do not assume that every point is observed for every outcome.
    This means that the array for each of those outcomes may be different, and in
    particular could have a different length (e.g. if a particular arm was observed
    only for half of the outcomes, it would be present in half of the arrays in the
    list but not the other half.)

    Returns:
        A `TrainingData` object with training data for single outcome or with
        batched multi-output training data if appropriate for given model and if
        all X inputs in Xs are equal.
    """
    if not isclass(model_class):  # pragma: no cover
        raise ValueError(
            f"Expected `Type[Model]`, got: {model_class} "
            f"(type: {type(model_class)})."
        )
    if len(Xs) == len(Ys) == 1:
        # Just one outcome, can use single model.
        return TrainingData(X=Xs[0], Y=Ys[0], Yvar=Yvars[0])
    elif issubclass(model_class, BatchedMultiOutputGPyTorchModel) and all(
        torch.equal(Xs[0], X) for X in Xs[1:]
    ):
        # All Xs are the same and model supports batched multioutput.
        return TrainingData(
            X=Xs[0], Y=torch.cat(Ys, dim=-1), Yvar=torch.cat(Yvars, dim=-1)
        )
    elif model_class is ModelListGP:  # pragma: no cover
        # TODO: This will be case for `ListSurrogate`.
        raise NotImplementedError("`ModelListGP` not yet supported.")
    raise ValueError(f"Unexpected training data format for {model_class}.")


def validate_data_format(
    Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor], metric_names: List[str]
) -> None:
    """Validates that Xs, Ys, Yvars, and metric names all have equal lengths."""
    if len({len(Xs), len(Ys), len(Yvars), len(metric_names)}) > 1:
        raise ValueError(  # pragma: no cover
            "Lengths of Xs, Ys, Yvars, and metric_names must match. Your "
            f"inputs have lengths {len(Xs)}, {len(Ys)}, {len(Yvars)}, and "
            f"{len(metric_names)}, respectively."
        )


def construct_acquisition_and_optimizer_options(
    acqf_options: TConfig, model_gen_options: Optional[TConfig] = None
) -> Tuple[TConfig, TConfig]:
    """Extract acquisition and optimizer options from `model_gen_options`."""
    acq_options = acqf_options.copy()
    opt_options = {}

    if model_gen_options:
        acq_options.update(
            checked_cast(dict, model_gen_options.get(Keys.ACQF_KWARGS, {}))
        )
        # TODO: Add this if all acq. functions accept the `subset_model`
        # kwarg or opt for kwarg filtering.
        # acq_options[SUBSET_MODEL] = model_gen_options.get(SUBSET_MODEL)
        opt_options = checked_cast(
            dict, model_gen_options.get(Keys.OPTIMIZER_KWARGS, {})
        ).copy()
    return acq_options, opt_options
