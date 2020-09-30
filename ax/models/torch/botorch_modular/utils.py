#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

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
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.utils.containers import TrainingData
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def use_model_list(Xs: List[Tensor], botorch_model_class: Type[Model]) -> bool:
    if len(Xs) == 1:
        # Just one outcome, can use single model.
        return False
    if issubclass(botorch_model_class, BatchedMultiOutputGPyTorchModel) and all(
        torch.equal(Xs[0], X) for X in Xs[1:]
    ):
        # Single model, batched multi-output case.
        return False
    # If there are multiple Xs and they are not all equal, we
    # use `ListSurrogate` and `ModelListGP`.
    return True


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


def choose_botorch_acqf_class() -> Type[AcquisitionFunction]:
    """Chooses a BoTorch `AcquisitionFunction` class."""
    # NOTE: In the future, this dispatch function could leverage any
    # of the attributes of `BoTorchModel` or kwargs passed to
    # `BoTorchModel.gen` to intelligently select acquisition function.
    return qNoisyExpectedImprovement


def construct_single_training_data(
    Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor]
) -> TrainingData:
    """Construct a `TrainingData` object for a single-outcome model or a batched
    multi-output model. **This function assumes that a single `TrainingData` is
    expected (so if all Xs are equal, it will produce `TrainingData` for a batched
    multi-output model).**

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
    if len(Xs) == len(Ys) == 1:
        # Just one outcome, can use single model.
        return TrainingData(X=Xs[0], Y=Ys[0], Yvar=Yvars[0])
    elif all(torch.equal(Xs[0], X) for X in Xs[1:]):
        if not len(Xs) == len(Ys) == len(Yvars):  # pragma: no cover
            raise ValueError("Xs, Ys, and Yvars must have equal lengths.")
        # All Xs are the same and model supports batched multioutput.
        return TrainingData(
            X=Xs[0], Y=torch.cat(Ys, dim=-1), Yvar=torch.cat(Yvars, dim=-1)
        )
    raise ValueError(
        "Unexpected training data format. Use `construct_training_data_list` if "
        "constructing training data for multiple outcomes (and not using batched "
        "multi-output)."
    )


def construct_training_data_list(
    Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor]
) -> List[TrainingData]:
    """Construct a list of `TrainingData` objects, for use in `ListSurrogate` and
    `ModelListGP`. Each `TrainingData` corresponds to an outcome.

    NOTE: All four outputs are organized as lists over outcomes. E.g. if there are two
    outcomes, 'x' and 'y', the Xs are formatted like so: `[Xs_x_ndarray, Xs_y_ndarray]`.
    We specifically do not assume that every point is observed for every outcome.
    This means that the array for each of those outcomes may be different, and in
    particular could have a different length (e.g. if a particular arm was observed
    only for half of the outcomes, it would be present in half of the arrays in the
    list but not the other half.)

    Returns:
        A list of `TrainingData` for all outcomes, preserves the order of Xs.
    """
    if not len(Xs) == len(Ys) == len(Yvars):  # pragma: no cover
        raise ValueError("Xs, Ys, and Yvars must have equal lengths.")
    return [TrainingData(X=X, Y=Y, Yvar=Yvar) for X, Y, Yvar in zip(Xs, Ys, Yvars)]


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
