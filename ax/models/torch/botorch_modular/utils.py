#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def use_model_list(Xs: List[Tensor], botorch_model_class: Type[Model]) -> bool:
    if issubclass(botorch_model_class, MultiTaskGP):
        # We currently always wrap multi-task models into `ModelListGP`.
        return True
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
    Yvars: List[Tensor],
    search_space_digest: SearchSpaceDigest,
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
    if len(search_space_digest.fidelity_features) > 1:
        raise NotImplementedError(
            "Only a single fidelity feature supported "
            f"(got: {search_space_digest.fidelity_features})."
        )
    if len(search_space_digest.task_features) > 1:
        raise NotImplementedError(
            f"Only a single task feature supported "
            f"(got: {search_space_digest.task_features})."
        )
    if search_space_digest.task_features and search_space_digest.fidelity_features:
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
    if search_space_digest.task_features and all_nan_Yvar:
        return MultiTaskGP  # Unknown observation noise.
    elif search_space_digest.task_features:
        return FixedNoiseMultiTaskGP  # Known observation noise.

    # Single-task multi-fidelity cases.
    if search_space_digest.fidelity_features and all_nan_Yvar:
        return SingleTaskMultiFidelityGP  # Unknown observation noise.
    elif search_space_digest.fidelity_features:
        return FixedNoiseMultiFidelityGP  # Known observation noise.

    # Single-task single-fidelity cases.
    elif all_nan_Yvar:  # Unknown observation noise.
        return SingleTaskGP
    return FixedNoiseGP  # Known observation noise.


def choose_botorch_acqf_class(
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    objective_thresholds: Optional[Tensor] = None,
) -> Type[AcquisitionFunction]:
    """Chooses a BoTorch `AcquisitionFunction` class."""
    if objective_thresholds is not None:
        return qNoisyExpectedHypervolumeImprovement
    return qNoisyExpectedImprovement


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
