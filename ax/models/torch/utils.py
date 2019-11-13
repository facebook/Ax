#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, List, Optional, Tuple

import torch
from ax.exceptions.model import ModelError
from ax.models.model_utils import filter_constraints_and_fixed_features, get_observed
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from torch import Tensor


NOISELESS_MODELS = {SingleTaskGP}


def is_noiseless(model: Model) -> bool:
    """Check if a given (single-task) botorch model is noiseless"""
    if isinstance(model, ModelListGP):
        raise ModelError(
            "Checking for noisless models only applies to sub-models of ModelListGP"
        )
    return model.__class__ in NOISELESS_MODELS


def _filter_X_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    r"""Filter input points to those appearing in objective or constraints.

    Args:
        Xs: The input tensors of a model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: All points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    # Get points observed for all objective and constraint outcomes
    X_obs = get_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
    )
    # Filter to those that satisfy constraints.
    X_obs = filter_constraints_and_fixed_features(
        X=X_obs,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    if len(X_obs) > 0:
        return torch.as_tensor(X_obs)  # please the linter


def _get_X_pending_and_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    r"""Get pending and observed points.

    Args:
        Xs: The input tensors of a model.
        pending_observations:  A list of m (k_i x d) feature tensors X
            for m outcomes and k_i pending observations for outcome i.
            (Only used if n > 1).
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: Pending points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
        Tensor: Observed points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    if pending_observations is None:
        X_pending = None
    else:
        X_pending = _filter_X_observed(
            Xs=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
    X_observed = _filter_X_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    return X_pending, X_observed


def normalize_indices(indices: List[int], d: int) -> List[int]:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between `0` and `d-1`.
    """
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices
