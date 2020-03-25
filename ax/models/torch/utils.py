#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple

import botorch.utils.sampling as botorch_sampling
import numpy as np
import torch
from ax.exceptions.model import ModelError
from ax.models.model_utils import filter_constraints_and_fixed_features, get_observed
from ax.models.random.sobol import SobolGenerator
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from torch import Tensor


NOISELESS_MODELS = {SingleTaskGP}


# Distributions
SIMPLEX = "simplex"
HYPERSPHERE = "hypersphere"


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


def _generate_sobol_points(
    n_sobol: int,
    bounds: List[Tuple[float, float]],
    device: torch.device,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    linear_constraints_array = None

    if linear_constraints is not None:
        linear_constraints_array = (
            linear_constraints[0].detach().numpy(),
            linear_constraints[1].detach().numpy(),
        )

    array_rounding_func = None
    if rounding_func is not None:
        array_rounding_func = tensor_callable_to_array_callable(
            tensor_func=rounding_func, device=device
        )

    sobol = SobolGenerator(deduplicate=False, seed=np.random.randint(10000))
    array_X, _ = sobol.gen(
        n=n_sobol,
        bounds=bounds,
        linear_constraints=linear_constraints_array,
        fixed_features=fixed_features,
        rounding_func=array_rounding_func,
    )
    return torch.from_numpy(array_X).to(device)


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


def subset_model(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Model, Tensor, Optional[Tuple[Tensor, Tensor]]]:
    """Subset a botorch model to the outputs used in the optimization.

    Args:
        model: A BoTorch Model. If the model does not implement the
            `subset_outputs` method, this function is a null-op and returns the
            input arguments.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)

    Returns:
        A three-tuple of model, objective_weights, and outcome_constraints, all
        subset to only those outputs that appear in either the objective weights
        or the outcome constraints.
    """
    nonzero = objective_weights != 0
    if outcome_constraints is not None:
        A, _ = outcome_constraints
        nonzero = nonzero | torch.any(A != 0, dim=0)
    idcs = torch.arange(nonzero.size(0))[nonzero].tolist()
    if len(idcs) == model.num_outputs:
        # if we use all model outputs, just return the inputs
        return model, objective_weights, outcome_constraints
    elif len(idcs) > model.num_outputs:
        raise RuntimeError(
            "Model size inconsistency. Tryting to subset a model with "
            f"{model.num_outputs} outputs to {len(idcs)} outputs"
        )
    try:
        model = model.subset_output(idcs=idcs)
        objective_weights = objective_weights[nonzero]
        if outcome_constraints is not None:
            A, b = outcome_constraints
            outcome_constraints = A[:, nonzero], b
    except NotImplementedError:
        pass
    return model, objective_weights, outcome_constraints


def _to_inequality_constraints(
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None
) -> Optional[List[Tuple[Tensor, Tensor, float]]]:
    if linear_constraints is not None:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indicies = A[i, :].nonzero().squeeze()
            coefficients = -A[i, indicies]
            rhs = -b[i, 0]
            inequality_constraints.append((indicies, coefficients, rhs))
    else:
        inequality_constraints = None
    return inequality_constraints


def sample_simplex(dim: int) -> Tensor:
    """Sample uniformly from a dim-simplex."""
    return botorch_sampling.sample_simplex(dim, dtype=torch.double).squeeze()


def sample_hypersphere_positive_quadrant(dim: int) -> Tensor:
    """Sample uniformly from the positive quadrant of a dim-sphere."""
    return torch.abs(
        botorch_sampling.sample_hypersphere(dim, dtype=torch.double).squeeze()
    )


def tensor_callable_to_array_callable(
    tensor_func: Callable[[Tensor], Tensor], device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    """"transfer a tensor callable to an array callable"""
    # TODO: move this reuseable function and its  equivalent reverse functions
    # to some utils files
    def array_func(x: np.ndarray) -> np.ndarray:
        return tensor_func(torch.from_numpy(x).to(device)).detach().numpy()

    return array_func
