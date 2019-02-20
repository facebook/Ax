#!/usr/bin/env python3

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from ae.lazarus.ae.core.types.types import TConfig, TParamCounter
from ae.lazarus.ae.models.torch_base import TorchModel


if TYPE_CHECKING:
    from ae.lazarus.ae.models.numpy.gpy import GPyGP  # noqa F401  # pragma: no cover
    from ae.lazarus.ae.models.random.base import (  # noqa F401  # pragma: no cover
        RandomModel,
    )
    from ae.lazarus.ae.models.torch.botorch import (  # noqa F401  # pragma: no cover
        BotorchModel,
    )


Tensoray = Union[torch.Tensor, np.ndarray]


DEFAULT_MAX_RS_DRAWS = 10000


def rejection_sample(
    model: "RandomModel",
    n: int,
    d: int,
    tunable_feature_indices: np.ndarray,
    linear_constraints: Tuple[np.ndarray, np.ndarray],
    max_draws: Optional[int] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, int]:
    """Rejection sample in parameter space.

    Models must implement a `gen_unconstrained` method in order to support
    rejection sampling via this utility.

    """
    failed_constraint_dict: TParamCounter = defaultdict(lambda: 0)  # pyre-ignore
    # Rejection sample with parameter constraints.
    points = np.zeros((n, d))
    attempted_draws = 0
    successful_draws = 0
    if max_draws is None:
        max_draws = DEFAULT_MAX_RS_DRAWS

    while successful_draws < n:
        # _gen_unconstrained returns points including fixed features.
        point = model._gen_unconstrained(
            n=1,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )
        attempted_draws += 1
        all_satisfied, violators = check_param_constraints(
            linear_constraints=linear_constraints, point=point
        )
        if all_satisfied:
            points[successful_draws] = point
            successful_draws += 1
        else:
            for violator in violators:
                failed_constraint_dict[violator] += 1
        if max_draws is not None and attempted_draws > max_draws:
            raise ValueError(
                f"Specified maximum draws ({max_draws}) exhausted, without "
                "finding sufficiently many ({n}) candidates."
            )
    return (points, attempted_draws)


def add_fixed_features(
    tunable_points: np.ndarray,
    d: int,
    fixed_features: Optional[Dict[int, float]],
    tunable_feature_indices: np.ndarray,
) -> np.ndarray:
    """Add fixed features to points in tunable space.

    Args:
        tunable_points: Points in tunable space.
        d: Dimension of parameter space.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        tunable_feature_indices: Parameter indices (in d) which are tunable.

    Returns:
        points: Points in the full d-dimensional space, defined by bounds.

    """
    n = np.shape(tunable_points)[0]
    points = np.zeros((n, d))
    points[:, tunable_feature_indices] = tunable_points
    if fixed_features:
        fixed_feature_indices = np.array(list(fixed_features.keys()))
        fixed_values = np.tile(list(fixed_features.values()), (n, 1))
        points[:, fixed_feature_indices] = fixed_values
    return points


def check_param_constraints(
    linear_constraints: Tuple[np.ndarray, np.ndarray], point: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """Check if a point satisfies parameter constraints.

    Args:
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
        point: A candidate point in d-dimensional space, as a (1 x d) matrix.

    Returns:
        all_satisfied: True if all constraints are satisfied by the point.
        violators: Indices of constraints which are violated by the point.
    """
    constraints_satisfied = linear_constraints[0] @ point.T <= linear_constraints[1]
    if np.all(constraints_satisfied):
        return True, np.array([])
    else:
        return (False, np.where(constraints_satisfied == False)[0])  # noqa: E712


def tunable_feature_indices(
    bounds: List[Tuple[float, float]], fixed_features: Optional[Dict[int, float]] = None
) -> np.ndarray:
    """Get the feature indices of tunable features.

    Args:
        bounds: A list of (lower, upper) tuples for each column of X.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        The indices of tunable features.
    """
    if fixed_features:
        fixed_feature_indices = np.array(list(fixed_features.keys()))
    else:
        fixed_feature_indices = np.array([])
    feature_indices = np.arange(len(bounds))
    return np.delete(feature_indices, fixed_feature_indices)


def validate_bounds(
    bounds: List[Tuple[float, float]], fixed_feature_indices: np.ndarray
) -> None:
    """Ensure the requested space is [0,1]^d.

    Args:
        bounds: A list of d (lower, upper) tuples for each column of X.
        fixed_feature_indices: Indices of features which are fixed at a
            particular value.
    """
    for feature_idx, bound in enumerate(bounds):
        # Bounds for fixed features are not unit-transformed.
        if feature_idx in fixed_feature_indices:
            continue

        if bound[0] != 0 or bound[1] != 1:
            raise ValueError(
                "This generator operates on [0,1]^d. Please make use "
                "of the UnitX transform in the Generator, and ensure "
                "task features are fixed."
            )


def best_observed_point(
    model: Union["GPyGP", "BotorchModel"],
    bounds: List[Tuple[float, float]],
    objective_weights: Optional[Tensoray],
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[TConfig] = None,
) -> Optional[np.ndarray]:
    """
    Select the best point that has been observed.

    Implements two approaches to selecting the best point.

    For both approaches, only points that satisfy parameter space constraints
    (bounds, linear_constraints, fixed_features) will be returned. Points must
    also be observed for all objective and constraint outcomes. Returned
    points may violate outcome constraints, depending on the method below.

    1: Select the point that maximizes the expected utility
    (objective_weights^T posterior_objective_means - baseline) * Prob(feasible)
    Here baseline should be selected so that at least one point has positive
    utility. It can be specified in the options dict, otherwise
    min (objective_weights^T posterior_objective_means)
    will be used, where the min is over observed points.

    2: Select the best-objective point that is feasible with at least
    probability p.

    The following quantities may be specified in the options dict:
    - best_point_method: 'max_utility' (default) or 'feasible_threshold'
        to select between the two approaches described above.
    - utility_baseline: Value for the baseline used in max_utility approach. If
        not provided, defaults to min objective value.
    - probability_threshold: Threshold for the feasible_threshold approach.
        Defaults to p=0.95.
    - feasibility_mc_samples: Number of MC samples used for estimating the
        probability of feasibility (defaults 10k).

    Returns None if no best point can be identified (for example, if no
    observations satisfying the constraints).

    Accepts either tensors or arrays, but returns an array.

    Args:
        model: Numpy or Torch model.
        bounds: A list of (lower, upper) tuples for each feature.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value in the best point.
        options: A config dictionary with settings described above.

        Returns: A d-array of the best point, or None if no feasible point
            exists.
    """
    # Parse options
    if options is None:
        options = {}
    method: str = options.get("best_point_method", "max_utility")  # pyre-ignore
    B: Optional[float] = options.get("utility_baseline", None)  # pyre-ignore
    threshold: float = options.get("probability_threshold", 0.95)  # pyre-ignore
    nsamp: int = options.get("feasibility_mc_samples", 10000)  # pyre-ignore
    # Get points observed for all objective and constraint outcomes
    if objective_weights is None:
        return None  # pragma: no cover
    objective_weights_np = as_array(objective_weights)
    used_outcomes: Set[int] = set(np.where(objective_weights_np != 0)[0])
    if len(used_outcomes) == 0:
        raise ValueError("At least one objective weight must be non-zero")
    if outcome_constraints is not None:
        used_outcomes = used_outcomes.union(
            np.where(as_array(outcome_constraints)[0] != 0)[1]
        )
    outcome_list = list(used_outcomes)
    X_obs_set = {tuple(float(x_i) for x_i in x) for x in model.Xs[outcome_list[0]]}
    for _, idx in enumerate(outcome_list, start=1):
        X_obs_set = X_obs_set.intersection(
            {tuple(float(x_i) for x_i in x) for x in model.Xs[idx]}
        )
    X_obs = np.array(list(X_obs_set))  # (n x d)
    # Filter to those that satisfy constraints.
    feas = np.ones(X_obs.shape[0], dtype=bool)  # (n)
    for i, b in enumerate(bounds):
        feas &= (X_obs[:, i] >= b[0]) & (X_obs[:, i] <= b[1])
    if linear_constraints is not None:
        A, b = as_array(linear_constraints)  # (m x d) and (m x 1)
        feas &= (A @ X_obs.transpose() <= b).all(axis=0)
    if fixed_features is not None:
        for idx, val in fixed_features.items():
            feas &= X_obs[:, idx] == val
    if sum(feas) == 0:
        # No feasible points
        return None
    X_obs = X_obs[feas, :]
    # Predict objective and P(feas) at these points
    if isinstance(model, TorchModel):
        X_obs = torch.tensor(X_obs, device=model.device, dtype=model.dtype)
    f, cov = as_array(model.predict(X_obs))
    obj = objective_weights_np @ f.transpose()  # pyre-ignore
    pfeas = np.ones_like(obj)
    if outcome_constraints is not None:
        A, b = as_array(outcome_constraints)  # (m x j) and (m x 1)
        # Use Monte Carlo to compute pfeas, to properly handle covariance
        # across outcomes.
        for i, _ in enumerate(X_obs):
            z = np.random.multivariate_normal(
                mean=f[i, :], cov=cov[i, :, :], size=nsamp
            )  # (nsamp x j)
            pfeas[i] = (A @ z.transpose() <= b).all(axis=0).mean()
    # Identify best point
    if method == "feasible_threshold":
        utility = obj
        utility[pfeas < threshold] = -np.Inf
    elif method == "max_utility":
        if B is None:
            B = obj.min()
        utility = (obj - B) * pfeas
    i = np.argmax(utility)
    if utility[i] == -np.Inf:
        return None
    else:
        return X_obs[i, :]


def as_array(
    x: Union[Tensoray, Tuple[Tensoray, ...]]
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Convert every item in a tuple of tensors/arrays into an array.

    Args:
        x: A tensor, array, or a tuple of potentially mixed tensors and arrays.

    Returns: x, with everything convert to array.
    """
    if isinstance(x, tuple):
        return tuple(as_array(x_i) for x_i in x)  # pyre-ignore
    elif isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    else:
        raise ValueError(
            "Input to as_array must be numpy array or torch tensor"
        )  # pragma: no cover
