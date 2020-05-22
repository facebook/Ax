#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from ax.core.types import TConfig, TParamCounter
from ax.models.torch_base import TorchModel


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import models  # noqa F401  # pragma: no cover

Tensoray = Union[torch.Tensor, np.ndarray]


DEFAULT_MAX_RS_DRAWS = 10000


def rejection_sample(
    gen_unconstrained: Callable[
        [int, int, np.ndarray, Optional[Dict[int, float]]], np.ndarray
    ],
    n: int,
    d: int,
    tunable_feature_indices: np.ndarray,
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    deduplicate: bool = False,
    max_draws: Optional[int] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    existing_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Rejection sample in parameter space.

    Models must implement a `gen_unconstrained` method in order to support
    rejection sampling via this utility.
    """
    # We need to perform the round trip transformation on our generated point
    # in order to deduplicate in the original search space.
    # The transformation is applied above.
    if deduplicate and rounding_func is None:
        raise ValueError(
            "Rounding function must be provided for deduplication."  # pragma: no cover
        )

    failed_constraint_dict: TParamCounter = defaultdict(lambda: 0)
    # Rejection sample with parameter constraints.
    points = np.zeros((n, d))

    attempted_draws = 0
    successful_draws = 0
    if max_draws is None:
        max_draws = DEFAULT_MAX_RS_DRAWS

    while successful_draws < n and attempted_draws <= max_draws:
        # _gen_unconstrained returns points including fixed features.
        # pyre-ignore: Anonymous function w/ named args.
        point = gen_unconstrained(
            n=1,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )[0]

        # Note: this implementation may not be performant, if the feasible volume
        # is small, since applying the rounding_func is relatively expensive.
        # If sampling in spaces with low feasible volume is slow, this function
        # could be applied after checking the linear constraints.
        if rounding_func is not None:
            point = rounding_func(point)

        # Check parameter constraints, always in raw transformed space.
        if linear_constraints is not None:
            all_constraints_satisfied, violators = check_param_constraints(
                linear_constraints=linear_constraints, point=point
            )
            for violator in violators:
                failed_constraint_dict[violator] += 1
        else:
            all_constraints_satisfied = True
            violators = np.array([])

        # Deduplicate: don't add the same point twice.
        duplicate = False
        if deduplicate:
            if existing_points is not None:
                prev_points = np.vstack([points[:successful_draws, :], existing_points])
            else:
                prev_points = points[:successful_draws, :]
            duplicate = check_duplicate(point=point, points=prev_points)

        # Add point if valid.
        if all_constraints_satisfied and not duplicate:
            points[successful_draws] = point
            successful_draws += 1
        attempted_draws += 1

    if successful_draws < n:
        # Only possible if attempted_draws >= max_draws.
        raise ValueError(
            f"Specified maximum draws ({max_draws}) exhausted, without "
            f"finding sufficiently many ({n}) candidates."
        )
    else:
        return (points, attempted_draws)


def check_duplicate(point: np.ndarray, points: np.ndarray) -> bool:
    """Check if a point exists in another array.

    Args:
        point: Newly generated point to check.
        points: Points previously generated.

    Returns:
        True if the point is contained in points, else False
    """
    for p in points:
        if np.array_equal(p, point):
            return True
    return False


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
        2-element tuple containing

        - Flag that is True if all constraints are satisfied by the point.
        - Indices of constraints which are violated by the point.
    """
    constraints_satisfied = (
        linear_constraints[0] @ np.expand_dims(point, axis=1) <= linear_constraints[1]
    )
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
    fixed_feature_indices = list(fixed_features.keys()) if fixed_features else []
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
                "of the UnitX transform in the ModelBridge, and ensure "
                "task features are fixed."
            )


def best_observed_point(
    model: Union["models.numpy_base.NumpyModel", "models.torch_base.TorchModel"],
    bounds: List[Tuple[float, float]],
    objective_weights: Optional[Tensoray],
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[TConfig] = None,
) -> Optional[np.ndarray]:
    """Select the best point that has been observed.

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

    Returns:
        A d-array of the best point, or None if no feasible point exists.
    """
    if not hasattr(model, "Xs"):
        raise ValueError(f"Model must store training data Xs, but {model} does not.")
    # Parse options
    if options is None:
        options = {}
    # pyre-fixme[9]: method has type `str`; used as `Union[AcquisitionFunction,
    #  float, int, str]`.
    method: str = options.get("best_point_method", "max_utility")
    # pyre-fixme[9]: B has type `Optional[float]`; used as
    #  `Optional[Union[AcquisitionFunction, float, int, str]]`.
    B: Optional[float] = options.get("utility_baseline", None)
    # pyre-fixme[9]: threshold has type `float`; used as `Union[AcquisitionFunction,
    #  float, int, str]`.
    threshold: float = options.get("probability_threshold", 0.95)
    # pyre-fixme[9]: nsamp has type `int`; used as `Union[AcquisitionFunction,
    #  float, int, str]`.
    nsamp: int = options.get("feasibility_mc_samples", 10000)
    # Get points observed for all objective and constraint outcomes
    if objective_weights is None:
        return None  # pragma: no cover
    objective_weights_np = as_array(objective_weights)
    X_obs = get_observed(
        # pyre-fixme[16]: attribute must exist, otherwise error raised above
        Xs=model.Xs,
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
    if len(X_obs) == 0:
        # No feasible points
        return None
    # Predict objective and P(feas) at these points
    if isinstance(model, TorchModel):
        X_obs = X_obs.clone().detach()
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

    Returns:
        x, with everything converted to array.
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


def get_observed(
    Xs: Union[List[torch.Tensor], List[np.ndarray]],
    objective_weights: Tensoray,
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
) -> Tensoray:
    """Filter points to those that are observed for objective outcomes and outcomes
    that show up in outcome_constraints (if there are any).

    Args:
        Xs: A list of m (k_i x d) feature matrices X. Number of rows k_i
            can vary from i=1,...,m.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.

    Returns:
        Points observed for all objective outcomes and outcome constraints.
    """
    objective_weights_np = as_array(objective_weights)
    used_outcomes: Set[int] = set(np.where(objective_weights_np != 0)[0])
    if len(used_outcomes) == 0:
        raise ValueError("At least one objective weight must be non-zero")
    if outcome_constraints is not None:
        used_outcomes = used_outcomes.union(
            np.where(as_array(outcome_constraints)[0] != 0)[1]
        )
    outcome_list = list(used_outcomes)
    # pyre-fixme[16]: `Tensor` has no attribute `__iter__`.
    X_obs_set = {tuple(float(x_i) for x_i in x) for x in Xs[outcome_list[0]]}
    for _, idx in enumerate(outcome_list, start=1):
        X_obs_set = X_obs_set.intersection(
            {tuple(float(x_i) for x_i in x) for x in Xs[idx]}
        )
    if isinstance(Xs[0], np.ndarray):
        # pyre-fixme[6]: Expected `Union[None, Dict[str, Tuple[typing.Any, int]],
        #  Dict[str, Union[typing.Sequence[typing.Any], typing.Sequence[Union[None,
        #  bytes, str]], typing.Sequence[int], typing.Sequence[str], int]],
        #  List[Union[Tuple[Union[Tuple[str, str], str], typing.Any],
        #  Tuple[Union[Tuple[str, str], str], typing.Any, Union[typing.Sequence[int],
        #  int]]]], Tuple[typing.Any, typing.Any], Tuple[typing.Any,
        #  Union[typing.Sequence[int], int]], Tuple[typing.Any, int],
        #  typing.Type[typing.Any], np.dtype, str]` for 2nd param but got
        #  `Union[np.dtype, torch.dtype]`.
        return np.array(list(X_obs_set), dtype=Xs[0].dtype)  # (n x d)
    if isinstance(Xs[0], torch.Tensor):
        # pyre-fixme[7]: Expected `Union[np.ndarray, torch.Tensor]` but got implicit
        #  return value of `None`.
        return torch.tensor(list(X_obs_set), device=Xs[0].device, dtype=Xs[0].dtype)


def filter_constraints_and_fixed_features(
    X: Tensoray,
    bounds: List[Tuple[float, float]],
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tensoray:
    """Filter points to those that satisfy bounds, linear_constraints, and
    fixed_features.

    Args:
        X: An tensor or array of points.
        bounds: A list of (lower, upper) tuples for each feature.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value in the best point.

    Returns:
        Feasible points.
    """
    if len(X) == 0:  # if there are no points, nothing to filter
        return X
    X_np = X
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    feas = np.ones(X_np.shape[0], dtype=bool)  # (n)
    for i, b in enumerate(bounds):
        feas &= (X_np[:, i] >= b[0]) & (X_np[:, i] <= b[1])
    if linear_constraints is not None:
        A, b = as_array(linear_constraints)  # (m x d) and (m x 1)
        feas &= (A @ X_np.transpose() <= b).all(axis=0)
    if fixed_features is not None:
        for idx, val in fixed_features.items():
            feas &= X_np[:, idx] == val
    X_feas = X_np[feas, :]
    if isinstance(X, torch.Tensor):
        return torch.from_numpy(X_feas).to(device=X.device, dtype=X.dtype)
    else:
        return X_feas
