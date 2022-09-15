#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TParamCounter
from ax.exceptions.core import SearchSpaceExhausted
from ax.models.torch_base import TorchModel
from ax.models.types import TConfig
from botorch.acquisition.risk_measures import RiskMeasureMCObjective


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
        raise SearchSpaceExhausted(
            f"Rejection sampling error (specified maximum draws ({max_draws}) exhausted"
            f", without finding sufficiently many ({n}) candidates). This likely means "
            "that there are no new points left in the search space."
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
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Optional[Tensoray],
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    risk_measure: Optional[RiskMeasureMCObjective] = None,
    options: Optional[TConfig] = None,
) -> Optional[Tensoray]:
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
        risk_measure: An optional risk measure for reporting best robust point.
        options: A config dictionary with settings described above.

    Returns:
        A d-array of the best point, or None if no feasible point exists.
    """
    if not hasattr(model, "Xs"):
        raise ValueError(f"Model must store training data Xs, but {model} does not.")
    best_point_and_value = best_in_sample_point(
        Xs=model.Xs,  # pyre-ignore[16]: Presence of attr. checked above.
        model=model,
        bounds=bounds,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
        risk_measure=risk_measure,
        options=options,
    )
    return None if best_point_and_value is None else best_point_and_value[0]


def best_in_sample_point(
    Xs: Union[List[torch.Tensor], List[np.ndarray]],
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Optional[Tensoray],
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    risk_measure: Optional[RiskMeasureMCObjective] = None,
    options: Optional[TConfig] = None,
) -> Optional[Tuple[Tensoray, float]]:
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
        Xs: Training data for the points, among which to select the best.
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
        risk_measure: An optional risk measure for reporting best robust point.
        options: A config dictionary with settings described above.

    Returns:
        A two-element tuple or None if no feasible point exist. In tuple:
        - d-array of the best point,
        - utility at the best point.
    """
    if risk_measure is not None:
        # TODO[T131759268]: We need to apply the risk measure. Instead of doing obj_w @,
        # we could use `get_botorch_objective_and_transform` to get the objective
        # then apply it, though we also need to decide how to deal with constraints.
        raise NotImplementedError  # pragma: no cover
    # Parse options
    if options is None:
        options = {}
    method: str = options.get("best_point_method", "max_utility")
    B: Optional[float] = options.get("utility_baseline", None)
    threshold: float = options.get("probability_threshold", 0.95)
    nsamp: int = options.get("feasibility_mc_samples", 10000)
    # Get points observed for all objective and constraint outcomes
    if objective_weights is None:
        return None  # pragma: no cover
    objective_weights_np = as_array(objective_weights)
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
    if len(X_obs) == 0:
        # No feasible points
        return None
    # Predict objective and P(feas) at these points for Torch models.
    if isinstance(Xs[0], torch.Tensor):
        X_obs = X_obs.detach().clone()
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
    # pyre-fixme[61]: `utility` may not be initialized here.
    i = np.argmax(utility)
    if utility[i] == -np.Inf:
        return None
    else:
        return X_obs[i, :], utility[i]


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
    X_obs_set = {tuple(float(x_i) for x_i in x) for x in Xs[outcome_list[0]]}
    for _, idx in enumerate(outcome_list, start=1):
        X_obs_set = X_obs_set.intersection(
            {tuple(float(x_i) for x_i in x) for x in Xs[idx]}
        )
    if isinstance(Xs[0], np.ndarray):
        # pyre-fixme[6]: For 2nd param expected `Union[None, Dict[str, Tuple[typing.A...
        return np.array(list(X_obs_set), dtype=Xs[0].dtype)  # (n x d)
    if isinstance(Xs[0], torch.Tensor):
        # pyre-fixme[7]: Expected `Union[np.ndarray, torch.Tensor]` but got implicit
        #  return value of `None`.
        # pyre-fixme[6]: For 3rd param expected `Optional[_C.dtype]` but got
        #  `Union[np.dtype, _C.dtype]`.
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


def mk_discrete_choices(
    ssd: SearchSpaceDigest,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Dict[int, List[Union[int, float]]]:
    discrete_choices = ssd.discrete_choices
    # Add in fixed features.
    if fixed_features is not None:
        # Note: if any discrete features are fixed we won't enumerate those.
        discrete_choices = {
            **discrete_choices,
            **{k: [v] for k, v in fixed_features.items()},
        }
    return discrete_choices


def enumerate_discrete_combinations(
    discrete_choices: Dict[int, List[Union[int, float]]],
) -> List[Dict[int, Union[float, int]]]:
    n_combos = np.prod([len(v) for v in discrete_choices.values()])
    if n_combos > 50:
        warnings.warn(
            f"Enumerating {n_combos} combinations of discrete parameter values "
            "while optimizing over a mixed search space. This can be very slow."
        )
    fixed_features_list = [
        dict(zip(discrete_choices.keys(), c))
        for c in itertools.product(*discrete_choices.values())
    ]
    return fixed_features_list
