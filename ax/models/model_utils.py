#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import SearchSpaceExhausted, UnsupportedError
from ax.models.types import TConfig
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.exceptions.warnings import OptimizationWarning
from pyre_extensions import assert_is_instance
from torch import Tensor


# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
Tensoray = Union[torch.Tensor, np.ndarray]
TTensoray = TypeVar("TTensoray", bound=Tensoray)


class TorchModelLike(Protocol):
    """A protocol that stands in for ``TorchModel`` like objects that
    have a ``predict`` method.
    """

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predicts outcomes given an input tensor.

        Args:
            X: A ``n x d`` tensor of input parameters.

        Returns:
            Tensor: The predicted posterior mean as an ``n x o``-dim tensor.
            Tensor: The predicted posterior covariance as a ``n x o x o``-dim tensor.
        """
        ...


DEFAULT_MAX_RS_DRAWS = 10000


def rejection_sample(
    gen_unconstrained: Callable[
        [int, int, npt.NDArray, dict[int, float] | None], npt.NDArray
    ],
    n: int,
    d: int,
    tunable_feature_indices: npt.NDArray,
    linear_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
    deduplicate: bool = False,
    max_draws: int | None = None,
    fixed_features: dict[int, float] | None = None,
    rounding_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
    existing_points: npt.NDArray | None = None,
) -> tuple[npt.NDArray, int]:
    """Rejection sample in parameter space. Parameter space is typically
    [0, 1] for all tunable parameters.

    Models must implement a `gen_unconstrained` method in order to support
    rejection sampling via this utility.

    Args:
        gen_unconstrained: A callable that generates unconstrained points in
            the parameter space. This is typically the `_gen_unconstrained` method
            of a `RandomModel`.
        n: Number of samples to generate.
        d: Dimensionality of the parameter space.
        tunable_feature_indices: Indices of the tunable features in the
            parameter space.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        deduplicate: If true, reject points that are duplicates of previously
            generated points. The points are deduplicated after applying the
            rounding function.
        max_draws: Maximum number of attemped draws before giving up.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (e.g., according to `round-trip` transformations).
        existing_points: A set of previously generated points to use
            for deduplication. These should be provided in the parameter
            space model operates in.
    """
    # We need to perform the round trip transformation on our generated point
    # in order to deduplicate in the original search space.
    # The transformation is applied above.
    if deduplicate and rounding_func is None:
        raise ValueError("Rounding function must be provided for deduplication.")

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

        # Check parameter constraints, always in raw transformed space.
        if linear_constraints is not None:
            all_constraints_satisfied, _ = check_param_constraints(
                linear_constraints=linear_constraints, point=point
            )
        else:
            all_constraints_satisfied = True

        if all_constraints_satisfied:
            # Apply the rounding function if the point satisfies the linear constraints.
            if rounding_func is not None:
                # NOTE: This could still fail rounding with a logger warning. But this
                # should be rare since the point is feasible in the continuous space.
                point = rounding_func(point)

            # Deduplicate: don't add the same point twice.
            duplicate = False
            if deduplicate:
                if existing_points is not None:
                    prev_points = np.vstack(
                        [points[:successful_draws, :], existing_points]
                    )
                else:
                    prev_points = points[:successful_draws, :]
                duplicate = check_duplicate(point=point, points=prev_points)

            # Add point if valid.
            if not duplicate:
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


def check_duplicate(point: npt.NDArray, points: npt.NDArray) -> bool:
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
    tunable_points: npt.NDArray,
    d: int,
    fixed_features: dict[int, float] | None,
    tunable_feature_indices: npt.NDArray,
) -> npt.NDArray:
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
    linear_constraints: tuple[npt.NDArray, npt.NDArray],
    point: npt.NDArray,
) -> tuple[bool, npt.NDArray]:
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
    bounds: list[tuple[float, float]],
    fixed_features: dict[int, float] | None = None,
) -> npt.NDArray:
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
    bounds: Sequence[tuple[float, float]],
    fixed_feature_indices: npt.NDArray,
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
    model: TorchModelLike,
    bounds: Sequence[tuple[float, float]],
    objective_weights: TTensoray | None,
    outcome_constraints: tuple[TTensoray, TTensoray] | None = None,
    linear_constraints: tuple[TTensoray, TTensoray] | None = None,
    fixed_features: dict[int, float] | None = None,
    risk_measure: RiskMeasureMCObjective | None = None,
    options: TConfig | None = None,
) -> TTensoray | None:
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
        model: A Torch model or Surrogate.
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
    Xs: Sequence[TTensoray],
    model: TorchModelLike,
    bounds: Sequence[tuple[float, float]],
    objective_weights: TTensoray | None,
    outcome_constraints: tuple[TTensoray, TTensoray] | None = None,
    linear_constraints: tuple[TTensoray, TTensoray] | None = None,
    fixed_features: dict[int, float] | None = None,
    risk_measure: RiskMeasureMCObjective | None = None,
    options: TConfig | None = None,
) -> tuple[TTensoray, float] | None:
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
        model: A Torch model or Surrogate.
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
        raise NotImplementedError
    # Parse options
    if options is None:
        options = {}
    method: str = options.get("best_point_method", "max_utility")
    B: float | None = options.get("utility_baseline", None)
    threshold: float = options.get("probability_threshold", 0.95)
    nsamp: int = options.get("feasibility_mc_samples", 10000)
    # Get points observed for all objective and constraint outcomes
    if objective_weights is None:
        return None
    objective_weights_np = assert_is_instance(as_array(objective_weights), np.ndarray)
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
        # pyre-fixme[16]: Item `ndarray` of `Union[ndarray[typing.Any, typing.Any],
        #  Tensor]` has no attribute `detach`.
        X_obs = X_obs.detach().clone()
    # (n_feasible x n_outcomes), (n_feasible x n_outcomes x n_outcomes)
    f, cov = as_array(model.predict(X_obs))
    # (n_outcomes,) x (n_outcomes, n_feasible) => (n_feasible,)
    obj = objective_weights_np @ f.transpose()
    pfeas = np.ones_like(obj)
    if outcome_constraints is not None:
        # (n_constraints x n_outcomes) and (n_constraints x 1)
        A, b = as_array(outcome_constraints)
        # Use Monte Carlo to compute pfeas, to properly handle covariance
        # across outcomes.
        for i, _ in enumerate(X_obs):
            # nsamp x n_outcomes
            z = np.random.multivariate_normal(
                mean=f[i, :], cov=cov[i, :, :], size=nsamp
            )
            # (n_constraints x n_outcomes) @ (n_outcomes x nsamp)
            pfeas[i] = (A @ z.transpose() <= b).all(axis=0).mean()
    # Identify best point
    if method == "feasible_threshold":
        utility = obj
        utility[pfeas < threshold] = -np.inf
    elif method == "max_utility":
        if B is None:
            B = obj.min()
        utility = (obj - B) * pfeas
    else:  # pragma: no cover
        raise UnsupportedError(f"Unknown best point method {method}.")
    i = np.argmax(utility)
    if utility[i] == -np.inf:
        return None
    else:
        return X_obs[i, :], utility[i]


def as_array(
    x: Tensoray | tuple[Tensoray, ...],
) -> npt.NDArray | tuple[npt.NDArray, ...]:
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
        raise ValueError("Input to as_array must be numpy array or torch tensor")


def get_observed(
    Xs: Sequence[TTensoray],
    objective_weights: TTensoray,
    outcome_constraints: tuple[TTensoray, TTensoray] | None = None,
) -> TTensoray:
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
    used_outcomes: set[int] = set(np.where(objective_weights_np != 0)[0])
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
        # pyre-fixme[7]: This function only returns a Numpy array when Xs
        # contains all Numpy arrays, but Pyre doesn't understand
        return np.array(list(X_obs_set), dtype=Xs[0].dtype)  # (n x d)
    # pyre-fixme[7]: This function only returns a tensor when Xs
    # contains all tensors, but Pyre doesn't understand`.
    return torch.tensor(
        list(X_obs_set),
        device=assert_is_instance(Xs[0], torch.Tensor).device,
        dtype=Xs[0].dtype,
    )


def filter_constraints_and_fixed_features(
    X: TTensoray,
    bounds: Sequence[tuple[float, float]],
    linear_constraints: tuple[TTensoray, TTensoray] | None = None,
    fixed_features: dict[int, float] | None = None,
) -> TTensoray:
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
    # pyre-ignore: Undefined attribute [16]: `np.ndarray` has no attribute
    # `cpu`.
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
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
    return X_feas


def mk_discrete_choices(
    ssd: SearchSpaceDigest,
    fixed_features: Mapping[int, float] | None = None,
) -> Mapping[int, Sequence[float]]:
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
    discrete_choices: Mapping[int, Sequence[float]],
) -> list[dict[int, float]]:
    n_combos = np.prod([len(v) for v in discrete_choices.values()])
    if n_combos > 50:
        warnings.warn(
            f"Enumerating {n_combos} combinations of discrete parameter values "
            "while optimizing over a mixed search space. This can be very slow.",
            OptimizationWarning,
            stacklevel=2,
        )
    fixed_features_list = [
        dict(zip(discrete_choices.keys(), c))
        for c in itertools.product(*discrete_choices.values())
    ]
    return fixed_features_list


def all_ordinal_features_are_integer_valued(
    ssd: SearchSpaceDigest,
) -> bool:
    """Check if all ordinal features are integer-valued.

    Args:
        ssd: A SearchSpaceDigest.

    Returns:
        True if all ordinal features are integer-valued, False otherwise.
    """
    for feature_idx in ssd.ordinal_features:
        choices = ssd.discrete_choices[feature_idx]
        int_choices = [int(c) for c in choices]
        if choices != int_choices:
            return False
    return True
