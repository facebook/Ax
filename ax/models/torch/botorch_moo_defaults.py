#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
References

.. [Daulton2020qehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems 33, 2020.

.. [Daulton2021nehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of
    Multiple Noisy Objectives with Expected Hypervolume Improvement. Advances
    in Neural Information Processing Systems 34, 2021.

.. [Ament2023logei]
    S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy.
    Unexpected Improvements to Expected Improvement for Bayesian Optimization. Advances
    in Neural Information Processing Systems 36, 2023.
"""

from __future__ import annotations

from collections.abc import Callable

from typing import cast, Optional, Union

import torch
from ax.exceptions.core import AxError
from ax.models.torch.botorch_defaults import NO_OBSERVED_POINTS_MESSAGE
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_outcome_constraint_transforms,
    subset_model,
)
from ax.models.torch_base import TorchModel
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf_list
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

DEFAULT_EHVI_MC_SAMPLES = 128


# Callable that takes tensors of observations and model parameters,
# then returns means of observations that make up a pareto frontier,
# along with their covariances and their index in the input observations.
TFrontierEvaluator = Callable[
    [
        TorchModel,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[tuple[Tensor, Tensor]],
    ],
    tuple[Tensor, Tensor, Tensor],
]

NO_FEASIBLE_POINTS_MESSAGE = (
    " Cannot infer objective thresholds due to no observed feasible points. "
    " This likely means that one or more outcome constraints is set too strictly.  "
    " Consider adding thresholds to your objectives to bypass this error."
)


def get_weighted_mc_objective_and_objective_thresholds(
    objective_weights: Tensor, objective_thresholds: Tensor
) -> tuple[WeightedMCMultiOutputObjective, Tensor]:
    r"""Construct weighted objective and apply the weights to objective thresholds.

    Args:
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        objective_thresholds: A tensor containing thresholds forming a reference point
            from which to calculate pareto frontier hypervolume. Points that do not
            dominate the objective_thresholds contribute nothing to hypervolume.

    Returns:
        A two-element tuple with the objective and objective thresholds:

            - The objective
            - The objective thresholds

    """
    nonzero_idcs = objective_weights.nonzero(as_tuple=False).view(-1)
    objective_weights = objective_weights[nonzero_idcs]
    objective_thresholds = objective_thresholds[nonzero_idcs]
    objective = WeightedMCMultiOutputObjective(
        weights=objective_weights, outcomes=nonzero_idcs.tolist()
    )
    objective_thresholds = torch.mul(objective_thresholds, objective_weights)
    return objective, objective_thresholds


def get_NEHVI(
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    prune_baseline: bool = True,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    marginalize_dim: int | None = None,
    cache_root: bool = True,
    seed: int | None = None,
) -> qNoisyExpectedHypervolumeImprovement:
    r"""Instantiates a qNoisyExpectedHyperVolumeImprovement acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        X_observed: A tensor containing points observed for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        X_pending: A tensor containing points whose evaluation is pending (i.e.
            that have been submitted for evaluation) present for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        prune_baseline: If True, prune the baseline points for NEI (default: True).
        mc_samples: The number of MC samples to use (default: 512).
        alpha: The hyperparameter controlling the approximate non-dominated
            partitioning. The default value of 0.0 means an exact partitioning
            is used. As the number of objectives `m` increases, consider increasing
            this parameter in order to limit computational complexity (default: None).
        marginalize_dim: The dimension along which to marginalize over, used for fully
            Bayesian models (default: None).
        cache_root: If True, cache the root of the covariance matrix (default: True).
        seed: The random seed for generating random starting points for optimization (
            default: None).

    Returns:
        qNoisyExpectedHyperVolumeImprovement: The instantiated acquisition function.
    """
    return assert_is_instance(
        _get_NEHVI(
            acqf_name="qNEHVI",
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            mc_samples=mc_samples,
            alpha=alpha,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
            seed=seed,
        ),
        qNoisyExpectedHypervolumeImprovement,
    )


def get_qLogNEHVI(
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    prune_baseline: bool = True,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    marginalize_dim: int | None = None,
    cache_root: bool = True,
    seed: int | None = None,
) -> qLogNoisyExpectedHypervolumeImprovement:
    r"""Instantiates a qLogNoisyExpectedHyperVolumeImprovement acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        X_observed: A tensor containing points observed for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        X_pending: A tensor containing points whose evaluation is pending (i.e.
            that have been submitted for evaluation) present for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        prune_baseline: If True, prune the baseline points for NEI (default: True).
        mc_samples: The number of MC samples to use (default: 512).
        alpha: The hyperparameter controlling the approximate non-dominated
            partitioning. The default value of 0.0 means an exact partitioning
            is used. As the number of objectives `m` increases, consider increasing
            this parameter in order to limit computational complexity (default: None).
        marginalize_dim: The dimension along which to marginalize over, used for fully
            Bayesian models (default: None).
        cache_root: If True, cache the root of the covariance matrix (default: True).
        seed: The random seed for generating random starting points for optimization (
            default: None).

    Returns:
        qLogNoisyExpectedHyperVolumeImprovement: The instantiated acquisition function.
    """
    return assert_is_instance(
        _get_NEHVI(
            acqf_name="qLogNEHVI",
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            mc_samples=mc_samples,
            alpha=alpha,
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
            seed=seed,
        ),
        qLogNoisyExpectedHypervolumeImprovement,
    )


def _get_NEHVI(
    acqf_name: str,
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    prune_baseline: bool = True,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    marginalize_dim: int | None = None,
    cache_root: bool = True,
    seed: int | None = None,
) -> qNoisyExpectedHypervolumeImprovement | qLogNoisyExpectedHypervolumeImprovement:
    if X_observed is None:
        raise ValueError(NO_OBSERVED_POINTS_MESSAGE)
    # construct Objective module
    (
        objective,
        objective_thresholds,
    ) = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=objective_weights, objective_thresholds=objective_thresholds
    )
    # For EHVI acquisition functions we pass the constraint transform directly.
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
    num_objectives = objective_thresholds.shape[0]
    if alpha is None:
        alpha = get_default_partitioning_alpha(num_objectives=num_objectives)
    # NOTE: Not using checked_cast here because for Python 3.9, isinstance fails with
    # `TypeError: Subscripted generics cannot be used with class and instance checks`.
    return cast(
        Union[
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ],
        get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=model,
            objective=objective,
            X_observed=X_observed,
            X_pending=X_pending,
            constraints=cons_tfs,
            prune_baseline=prune_baseline,
            mc_samples=mc_samples,
            alpha=alpha,
            seed=(
                seed
                if seed is not None
                else cast(int, torch.randint(1, 10000, (1,)).item())
            ),
            ref_point=objective_thresholds.tolist(),
            marginalize_dim=marginalize_dim,
            cache_root=cache_root,
        ),
    )


def get_EHVI(
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    seed: int | None = None,
) -> qExpectedHypervolumeImprovement:
    r"""Instantiates a qExpectedHyperVolumeImprovement acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        objective_thresholds:  A tensor containing thresholds forming a reference point
            from which to calculate pareto frontier hypervolume. Points that do not
            dominate the objective_thresholds contribute nothing to hypervolume.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        X_observed: A tensor containing points observed for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        X_pending: A tensor containing points whose evaluation is pending (i.e.
            that have been submitted for evaluation) present for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        mc_samples: The number of MC samples to use (default: 512).
        alpha: The hyperparameter controlling the approximate non-dominated
            partitioning. The default value of 0.0 means an exact partitioning
            is used. As the number of objectives `m` increases, consider increasing
            this parameter in order to limit computational complexity.
        seed: The random seed for generating random starting points for optimization.

    Returns:
        qExpectedHypervolumeImprovement: The instantiated acquisition function.
    """
    return assert_is_instance(
        _get_EHVI(
            acqf_name="qEHVI",
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            X_pending=X_pending,
            mc_samples=mc_samples,
            alpha=alpha,
            seed=seed,
        ),
        qExpectedHypervolumeImprovement,
    )


def get_qLogEHVI(
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    seed: int | None = None,
) -> qLogExpectedHypervolumeImprovement:
    r"""Instantiates a qLogExpectedHyperVolumeImprovement acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        objective_thresholds:  A tensor containing thresholds forming a reference point
            from which to calculate pareto frontier hypervolume. Points that do not
            dominate the objective_thresholds contribute nothing to hypervolume.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        X_observed: A tensor containing points observed for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        X_pending: A tensor containing points whose evaluation is pending (i.e.
            that have been submitted for evaluation) present for all objective
            outcomes and outcomes that appear in the outcome constraints (if
            there are any).
        mc_samples: The number of MC samples to use (default: 512).
        alpha: The hyperparameter controlling the approximate non-dominated
            partitioning. The default value of 0.0 means an exact partitioning
            is used. As the number of objectives `m` increases, consider increasing
            this parameter in order to limit computational complexity.
        seed: The random seed for generating random starting points for optimization.

    Returns:
        qLogExpectedHypervolumeImprovement: The instantiated acquisition function.
    """
    return assert_is_instance(
        _get_EHVI(
            acqf_name="qLogEHVI",
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            X_pending=X_pending,
            mc_samples=mc_samples,
            alpha=alpha,
            seed=seed,
        ),
        qLogExpectedHypervolumeImprovement,
    )


def _get_EHVI(
    acqf_name: str,
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    *,
    mc_samples: int = DEFAULT_EHVI_MC_SAMPLES,
    alpha: float | None = None,
    seed: int | None = None,
) -> qExpectedHypervolumeImprovement | qLogExpectedHypervolumeImprovement:
    if X_observed is None:
        raise ValueError(NO_OBSERVED_POINTS_MESSAGE)
    # construct Objective module
    (
        objective,
        objective_thresholds,
    ) = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=objective_weights, objective_thresholds=objective_thresholds
    )
    with torch.no_grad():
        Y = _check_posterior_type(model.posterior(X_observed)).mean
    # For EHVI acquisition functions we pass the constraint transform directly.
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
    num_objectives = objective_thresholds.shape[0]
    # NOTE: Not using checked_cast here because for Python 3.9, isinstance fails with
    # `TypeError: Subscripted generics cannot be used with class and instance checks`.
    return cast(
        Union[qExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement],
        get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=model,
            objective=objective,
            X_observed=X_observed,
            X_pending=X_pending,
            constraints=cons_tfs,
            mc_samples=mc_samples,
            alpha=(
                get_default_partitioning_alpha(num_objectives=num_objectives)
                if alpha is None
                else alpha
            ),
            seed=(
                seed
                if seed is not None
                else cast(int, torch.randint(1, 10000, (1,)).item())
            ),
            ref_point=objective_thresholds.tolist(),
            Y=Y,
        ),
    )


# TODO (jej): rewrite optimize_acqf wrappers to avoid duplicate code.
def scipy_optimizer_list(
    acq_function_list: list[AcquisitionFunction],
    bounds: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    fixed_features: dict[int, float] | None = None,
    rounding_func: Callable[[Tensor], Tensor] | None = None,
    num_restarts: int = 20,
    raw_samples: int | None = None,
    options: dict[str, bool | float | int | str] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Sequential optimizer using scipy's minimize module on a numpy-adaptor.

    The ith acquisition in the sequence uses the ith given acquisition_function.

    Args:
        acq_function_list: A list of botorch AcquisitionFunctions,
            optimized sequentially.
        bounds: A `2 x d`-dim tensor, where `bounds[0]` (`bounds[1]`) are the
            lower (upper) bounds of the feasible hyperrectangle.
        n: The number of candidates to generate.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        fixed_features: A map {feature_index: value} for features that should
            be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).

    Returns:
        2-element tuple containing

        - A `n x d`-dim tensor of generated candidates.
        - A `n`-dim tensor of conditional acquisition
          values, where `i`-th element is the expected acquisition value
          conditional on having observed candidates `0,1,...,i-1`.
    """
    # Use SLSQP by default for small problems since it yields faster wall times.
    optimize_options: dict[str, bool | float | int | str] = {
        "batch_limit": 5,
        "init_batch_limit": 32,
        "method": "SLSQP",
    }
    if options is not None:
        optimize_options.update(options)
    X, expected_acquisition_value = optimize_acqf_list(
        acq_function_list=acq_function_list,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=50 * num_restarts if raw_samples is None else raw_samples,
        options=optimize_options,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        post_processing_func=rounding_func,
    )
    return X, expected_acquisition_value


def pareto_frontier_evaluator(
    model: TorchModel | None,
    objective_weights: Tensor,
    objective_thresholds: Tensor | None = None,
    X: Tensor | None = None,
    Y: Tensor | None = None,
    Yvar: Tensor | None = None,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return outcomes predicted to lie on a pareto frontier.

    Given a model and points to evaluate, use the model to predict which points
    lie on the Pareto frontier.

    Args:
        model: Model used to predict outcomes.
        objective_weights: A `m` tensor of values indicating the weight to put
            on different outcomes. For pareto frontiers only the sign matters.
        objective_thresholds:  A tensor containing thresholds forming a reference point
            from which to calculate pareto frontier hypervolume. Points that do not
            dominate the objective_thresholds contribute nothing to hypervolume.
        X: A `n x d` tensor of features to evaluate.
        Y: A `n x m` tensor of outcomes to use instead of predictions.
        Yvar: A `n x m x m` tensor of input covariances (NaN if unobserved).
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.

    Returns:
        3-element tuple containing

        - A `j x m` tensor of outcome on the pareto frontier. j is the number
            of frontier points.
        - A `j x m x m` tensor of predictive covariances.
            cov[j, m1, m2] is Cov[m1@j, m2@j].
        - A `j` tensor of the index of each frontier point in the input Y.
    """
    # TODO: better input validation, making more explicit whether we are using
    # model predictions or not
    if X is not None:
        Y, Yvar = none_throws(model).predict(X)
        # model.predict returns cpu tensors
        Y = Y.to(X.device)
        Yvar = Yvar.to(X.device)
    elif Y is None or Yvar is None:
        raise ValueError(
            "Requires `X` to predict or both `Y` and `Yvar` to select a subset of "
            "points on the pareto frontier."
        )

    # Apply objective_weights to outcomes and objective_thresholds.
    # If objective_thresholds is not None use a dummy tensor of zeros.
    (
        obj,
        weighted_objective_thresholds,
    ) = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=objective_weights,
        objective_thresholds=(
            objective_thresholds
            if objective_thresholds is not None
            else torch.zeros(
                objective_weights.shape,
                dtype=objective_weights.dtype,
                device=objective_weights.device,
            )
        ),
    )
    Y_obj = obj(Y)
    indx_frontier = torch.arange(Y.shape[0], dtype=torch.long, device=Y.device)

    # Filter Y, Yvar, Y_obj to items that dominate all objective thresholds
    if objective_thresholds is not None:
        objective_thresholds_mask = torch.all(
            Y_obj >= weighted_objective_thresholds, dim=1
        )
        Y = Y[objective_thresholds_mask]
        Yvar = Yvar[objective_thresholds_mask]
        Y_obj = Y_obj[objective_thresholds_mask]
        indx_frontier = indx_frontier[objective_thresholds_mask]

    # Get feasible points that do not violate outcome_constraints
    if outcome_constraints is not None:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # Handle NaNs in Y, if those elements are not part of the constraints.
        # By setting the unused elements to 0, we prevent them from marking
        # the whole constraint value as NaN and evaluating to infeasible.
        Y_cons = Y.clone()
        Y_cons[..., (outcome_constraints[0] == 0).all(dim=0)] = 0
        # pyre-ignore [16]
        feas = torch.stack([c(Y_cons) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        Y = Y[feas]
        Yvar = Yvar[feas]
        Y_obj = Y_obj[feas]
        indx_frontier = indx_frontier[feas]

    if Y.shape[0] == 0:
        # if there are no feasible points that are better than the reference point
        # return empty tensors
        return Y.cpu(), Yvar.cpu(), indx_frontier.cpu()

    # calculate pareto front with only objective outcomes:
    frontier_mask = is_non_dominated(Y_obj)

    # Apply masks
    Y_frontier = Y[frontier_mask]
    Yvar_frontier = Yvar[frontier_mask]
    indx_frontier = indx_frontier[frontier_mask]
    return Y_frontier.cpu(), Yvar_frontier.cpu(), indx_frontier.cpu()


def infer_objective_thresholds(
    model: Model,
    objective_weights: Tensor,  # objective_directions
    bounds: list[tuple[float, float]] | None = None,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    linear_constraints: tuple[Tensor, Tensor] | None = None,
    fixed_features: dict[int, float] | None = None,
    subset_idcs: Tensor | None = None,
    Xs: list[Tensor] | None = None,
    X_observed: Tensor | None = None,
    objective_thresholds: Tensor | None = None,
) -> Tensor:
    """Infer objective thresholds.

    This method uses the model-estimated Pareto frontier over the in-sample points
    to infer absolute (not relativized) objective thresholds.

    This uses a heuristic that sets the objective threshold to be a scaled nadir
    point, where the nadir point is scaled back based on the range of each
    objective across the current in-sample Pareto frontier.

    See `botorch.utils.multi_objective.hypervolume.infer_reference_point` for
    details on the heuristic.

    Args:
        model: A fitted botorch Model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights. These should not
            be subsetted.
        bounds: A list of (lower, upper) tuples for each column of X.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. These should not be subsetted.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        subset_idcs: The indices of the outcomes that are modeled by the
            provided model. If subset_idcs not None, this method infers
            whether the model is subsetted.
        Xs: A list of m (k_i x d) feature tensors X. Number of rows k_i can
            vary from i=1,...,m.
        X_observed: A `n x d`-dim tensor of in-sample points to use for
            determining the current in-sample Pareto frontier.
        objective_thresholds: Any known objective thresholds to pass to
            `infer_reference_point` heuristic. This should not be subsetted.
            If only a subset of the objectives have known thresholds, the
            remaining objectives should be NaN. If no objective threshold
            was provided, this can be `None`.

    Returns:
        A `m`-dim tensor of objective thresholds, where the objective
            threshold is `nan` if the outcome is not an objective.
    """
    if X_observed is None:
        if bounds is None:
            raise ValueError("bounds is required if X_observed is None.")
        elif Xs is None:
            raise ValueError("Xs is required if X_observed is None.")
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
    num_outcomes = objective_weights.shape[0]
    if subset_idcs is None:
        # Subset the model so that we only compute the posterior
        # over the relevant outcomes.
        # This is a no-op if the model is already only modeling
        # the relevant outcomes.
        subset_model_results = subset_model(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )
        model = subset_model_results.model
        objective_weights = subset_model_results.objective_weights
        outcome_constraints = subset_model_results.outcome_constraints
        subset_idcs = subset_model_results.indices
    else:
        objective_weights = objective_weights[subset_idcs]
        if outcome_constraints is not None:
            outcome_constraints = (
                outcome_constraints[0][:, subset_idcs],
                outcome_constraints[1],
            )
    with torch.no_grad():
        pred = _check_posterior_type(
            none_throws(model).posterior(none_throws(X_observed))
        ).mean

    if outcome_constraints is not None:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # pyre-ignore [16]
        feas = torch.stack([c(pred) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        pred = pred[feas]
    if pred.shape[0] == 0:
        raise AxError(NO_FEASIBLE_POINTS_MESSAGE)
    obj_mask = objective_weights.nonzero().view(-1)
    obj_weights_subset = objective_weights[obj_mask]
    obj = pred[..., obj_mask] * obj_weights_subset
    pareto_obj = obj[is_non_dominated(obj)]
    # If objective thresholds are provided, set max_ref_point accordingly.
    if objective_thresholds is not None:
        max_ref_point = objective_thresholds[obj_mask] * obj_weights_subset
    else:
        max_ref_point = None
    objective_thresholds = infer_reference_point(
        pareto_Y=pareto_obj,
        max_ref_point=max_ref_point,
        scale=0.1,
    )
    # multiply by objective weights to return objective thresholds in the
    # unweighted space
    objective_thresholds = objective_thresholds * obj_weights_subset
    full_objective_thresholds = torch.full(
        (num_outcomes,),
        float("nan"),
        dtype=objective_weights.dtype,
        device=objective_weights.device,
    )
    obj_idcs = subset_idcs[obj_mask]
    full_objective_thresholds[obj_idcs] = objective_thresholds.clone()
    return full_objective_thresholds


def _check_posterior_type(
    posterior: Posterior,
) -> GPyTorchPosterior | PosteriorList:
    """Check whether the posterior type is  `GPyTorchPosterior` or `PosteriorList`."""
    if isinstance(posterior, GPyTorchPosterior) or isinstance(posterior, PosteriorList):
        return posterior
    else:
        raise ValueError(
            f"Value was not of type GPyTorchPosterior or PosteriorList:\n{posterior}"
        )
