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

import torch
from ax.exceptions.core import AxError
from ax.generators.torch.utils import extract_objectives, subset_model
from ax.generators.torch_base import TorchGenerator
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
from pyre_extensions import none_throws
from torch import Tensor


NO_FEASIBLE_POINTS_MESSAGE = (
    " Cannot infer objective thresholds due to no observed feasible points. "
    " This likely means that one or more outcome constraints is set too strictly.  "
    " Consider adding thresholds to your objectives to bypass this error."
)


def get_weighted_mc_objective(
    objective_weights: Tensor,
) -> WeightedMCMultiOutputObjective:
    r"""Construct a weighted MC multi-output objective from objective weights.

    Args:
        objective_weights: A ``(n_objectives, n_outcomes)`` tensor of objective
            weights.

    Returns:
        A WeightedMCMultiOutputObjective.
    """
    outcome_indices, weights = extract_objectives(objective_weights)
    return WeightedMCMultiOutputObjective(
        weights=weights, outcomes=outcome_indices.tolist()
    )


def pareto_frontier_evaluator(
    model: TorchGenerator | None,
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
        objective_weights: A ``(n_objectives, m)`` tensor of objective
            weights. For pareto frontiers only the sign matters.
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

    # Apply objective_weights to outcomes.
    obj = get_weighted_mc_objective(objective_weights=objective_weights)
    Y_obj = obj(Y)
    indx_frontier = torch.arange(Y.shape[0], dtype=torch.long, device=Y.device)

    # Filter Y, Yvar, Y_obj to items that dominate all objective thresholds
    if objective_thresholds is not None:
        objective_thresholds_mask = torch.all(Y_obj >= objective_thresholds, dim=1)
        Y = Y[objective_thresholds_mask]
        Yvar = Yvar[objective_thresholds_mask]
        Y_obj = Y_obj[objective_thresholds_mask]
        indx_frontier = indx_frontier[objective_thresholds_mask]

    # Get feasible points that do not violate outcome_constraints
    if outcome_constraints is not None:
        cons_tfs = none_throws(get_outcome_constraint_transforms(outcome_constraints))
        # Handle NaNs in Y, if those elements are not part of the constraints.
        # By setting the unused elements to 0, we prevent them from marking
        # the whole constraint value as NaN and evaluating to infeasible.
        Y_cons = Y.clone()
        Y_cons[..., (outcome_constraints[0] == 0).all(dim=0)] = 0
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
    objective_weights: Tensor,
    X_observed: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    subset_idcs: Tensor | None = None,
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
        objective_weights: A ``(n_objectives, n_outcomes)`` tensor of objective
            weights. These should not be subsetted.
        X_observed: A ``n x d``-dim tensor of in-sample points to use for
            determining the current in-sample Pareto frontier.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. These should not be subsetted.
        subset_idcs: The indices of the outcomes that are modeled by the
            provided model. If subset_idcs is not None, this method infers
            whether the model is subsetted.
        objective_thresholds: A ``(n_objectives,)`` tensor of maximization-
            aligned objective thresholds. NaN entries indicate thresholds to
            infer. If no objective thresholds are provided, this can be
            ``None``.

    Returns:
        A ``(n_objectives,)`` tensor of maximization-aligned objective
        thresholds.
    """
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
        objective_weights = objective_weights[:, subset_idcs]
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
        cons_tfs = none_throws(get_outcome_constraint_transforms(outcome_constraints))
        feas = torch.stack([c(pred) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        pred = pred[feas]
    if pred.shape[0] == 0:
        raise AxError(NO_FEASIBLE_POINTS_MESSAGE)
    obj_indices, obj_weights_subset = extract_objectives(objective_weights)
    obj_mask = torch.tensor(obj_indices, device=objective_weights.device)
    # Convert predictions to maximization-aligned objective values.
    obj = pred[..., obj_mask] * obj_weights_subset
    pareto_obj = obj[is_non_dominated(obj)]
    # Input thresholds are already (n_objectives,) and maximization-aligned.
    max_ref_point = objective_thresholds
    return infer_reference_point(
        pareto_Y=pareto_obj,
        max_ref_point=max_ref_point,
        scale=0.1,
    )


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
