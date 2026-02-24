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
from ax.generators.torch.utils import subset_model
from ax.generators.torch.utils import (
    collapse_objective_weights,
    get_outcome_mask,
)
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
    collapsed = collapse_objective_weights(objective_weights)
    nonzero_idcs = collapsed.nonzero(as_tuple=False).view(-1)
    collapsed = collapsed[nonzero_idcs]
    objective_thresholds = objective_thresholds[nonzero_idcs]
    objective = WeightedMCMultiOutputObjective(
        weights=collapsed, outcomes=nonzero_idcs.tolist()
    )
    objective_thresholds = torch.mul(objective_thresholds, collapsed)
    return objective, objective_thresholds


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
    # Guard: if objective_weights is 2D, collapse to 1D for this function.
    if objective_weights.dim() == 2:
        objective_weights = collapse_objective_weights(objective_weights)
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
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights. These should not
            be subsetted.
        X_observed: A `n x d`-dim tensor of in-sample points to use for
            determining the current in-sample Pareto frontier.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. These should not be subsetted.
        subset_idcs: The indices of the outcomes that are modeled by the
            provided model. If subset_idcs not None, this method infers
            whether the model is subsetted.
        objective_thresholds: Any known objective thresholds to pass to
            `infer_reference_point` heuristic. This should not be subsetted.
            If only a subset of the objectives have known thresholds, the
            remaining objectives should be NaN. If no objective threshold
            was provided, this can be `None`.

    Returns:
        A `m`-dim tensor of objective thresholds, where the objective
            threshold is `nan` if the outcome is not an objective.
    """
    num_outcomes = objective_weights.shape[1]
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
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # pyre-ignore [16]
        feas = torch.stack([c(pred) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        pred = pred[feas]
    if pred.shape[0] == 0:
        raise AxError(NO_FEASIBLE_POINTS_MESSAGE)
    obj_mask = get_outcome_mask(objective_weights).nonzero().view(-1)
    obj_weights_subset = collapse_objective_weights(objective_weights)[obj_mask]
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
