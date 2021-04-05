#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Optional, Tuple

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.model import Model
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from torch import Tensor


def get_PosteriorMean(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a PosteriorMean acquisition function.

    Note: If no OutcomeConstraints given, return an analytic acquisition
    function. This requires {optimizer_kwargs: {joint_optimization: True}} or an
    optimizer that does not assume pending point support.

    Args:
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

    Returns:
        PosteriorMean: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
    # construct Objective module
    if kwargs.get("chebyshev_scalarization", False):
        with torch.no_grad():
            Y = model.posterior(X_observed).mean
        obj_tf = get_chebyshev_scalarization(weights=objective_weights, Y=Y)
    else:
        obj_tf = get_objective_weights_transform(objective_weights)

    def obj_fn(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return obj_tf(samples)

    if outcome_constraints is None:
        objective = GenericMCObjective(objective=obj_fn)
    else:
        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=obj_fn)
        objective = ConstrainedMCObjective(
            objective=obj_fn, constraints=con_tfs or [], infeasible_cost=inf_cost
        )
    # Use qSimpleRegret, not analytic posterior, to handle arbitrary objective fns.
    acq_func = qSimpleRegret(model, objective=objective)
    return acq_func
