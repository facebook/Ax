#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.models.torch.utils import (  # noqa F40
    _to_inequality_constraints,
    get_outcome_constraint_transforms,
    predict_from_model,
)
from ax.models.torch_base import TorchModel
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.utils import get_acquisition_function
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import squeeze_last_dim
from torch import Tensor


DEFAULT_EHVI_MC_SAMPLES = 128


def _get_weighted_mc_objective_and_ref_point(
    objective_weights: Tensor, ref_point: Tensor
) -> Tuple[WeightedMCMultiOutputObjective, Tensor]:
    r"""Construct weighted objective and apply the weights to the reference point.

    Args:
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        ref_point: A reference point from which to calculate pareto frontier
            hypervolume. Points that do not dominate the ref_point contribute
            nothing to hypervolume.

    Returns:
        A two-element tuple with the objective and reference point:

            - The objective
            - The reference point

    """
    # pyre-ignore [16]
    nonzero_idcs = objective_weights.nonzero(as_tuple=False).view(-1)
    objective_weights = objective_weights[nonzero_idcs]
    objective = WeightedMCMultiOutputObjective(
        weights=objective_weights, outcomes=nonzero_idcs.tolist()
    )
    ref_point = torch.mul(ref_point, objective_weights)
    return objective, ref_point


def get_EHVI(
    model: Model,
    objective_weights: Tensor,
    ref_point: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a qExpectedHyperVolumeImprovement acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        ref_point: A reference point from which to calculate pareto frontier
            hypervolume. Points that do not dominate the ref_point contribute
            nothing to hypervolume.
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
        qmc: If True, use qMC instead of MC (default: True).

    Returns:
        qExpectedHypervolumeImprovement: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
    # construct Objective module
    objective, ref_point = _get_weighted_mc_objective_and_ref_point(
        objective_weights=objective_weights, ref_point=ref_point
    )
    if "Ys" not in kwargs:
        raise ValueError("Expected Hypervolume Improvement requires Ys argument")
    Y_tensor = squeeze_last_dim(torch.stack(kwargs.get("Ys")).transpose(0, 1))
    # For EHVI acquisition functions we pass the constraint transform directly.
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)

    return get_acquisition_function(
        acquisition_function_name="qEHVI",
        model=model,
        # TODO (jej): Fix pyre error below by restructuring class hierarchy.
        # pyre-fixme[6]: Expected `botorch.acquisition.objective.
        #  MCAcquisitionObjective` for 3rd parameter `objective` to call
        #  `get_acquisition_function` but got `IdentityMCMultiOutputObjective`.
        objective=objective,
        X_observed=X_observed,
        X_pending=X_pending,
        constraints=cons_tfs,
        mc_samples=kwargs.get("mc_samples", DEFAULT_EHVI_MC_SAMPLES),
        qmc=kwargs.get("qmc", True),
        seed=torch.randint(1, 10000, (1,)).item(),
        ref_point=ref_point.tolist(),
        Y=Y_tensor,
    )


# TODO (jej): rewrite optimize_acqf wrappers to avoid duplicate code.
def scipy_optimizer_list(
    acq_function_list: List[AcquisitionFunction],
    bounds: Tensor,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
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
    num_restarts: int = kwargs.get("num_restarts", 20)
    raw_samples: int = kwargs.get("num_raw_samples", 50 * num_restarts)

    # use SLSQP by default for small problems since it yields faster wall times
    if "method" not in kwargs:
        kwargs["method"] = "SLSQP"
    X, expected_acquisition_value = optimize_acqf_list(
        acq_function_list=acq_function_list,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        post_processing_func=rounding_func,
    )
    return X, expected_acquisition_value


def pareto_frontier_evaluator(
    model: TorchModel,
    objective_weights: Tensor,
    ref_point: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    Y: Optional[Tensor] = None,
    Yvar: Optional[Tensor] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Return outcomes predicted to lie on a pareto frontier.

    Given a model and a points to evaluate use the model to predict which points
    lie on the pareto frontier.

    Args:
        model: Model used to predict outcomes.
        objective_weights: A `m` tensor of values indicating the weight to put
            on different outcomes. For pareto frontiers only the sign matters.
        ref_point: reference point for hypervolume computation.
        X: A `n x d` tensor of features to evaluate.
        Y: A `n x m` tensor of outcomes to use instead of predictions.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.

    Returns:
        2-element tuple containing

        - A `j x m` tensor of outcome on the pareto frontier. j is the number
            of frontier points.
        - A `j x m x m` tensor of predictive covariances.
            cov[j, m1, m2] is Cov[m1@j, m2@j].
    """
    if X is not None:
        Y, Yvar = model.predict(X)
    elif Y is None or Yvar is None:
        raise ValueError(
            "Requires `X` to predict or both `Y` and `Yvar` to select a subset of "
            "points on the pareto frontier."
        )

    # Apply objective_weights to outcomes and reference_point.
    # If ref_point is not None use a dummy tensor of zeros.
    obj, weighted_ref_point = _get_weighted_mc_objective_and_ref_point(
        objective_weights=objective_weights,
        ref_point=(
            ref_point if ref_point is not None else torch.zeros(objective_weights.shape)
        ),
    )
    Y_obj = obj(Y)

    # Filter Y, Yvar, Y_obj to items that dominate the reference point
    if ref_point is not None:
        ref_point_mask = (Y_obj >= weighted_ref_point).all(dim=1)
        Y = Y[ref_point_mask]
        Yvar = Yvar[ref_point_mask]
        Y_obj = Y_obj[ref_point_mask]

    # Get feasible points that do not violate outcome_constraints
    if outcome_constraints is not None:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # pyre-ignore [16]
        feas = torch.stack([c(Y) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        Y = Y[feas]
        Yvar = Yvar[feas]
        Y_obj = Y_obj[feas]

    # calculate pareto front with only objective outcomes:
    frontier_mask = is_non_dominated(Y_obj)

    # Apply masks
    Y_frontier = Y[frontier_mask]
    Yvar_frontier = Yvar[frontier_mask]
    return Y_frontier, Yvar_frontier
