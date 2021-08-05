#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
References

.. [Daulton2020qehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Differentiable Expected Hypervolume
    Improvement for Parallel Multi-Objective Bayesian Optimization. Advances in Neural
    Information Processing Systems 33, 2020.

.. [Daulton2021nehvi]
    S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of
    Multiple Noisy Objectives. ArXiv, 2021.

"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.exceptions.core import AxError
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    subset_model,
)
from ax.models.torch.utils import (  # noqa F40
    _to_inequality_constraints,
    get_outcome_constraint_transforms,
    predict_from_model,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.utils import (
    get_acquisition_function,
)
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
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
        Optional[Tuple[Tensor, Tensor]],
    ],
    Tuple[Tensor, Tensor, Tensor],
]


def get_default_frontier_evaluator() -> TFrontierEvaluator:
    return pareto_frontier_evaluator


def get_weighted_mc_objective_and_objective_thresholds(
    objective_weights: Tensor, objective_thresholds: Tensor
) -> Tuple[WeightedMCMultiOutputObjective, Tensor]:
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
    # pyre-ignore [16]
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
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
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
        mc_samples: The number of MC samples to use (default: 512).
        qmc: If True, use qMC instead of MC (default: True).
        prune_baseline: If True, prune the baseline points for NEI (default: True).
        chebyshev_scalarization: Use augmented Chebyshev scalarization.

    Returns:
        qNoisyExpectedHyperVolumeImprovement: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
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
    return get_acquisition_function(
        acquisition_function_name="qNEHVI",
        model=model,
        objective=objective,  # pyre-ignore [6]
        X_observed=X_observed,
        X_pending=X_pending,
        constraints=cons_tfs,
        prune_baseline=kwargs.get("prune_baseline", True),
        mc_samples=kwargs.get("mc_samples", DEFAULT_EHVI_MC_SAMPLES),
        alpha=kwargs.get(
            "alpha", get_default_partitioning_alpha(num_objectives=num_objectives)
        ),
        qmc=kwargs.get("qmc", True),
        # pyre-fixme[6]: Expected `Optional[int]` for 11th param but got
        #  `Union[float, int]`.
        seed=torch.randint(1, 10000, (1,)).item(),
        ref_point=objective_thresholds.tolist(),
        marginalize_dim=kwargs.get("marginalize_dim"),
        match_right_most_batch_dim=kwargs.get("match_right_most_batch_dim", False),
    )


def get_EHVI(
    model: Model,
    objective_weights: Tensor,
    objective_thresholds: Tensor,
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
        qmc: If True, use qMC instead of MC (default: True).

    Returns:
        qExpectedHypervolumeImprovement: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
    # construct Objective module
    (
        objective,
        objective_thresholds,
    ) = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=objective_weights, objective_thresholds=objective_thresholds
    )
    with torch.no_grad():
        Y = model.posterior(X_observed).mean
    # For EHVI acquisition functions we pass the constraint transform directly.
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
    num_objectives = objective_thresholds.shape[0]
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
        alpha=kwargs.get(
            "alpha", get_default_partitioning_alpha(num_objectives=num_objectives)
        ),
        # pyre-fixme[6]: Expected `Optional[int]` for 10th param but got
        #  `Union[float, int]`.
        seed=torch.randint(1, 10000, (1,)).item(),
        ref_point=objective_thresholds.tolist(),
        Y=Y,
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
    num_restarts: int = kwargs.pop(Keys.NUM_RESTARTS, 20)
    raw_samples: int = kwargs.pop(Keys.RAW_SAMPLES, 50 * num_restarts)

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
    objective_thresholds: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    Y: Optional[Tensor] = None,
    Yvar: Optional[Tensor] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return outcomes predicted to lie on a pareto frontier.

    Given a model and a points to evaluate use the model to predict which points
    lie on the pareto frontier.

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
    if X is not None:
        Y, Yvar = model.predict(X)
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
        objective_thresholds_mask = (Y_obj >= weighted_objective_thresholds).all(dim=1)
        Y = Y[objective_thresholds_mask]
        Yvar = Yvar[objective_thresholds_mask]
        Y_obj = Y_obj[objective_thresholds_mask]
        indx_frontier = indx_frontier[objective_thresholds_mask]

    # Get feasible points that do not violate outcome_constraints
    if outcome_constraints is not None:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # pyre-ignore [16]
        feas = torch.stack([c(Y) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
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
    bounds: Optional[List[Tuple[float, float]]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    subset_idcs: Optional[Tensor] = None,
    Xs: Optional[List[Tensor]] = None,
    X_observed: Optional[Tensor] = None,
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
        # check if only a subset of outcomes are modeled
        nonzero = objective_weights != 0
        if outcome_constraints is not None:
            A, _ = outcome_constraints
            nonzero = nonzero | torch.any(A != 0, dim=0)
        expected_subset_idcs = nonzero.nonzero().view(-1)  # pyre-ignore [16]
        if model.num_outputs > expected_subset_idcs.numel():
            # subset the model so that we only compute the posterior
            # over the relevant outcomes
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
            # model is already subsetted.
            subset_idcs = expected_subset_idcs
            # subset objective weights and outcome constraints
            objective_weights = objective_weights[subset_idcs]
            if outcome_constraints is not None:
                outcome_constraints = (
                    outcome_constraints[0][:, subset_idcs],
                    outcome_constraints[1],
                )
    else:
        objective_weights = objective_weights[subset_idcs]
        if outcome_constraints is not None:
            outcome_constraints = (
                outcome_constraints[0][:, subset_idcs],
                outcome_constraints[1],
            )
    with torch.no_grad():
        pred = not_none(model).posterior(not_none(X_observed)).mean
    if outcome_constraints is not None:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # pyre-ignore [16]
        feas = torch.stack([c(pred) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        pred = pred[feas]
    if pred.shape[0] == 0:
        raise AxError("There are no feasible observed points.")
    obj_mask = objective_weights.nonzero().view(-1)
    obj_weights_subset = objective_weights[obj_mask]
    obj = pred[..., obj_mask] * obj_weights_subset
    pareto_obj = obj[is_non_dominated(obj)]
    objective_thresholds = infer_reference_point(
        pareto_Y=pareto_obj,
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
    full_objective_thresholds[subset_idcs] = objective_thresholds.clone()
    return full_objective_thresholds
