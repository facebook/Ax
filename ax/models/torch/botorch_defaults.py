#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, LinearMCObjective
from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model import Model
from botorch.models.multi_output_gp_regression import MultiOutputGP
from botorch.optim.optimize import joint_optimize, sequential_optimize
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def get_and_fit_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    state_dict: Optional[Dict[str, Tensor]] = None,
    **kwargs: Any,
) -> MultiOutputGP:
    r"""Instantiates and fits a botorch MultiOutputGP using the given data.

    Args:
        Xs: List of X data, one tensor per outcome
        Ys: List of Y data, one tensor per outcome
        Yvars: List of observed variance of Ys.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.

    Returns:
        A fitted MultiOutputGP.
    """
    models = [_get_model(X=X, Y=Y, Yvar=Yvar) for X, Y, Yvar in zip(Xs, Ys, Yvars)]
    model = MultiOutputGP(gp_models=models)
    model.to(dtype=Xs[0].dtype, device=Xs[0].device)  # pyre-ignore
    if state_dict is None:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_model(mll, bounds=bounds)
    else:
        model.load_state_dict(state_dict)
    return model


def predict_from_model(model: Model, X: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Predicts outcomes given a model and input tensor.

    Args:
        model: A botorch Model.
        X: A `n x d` tensor of input parameters.

    Returns:
        Tensor: The predicted posterior mean as an `n x o`-dim tensor.
        Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
    """
    with torch.no_grad():
        posterior = model.posterior(X)
    mean = posterior.mean.cpu().detach()
    # TODO: Allow Posterior to (optionally) return the full covariance matrix
    variance = posterior.variance.cpu().detach()
    cov = variance.unsqueeze(-1) * torch.eye(variance.shape[-1], dtype=variance.dtype)
    return mean, cov


def get_NEI(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a qNoisyExpectedImprovement acquisition function.

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
        mc_samples: The number of MC samples to use (default: 500).
        qmc: If True, use qMC instead of MC (default: True).

    Returns:
        qNoisyExpectedImprovement: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
    # construct Objective module
    if outcome_constraints is None:
        objective = LinearMCObjective(weights=objective_weights)
    else:
        obj_tf = get_objective_weights_transform(objective_weights)
        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        X_observed = torch.as_tensor(X_observed)
        inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=obj_tf)
        objective = ConstrainedMCObjective(
            objective=obj_tf, constraints=con_tfs or [], infeasible_cost=inf_cost
        )
    return get_acquisition_function(
        acquisition_function_name="qNEI",
        model=model,
        objective=objective,
        X_observed=X_observed,
        X_pending=X_pending,
        mc_samples=kwargs.get("mc_samples", 500),
        qmc=kwargs.get("qmc", True),
        seed=torch.randint(1, 10000, (1,)).item(),
    )


def scipy_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    **kwargs: Any,
) -> Tensor:
    r"""Optimizer using scipy's minimize module on a numpy-adpator.

    Args:
        acq_function: A botorch AcquisitionFunction.
        bounds: A `2 x d`-dim tensor, where `bounds[0]` (`bounds[1]`) are the
            lower (upper) bounds of the feasible hyperrectangle.
        n: The number of candidates to generate.
        fixed_features: A map {feature_index: value} for features that should
            be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).

    Returns:
        Tensor: A `n x d`-dim tensor of generated candidates.
    """
    num_restarts: int = kwargs.get("num_restarts", 20)
    raw_samples: int = kwargs.get("num_raw_samples", 50 * num_restarts)
    if kwargs.get("joint_optimization", False):
        optimize = joint_optimize
    else:
        optimize = sequential_optimize
    return optimize(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        fixed_features=fixed_features,
        post_processing_func=rounding_func,
    )


def _get_model(X: Tensor, Y: Tensor, Yvar: Tensor) -> Model:
    """Instantiate a model of type depending on the input data."""
    Yvar = Yvar.clamp_min_(MIN_OBSERVED_NOISE_LEVEL)  # pyre-ignore: [16]
    gp = FixedNoiseGP(train_X=X, train_Y=Y.view(-1), train_Y_se=Yvar.view(-1).sqrt())
    return gp.to(X)  # pyre-ignore: [7]
