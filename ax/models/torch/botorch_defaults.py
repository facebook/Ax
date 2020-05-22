#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TConfig
from ax.models.model_utils import best_observed_point, get_observed
from ax.models.torch.utils import (
    HYPERSPHERE,
    SIMPLEX,
    _to_inequality_constraints,
    sample_hypersphere_positive_quadrant,
    sample_simplex,
)
from ax.models.torch_base import TorchModel
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, LinearMCObjective
from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


def get_and_fit_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    **kwargs: Any,
) -> GPyTorchModel:
    r"""Instantiates and fits a botorch ModelListGP using the given data.

    Args:
        Xs: List of X data, one tensor per outcome.
        Ys: List of Y data, one tensor per outcome.
        Yvars: List of observed variance of Ys.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.
        metric_names: Names of each outcome Y in Ys.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        refit_model: Flag for refitting model.

    Returns:
        A fitted GPyTorchModel.
    """
    if len(fidelity_features) > 0 and len(task_features) > 0:
        raise NotImplementedError(
            "Currently do not support MF-GP models with task_features!"
        )
    if len(fidelity_features) > 1:
        raise NotImplementedError(
            "Fidelity MF-GP models currently support only a single fidelity parameter!"
        )
    if len(task_features) > 1:
        raise NotImplementedError(
            f"This model only supports 1 task feature (got {task_features})"
        )
    elif len(task_features) == 1:
        task_feature = task_features[0]
    else:
        task_feature = None
    model = None
    if task_feature is None:
        if len(Xs) == 1:
            # Use single output, single task GP
            model = _get_model(
                X=Xs[0],
                Y=Ys[0],
                Yvar=Yvars[0],
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                **kwargs,
            )
        elif all(torch.equal(Xs[0], X) for X in Xs[1:]):
            # Use batched multioutput, single task GP
            Y = torch.cat(Ys, dim=-1)
            Yvar = torch.cat(Yvars, dim=-1)
            model = _get_model(
                X=Xs[0],
                Y=Y,
                Yvar=Yvar,
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                **kwargs,
            )
    # TODO: Is this equivalent an "else:" here?
    if model is None:
        # Use a ModelListGP
        models = [
            _get_model(X=X, Y=Y, Yvar=Yvar, task_feature=task_feature, **kwargs)
            for X, Y, Yvar in zip(Xs, Ys, Yvars)
        ]
        model = ModelListGP(*models)
    model.to(Xs[0])
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        if isinstance(model, ModelListGP):
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        else:
            # pyre-ignore: [16]
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_model(mll, bounds=bounds)
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
    variance = posterior.variance.cpu().detach().clamp_min(0)  # pyre-ignore
    cov = torch.diag_embed(variance)
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
        mc_samples: The number of MC samples to use (default: 512).
        qmc: If True, use qMC instead of MC (default: True).
        prune_baseline: If True, prune the baseline points for NEI (default: True).

    Returns:
        qNoisyExpectedImprovement: The instantiated acquisition function.
    """
    if X_observed is None:
        raise ValueError("There are no feasible observed points.")
    # Parse random_scalarization params
    objective_weights = _extract_random_scalarization_settings(
        objective_weights, outcome_constraints, **kwargs
    )
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
        prune_baseline=kwargs.get("prune_baseline", True),
        mc_samples=kwargs.get("mc_samples", 512),
        qmc=kwargs.get("qmc", True),
        # pyre-fixme[6]: Expected `Optional[int]` for 9th param but got
        #  `Union[float, int]`.
        seed=torch.randint(1, 10000, (1,)).item(),
    )


def scipy_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimizer using scipy's minimize module on a numpy-adpator.

    Args:
        acq_function: A botorch AcquisitionFunction.
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
        - In the case of joint optimization, a scalar tensor containing
          the joint acquisition value of the `n` points. In the case of
          sequential optimization, a `n`-dim tensor of conditional acquisition
          values, where `i`-th element is the expected acquisition value
          conditional on having observed candidates `0,1,...,i-1`.
    """

    num_restarts: int = kwargs.get("num_restarts", 20)
    raw_samples: int = kwargs.get("num_raw_samples", 50 * num_restarts)

    if kwargs.get("joint_optimization", False):
        sequential = False
    else:
        sequential = True
        # use SLSQP by default for small problems since it yields faster wall times
        if "method" not in kwargs:
            kwargs["method"] = "SLSQP"
    X, expected_acquisition_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        post_processing_func=rounding_func,
    )
    return X, expected_acquisition_value


def recommend_best_observed_point(
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    model_gen_options: Optional[TConfig] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    """
    A wrapper around `ax.models.model_utils.best_observed_point` for TorchModel
    that recommends a best point from previously observed points using either a
    "max_utility" or "feasible_threshold" strategy.

    Args:
        model: A TorchModel.
        bounds: A list of (lower, upper) tuples for each column of X.
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
        model_gen_options: A config dictionary that can contain
            model-specific options.
        target_fidelities: A map {feature_index: value} of fidelity feature
            column indices to their respective target fidelities. Used for
            multi-fidelity optimization.

    Returns:
        A d-array of the best point, or None if no feasible point was observed.
    """
    if target_fidelities:
        raise NotImplementedError(
            "target_fidelities not implemented for base BotorchModel"
        )

    x_best = best_observed_point(
        model=model,
        bounds=bounds,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
        options=model_gen_options,
    )
    if x_best is None:
        return None
    return x_best.to(dtype=model.dtype, device=torch.device("cpu"))


def recommend_best_out_of_sample_point(
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    model_gen_options: Optional[TConfig] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    """
    Identify the current best point by optimizing the posterior mean of the model.
    This is "out-of-sample" because it considers un-observed designs as well.

    Return None if no such point can be identified.

    Args:
        model: A TorchModel.
        bounds: A list of (lower, upper) tuples for each column of X.
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
        model_gen_options: A config dictionary that can contain
            model-specific options.
        target_fidelities: A map {feature_index: value} of fidelity feature
            column indices to their respective target fidelities. Used for
            multi-fidelity optimization.

    Returns:
        A d-array of the best point, or None if no feasible point exists.
    """
    options = model_gen_options or {}
    fixed_features = fixed_features or {}
    acf_options = options.get("acquisition_function_kwargs", {})
    optimizer_options = options.get("optimizer_kwargs", {})

    X_observed = get_observed(
        Xs=model.Xs,  # pyre-ignore: [16]
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
    )

    if hasattr(model, "_get_best_point_acqf"):
        acq_function, non_fixed_idcs = model._get_best_point_acqf(  # pyre-ignore: [16]
            X_observed=X_observed,
            objective_weights=objective_weights,
            mc_samples=acf_options.get("mc_samples", 512),
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            outcome_constraints=outcome_constraints,
            seed_inner=acf_options.get("seed_inner", None),
            qmc=acf_options.get("qmc", True),
        )
    else:
        raise RuntimeError("The model should implement _get_best_point_acqf.")

    inequality_constraints = _to_inequality_constraints(linear_constraints)
    # TODO: update optimizers to handle inequality_constraints
    # (including transforming constraints b/c of fixed features)
    if inequality_constraints is not None:
        raise UnsupportedError("Inequality constraints are not supported!")

    return_best_only = optimizer_options.get("return_best_only", True)
    bounds_ = torch.tensor(bounds, dtype=model.dtype, device=model.device)
    bounds_ = bounds_.transpose(-1, -2)
    if non_fixed_idcs is not None:
        bounds_ = bounds_[..., non_fixed_idcs]

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds_,
        q=1,
        num_restarts=optimizer_options.get("num_restarts", 60),
        raw_samples=optimizer_options.get("raw_samples", 1024),
        inequality_constraints=inequality_constraints,
        fixed_features=None,  # handled inside the acquisition function
        options={
            "batch_limit": optimizer_options.get("batch_limit", 8),
            "maxiter": optimizer_options.get("maxiter", 200),
            "nonnegative": optimizer_options.get("nonnegative", False),
            "method": "L-BFGS-B",
        },
        return_best_only=return_best_only,
    )
    rec_point = candidates.detach().cpu()
    if isinstance(acq_function, FixedFeatureAcquisitionFunction):
        rec_point = acq_function._construct_X_full(rec_point)
    if return_best_only:
        rec_point = rec_point.view(-1)
    return rec_point


def _get_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: Optional[int] = None,
    fidelity_features: Optional[List[int]] = None,
    **kwargs: Any,
) -> GPyTorchModel:
    """Instantiate a model of type depending on the input data.

    Args:
        X: A `n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        task_feature: The index of the column pertaining to the task feature
            (if present).
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A GPyTorchModel (unfitted).
    """
    Yvar = Yvar.clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
    is_nan = torch.isnan(Yvar)
    any_nan_Yvar = torch.any(is_nan)
    all_nan_Yvar = torch.all(is_nan)
    if any_nan_Yvar and not all_nan_Yvar:
        if task_feature:
            # TODO (jej): Replace with inferred noise before making perf judgements.
            Yvar[Yvar != Yvar] = MIN_OBSERVED_NOISE_LEVEL
        else:
            raise ValueError(
                "Mix of known and unknown variances indicates valuation function "
                "errors. Variances should all be specified, or none should be."
            )
    if fidelity_features is None:
        fidelity_features = []
    if len(fidelity_features) == 0:
        # only pass linear_truncated arg if there are fidelities
        kwargs = {k: v for k, v in kwargs.items() if k != "linear_truncated"}
    if len(fidelity_features) > 0:
        if task_feature:
            raise NotImplementedError(  # pragma: no cover
                "multi-task multi-fidelity models not yet available"
            )
        # at this point we can assume that there is only a single fidelity parameter
        gp = SingleTaskMultiFidelityGP(
            train_X=X, train_Y=Y, data_fidelity=fidelity_features[0], **kwargs
        )
    elif task_feature is None and all_nan_Yvar:
        gp = SingleTaskGP(train_X=X, train_Y=Y, **kwargs)
    elif task_feature is None:
        gp = FixedNoiseGP(train_X=X, train_Y=Y, train_Yvar=Yvar, **kwargs)
    elif all_nan_Yvar:
        gp = MultiTaskGP(train_X=X, train_Y=Y, task_feature=task_feature, **kwargs)
    else:
        gp = FixedNoiseMultiTaskGP(
            train_X=X, train_Y=Y, train_Yvar=Yvar, task_feature=task_feature, **kwargs
        )
    return gp


def _extract_random_scalarization_settings(
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    **kwargs: Any,
) -> Tensor:
    """Generate a random weighting based on scalarization settings."""
    use_random_scalarization = kwargs.get("random_scalarization", False)
    random_weights = None
    if use_random_scalarization:
        # Pareto Optimization incompatible with outcome constraints.
        if outcome_constraints is not None:
            raise ValueError(
                "Random scalarization for pareto frontier exploration "
                "is incompatible with outcome constraints. Remove one."
            )
        # Set distribution and sample weights.
        distribution = kwargs.get("random_scalarization_distribution", SIMPLEX)
        if distribution == SIMPLEX:
            random_weights = sample_simplex(len(objective_weights))
        elif distribution == HYPERSPHERE:
            random_weights = sample_hypersphere_positive_quadrant(
                len(objective_weights)
            )

    if random_weights is not None:
        objective_weights = torch.mul(objective_weights, random_weights)
    return objective_weights
