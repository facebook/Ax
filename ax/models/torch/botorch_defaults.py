#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TConfig
from ax.models.model_utils import best_observed_point, get_observed
from ax.models.torch.utils import (  # noqa F401
    _to_inequality_constraints,
    predict_from_model,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.constants import Keys
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.transforms.input import Warp
from botorch.optim.optimize import optimize_acqf
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
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
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    **kwargs: Any,
) -> GPyTorchModel:
    r"""Instantiates and fits a botorch GPyTorchModel using the given data.
    N.B. Currently, the logic for choosing ModelListGP vs other models is handled
    using if-else statements in lines 96-137. In the future, this logic should be
    taken care of by modular botorch.

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

    # TODO: Better logic for deciding when to use a ModelListGP. Currently the
    # logic is unclear. The two cases in which ModelListGP is used are
    # (i) the training inputs (Xs) are not the same for the different outcomes, and
    # (ii) a multi-task model is used

    if task_feature is None:
        if len(Xs) == 1:
            # Use single output, single task GP
            model = _get_model(
                X=Xs[0],
                Y=Ys[0],
                Yvar=Yvars[0],
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                use_input_warping=use_input_warping,
                **kwargs,
            )
        elif all(torch.equal(Xs[0], X) for X in Xs[1:]) and not use_input_warping:
            # Use batched multioutput, single task GP
            # Require using a ModelListGP if using input warping
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
        if task_feature is None:
            models = [
                _get_model(
                    X=X, Y=Y, Yvar=Yvar, use_input_warping=use_input_warping, **kwargs
                )
                for X, Y, Yvar in zip(Xs, Ys, Yvars)
            ]
        else:
            # use multi-task GP
            mtgp_rank_dict = kwargs.pop("multitask_gp_ranks", {})
            # assembles list of ranks associated with each metric
            if len({len(Xs), len(Ys), len(Yvars), len(metric_names)}) > 1:
                raise ValueError(
                    "Lengths of Xs, Ys, Yvars, and metric_names must match. Your "
                    f"inputs have lengths {len(Xs)}, {len(Ys)}, {len(Yvars)}, and "
                    f"{len(metric_names)}, respectively."
                )
            mtgp_rank_list = [
                mtgp_rank_dict.get(metric, None) for metric in metric_names
            ]
            models = [
                _get_model(
                    X=X,
                    Y=Y,
                    Yvar=Yvar,
                    task_feature=task_feature,
                    rank=mtgp_rank,
                    use_input_warping=use_input_warping,
                    **kwargs,
                )
                for X, Y, Yvar, mtgp_rank in zip(Xs, Ys, Yvars, mtgp_rank_list)
            ]
        model = ModelListGP(*models)
    model.to(Xs[0])
    if state_dict is not None:
        # pyre-fixme[6]: Expected `OrderedDict[typing.Any, typing.Any]` for 1st
        #  param but got `Dict[str, Tensor]`.
        model.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        if use_loocv_pseudo_likelihood:
            mll_cls = LeaveOneOutPseudoLikelihood
        else:
            mll_cls = ExactMarginalLogLikelihood
        if isinstance(model, ModelListGP):
            mll = SumMarginalLogLikelihood(model.likelihood, model, mll_cls=mll_cls)
        else:
            mll = mll_cls(model.likelihood, model)
        mll = fit_gpytorch_model(mll, bounds=bounds)
    return model


def get_NEI(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a qNoisyExpectedImprovement acquisition function."""
    return _get_acqusition_func(
        model=model,
        acquisition_function_name="qNEI",
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        X_pending=X_pending,
        **kwargs,
    )


def _get_acqusition_func(
    model: Model,
    acquisition_function_name: str,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    r"""Instantiates a acquisition function.

    Args:
        model: The underlying model which the acqusition function uses
            to estimate acquisition values of candidates.
        acquisition_function_name: Name of the acquisition function.
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
        The instantiated acquisition function.
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

    def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return obj_tf(samples)

    if outcome_constraints is None:
        objective = GenericMCObjective(objective=objective)
    else:
        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=objective)
        objective = ConstrainedMCObjective(
            objective=objective, constraints=con_tfs or [], infeasible_cost=inf_cost
        )
    return get_acquisition_function(
        acquisition_function_name=acquisition_function_name,
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
        marginalize_dim=kwargs.get("marginalize_dim"),
    )


def scipy_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
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
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) == rhs`
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
    num_restarts: int = kwargs.pop(Keys.NUM_RESTARTS, 20)
    raw_samples: int = kwargs.pop(Keys.RAW_SAMPLES, 50 * num_restarts)

    if kwargs.get("joint_optimization", False):
        sequential = False
    else:
        sequential = True
    X, expected_acquisition_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
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
    use_input_warping: bool = False,
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
    Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)  # pyre-ignore[16]
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
    if use_input_warping:
        warp_tf = get_warping_transform(
            d=X.shape[-1],
            task_feature=task_feature,
            batch_shape=X.shape[:-2],  # pyre-ignore [6]
        )
    else:
        warp_tf = None
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
            train_X=X,
            train_Y=Y,
            data_fidelity=fidelity_features[0],
            input_transform=warp_tf,
            **kwargs,
        )
    elif task_feature is None and all_nan_Yvar:
        gp = SingleTaskGP(train_X=X, train_Y=Y, input_transform=warp_tf, **kwargs)
    elif task_feature is None:
        gp = FixedNoiseGP(
            train_X=X, train_Y=Y, train_Yvar=Yvar, input_transform=warp_tf, **kwargs
        )
    else:
        # instantiate multitask GP
        all_tasks, _, _ = MultiTaskGP.get_all_tasks(X, task_feature)
        num_tasks = len(all_tasks)
        prior_dict = kwargs.get("prior")
        prior = None
        if prior_dict is not None:
            prior_type = prior_dict.get("type", None)
            if issubclass(prior_type, LKJCovariancePrior):
                sd_prior = prior_dict.get("sd_prior", GammaPrior(1.0, 0.15))
                sd_prior._event_shape = torch.Size([num_tasks])
                eta = prior_dict.get("eta", 0.5)
                if not isinstance(eta, float) and not isinstance(eta, int):
                    raise ValueError(f"eta must be a real number, your eta was {eta}")
                prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

            else:
                raise NotImplementedError(
                    "Currently only LKJ prior is supported,"
                    f"your prior type was {prior_type}."
                )

        if all_nan_Yvar:
            gp = MultiTaskGP(
                train_X=X,
                train_Y=Y,
                task_feature=task_feature,
                rank=kwargs.get("rank"),
                task_covar_prior=prior,
                input_transform=warp_tf,
            )
        else:
            gp = FixedNoiseMultiTaskGP(
                train_X=X,
                train_Y=Y,
                train_Yvar=Yvar,
                task_feature=task_feature,
                rank=kwargs.get("rank"),
                task_covar_prior=prior,
                input_transform=warp_tf,
            )
    return gp


def get_warping_transform(
    d: int,
    batch_shape: Optional[torch.Size] = None,
    task_feature: Optional[int] = None,
) -> Warp:
    """Construct input warping transform.

    Args:
        d: The dimension of the input, including task features
        batch_shape: The batch_shape of the model
        task_feature: The index of the task feature

    Returns:
        The input warping transform.
    """
    indices = list(range(d))
    # apply warping to all non-task features, including fidelity features
    if task_feature is not None:
        del indices[task_feature]
    # Note: this currently uses the same warping functions for all tasks
    tf = Warp(
        indices=indices,
        # prior with a median of 1
        concentration1_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
        concentration0_prior=LogNormalPrior(0.0, 0.75 ** 0.5),
        batch_shape=batch_shape,
    )
    return tf
