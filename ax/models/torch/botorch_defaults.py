#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
from collections.abc import Callable
from copy import deepcopy
from random import randint
from typing import Any, Protocol

import torch
from ax.models.model_utils import best_observed_point
from ax.models.torch_base import TorchModel
from ax.models.types import TConfig
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.acquisition.utils import get_infeasible_cost
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Warp
from botorch.optim.optimize import optimize_acqf
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.transforms import is_ensemble
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import Prior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-6
NO_OBSERVED_POINTS_MESSAGE = (
    "There are no observed points meeting all parameter "
    "constraints or have all necessary metrics attached."
)


def _construct_model(
    task_feature: int | None,
    Xs: list[Tensor],
    Ys: list[Tensor],
    Yvars: list[Tensor],
    fidelity_features: list[int],
    metric_names: list[str],
    use_input_warping: bool = False,
    prior: dict[str, Any] | None = None,
    *,
    multitask_gp_ranks: dict[str, Prior | float] | None = None,
    **kwargs: Any,
) -> GPyTorchModel:
    """
    Figures out how to call `_get_model` depending on inputs. Used by
    `get_and_fit_model`.
    """
    if task_feature is None:
        if len(Xs) == 1:
            # Use single output, single task GP
            return _get_model(
                X=Xs[0],
                Y=Ys[0],
                Yvar=Yvars[0],
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                use_input_warping=use_input_warping,
                prior=deepcopy(prior),
                **kwargs,
            )
        if all(torch.equal(Xs[0], X) for X in Xs[1:]) and not use_input_warping:
            # Use batched multioutput, single task GP
            # Require using a ModelListGP if using input warping
            Y = torch.cat(Ys, dim=-1)
            Yvar = torch.cat(Yvars, dim=-1)
            return _get_model(
                X=Xs[0],
                Y=Y,
                Yvar=Yvar,
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                prior=deepcopy(prior),
                **kwargs,
            )

    if task_feature is None:
        models = [
            _get_model(
                X=X,
                Y=Y,
                Yvar=Yvar,
                use_input_warping=use_input_warping,
                prior=deepcopy(prior),
                **kwargs,
            )
            for X, Y, Yvar in zip(Xs, Ys, Yvars)
        ]
    else:
        # use multi-task GP
        mtgp_rank_dict = {} if multitask_gp_ranks is None else multitask_gp_ranks
        # assembles list of ranks associated with each metric
        if len({len(Xs), len(Ys), len(Yvars), len(metric_names)}) > 1:
            raise ValueError(
                "Lengths of Xs, Ys, Yvars, and metric_names must match. Your "
                f"inputs have lengths {len(Xs)}, {len(Ys)}, {len(Yvars)}, and "
                f"{len(metric_names)}, respectively."
            )
        mtgp_rank_list = [mtgp_rank_dict.get(metric, None) for metric in metric_names]
        models = [
            _get_model(
                X=X,
                Y=Y,
                Yvar=Yvar,
                task_feature=task_feature,
                rank=mtgp_rank,
                use_input_warping=use_input_warping,
                prior=deepcopy(prior),
                **kwargs,
            )
            for X, Y, Yvar, mtgp_rank in zip(Xs, Ys, Yvars, mtgp_rank_list)
        ]
    return ModelListGP(*models)


def get_and_fit_model(
    Xs: list[Tensor],
    Ys: list[Tensor],
    Yvars: list[Tensor],
    task_features: list[int],
    fidelity_features: list[int],
    metric_names: list[str],
    state_dict: dict[str, Tensor] | None = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    prior: dict[str, Any] | None = None,
    *,
    multitask_gp_ranks: dict[str, Prior | float] | None = None,
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
        prior: Optional[Dict]. A dictionary that contains the specification of
            GP model prior. Currently, the keys include:
            - covar_module_prior: prior on covariance matrix e.g.
                {"lengthscale_prior": GammaPrior(3.0, 6.0)}.
            - type: type of prior on task covariance matrix e.g.`LKJCovariancePrior`.
            - sd_prior: A scalar prior over nonnegative numbers, which is used for the
                default LKJCovariancePrior task_covar_prior.
            - eta: The eta parameter on the default LKJ task_covar_prior.
        kwargs: Passed to `_get_model`.

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

    model = _construct_model(
        task_feature=task_feature,
        Xs=Xs,
        Ys=Ys,
        Yvars=Yvars,
        fidelity_features=fidelity_features,
        metric_names=metric_names,
        use_input_warping=use_input_warping,
        prior=prior,
        multitask_gp_ranks=multitask_gp_ranks,
        **kwargs,
    )

    # TODO: Better logic for deciding when to use a ModelListGP. Currently the
    # logic is unclear. The two cases in which ModelListGP is used are
    # (i) the training inputs (Xs) are not the same for the different outcomes, and
    # (ii) a multi-task model is used

    model.to(Xs[0])
    if state_dict is not None:
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
        mll = fit_gpytorch_mll(mll, optimizer_kwargs={"bounds": bounds})
    return model


class TAcqfConstructor(Protocol):
    def __call__(
        self,  # making this a static method makes Pyre unhappy, better to keep `self`
        model: Model,
        objective_weights: Tensor,
        outcome_constraints: tuple[Tensor, Tensor] | None = None,
        X_observed: Tensor | None = None,
        X_pending: Tensor | None = None,
        **kwargs: Any,
    ) -> AcquisitionFunction: ...  # pragma: no cover


def get_acqf(
    acquisition_function_name: str,
) -> Callable[[Callable[[], None]], TAcqfConstructor]:
    """Returns a decorator whose wrapper function instantiates an acquisition function.

    NOTE: This is a decorator factory instead of a simple factory as serialization
    of Botorch model kwargs requires callables to be have module-level paths, and
    closures created by a simple factory do not have such paths. We solve this by
    wrapping "empty" module-level functions with this decorator, we ensure that they
    are serialized correctly, in addition to reducing code duplication.

    Example:
        >>> @get_acqf("qEI")
        ... def get_qEI() -> None:
        ...     pass
        >>> acqf = get_qEI(
        ...     model=model,
        ...     objective_weights=objective_weights,
        ...     outcome_constraints=outcome_constraints,
        ...     X_observed=X_observed,
        ...     X_pending=X_pending,
        ...     **kwargs,
        ... )
        >>> type(acqf)
        ... botorch.acquisition.monte_carlo.qExpectedImprovement

    Args:
        acquisition_function_name: The name of the acquisition function to be
            instantiated by the returned function.

    Returns:
        A decorator whose wrapper function is a TAcqfConstructor, i.e. it requires a
        `model`, `objective_weights`, and optional `outcome_constraints`, `X_observed`,
        and `X_pending` as inputs, as well as `kwargs`, and returns an
        `AcquisitionFunction` instance that corresponds to `acquisition_function_name`.
    """

    def decorator(empty_acqf_getter: Callable[[], None]) -> TAcqfConstructor:
        # `wraps` allows the function to keep its original, module-level name, enabling
        # serialization via `callable_to_reference`. `empty_acqf_getter` is otherwise
        # not used in the wrapper.
        @functools.wraps(empty_acqf_getter)
        def wrapper(
            model: Model,
            objective_weights: Tensor,
            outcome_constraints: tuple[Tensor, Tensor] | None = None,
            X_observed: Tensor | None = None,
            X_pending: Tensor | None = None,
            **kwargs: Any,
        ) -> AcquisitionFunction:
            kwargs.pop("objective_thresholds", None)
            return _get_acquisition_func(
                model=model,
                acquisition_function_name=acquisition_function_name,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
                **kwargs,
            )

        return wrapper

    return decorator


@get_acqf("qEI")
def get_qEI() -> None:
    """A TAcqfConstructor to instantiate a qEI acquisition function. The function body
    is filled in by the decorator function `get_acqf` to simultaneously reduce code
    duplication and allow serialization in Ax. TODO: Deprecate with legacy Ax model.
    """


@get_acqf("qLogEI")
def get_qLogEI() -> None:
    """TAcqfConstructor instantiating qLogEI. See docstring of get_qEI for details."""


@get_acqf("qNEI")
def get_NEI() -> None:  # no "q" in method name for backward compatibility
    """TAcqfConstructor instantiating qNEI. See docstring of get_qEI for details."""


@get_acqf("qLogNEI")
def get_qLogNEI() -> None:
    """TAcqfConstructor instantiating qLogNEI. See docstring of get_qEI for details."""


def _get_acquisition_func(
    model: Model,
    acquisition_function_name: str,
    objective_weights: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    X_observed: Tensor | None = None,
    X_pending: Tensor | None = None,
    mc_objective: type[GenericMCObjective] = GenericMCObjective,
    constrained_mc_objective: None
    | (type[ConstrainedMCObjective]) = ConstrainedMCObjective,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict` to avoid runtime subscripting errors.
    mc_objective_kwargs: dict | None = None,
    *,
    chebyshev_scalarization: bool = False,
    prune_baseline: bool = True,
    mc_samples: int = 512,
    marginalize_dim: int | None = None,
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
        mc_objective: GenericMCObjective class, used for constructing a
            MC-objective. If constructing a penalized MC-objective, pass in
            PenalizedMCObjective together with mc_objective_kwargs .
        constrained_mc_objective: ConstrainedMCObjective class, used when
            applying constraints on the outcomes.
        mc_objective_kwargs: kwargs for constructing MC-objective.
            For GenericMCObjective, leave it as None. For PenalizedMCObjective,
            it needs to be specified in the format of kwargs.
        mc_samples: The number of MC samples to use (default: 512).
        prune_baseline: If True, prune the baseline points for NEI (default: True).
        chebyshev_scalarization: Use augmented Chebyshev scalarization.

    Returns:
        The instantiated acquisition function.
    """
    if acquisition_function_name not in [
        "qSR",
        "qEI",
        "qLogEI",
        "qPI",
        "qNEI",
        "qLogNEI",
    ]:
        raise NotImplementedError(f"{acquisition_function_name=} not implemented yet.")

    if X_observed is None:
        raise ValueError(NO_OBSERVED_POINTS_MESSAGE)
    # construct Objective module
    if chebyshev_scalarization:
        with torch.no_grad():
            Y = model.posterior(X_observed).mean  # pyre-ignore [16]
        if is_ensemble(model):
            Y = torch.mean(Y, dim=0)
        obj_tf = get_chebyshev_scalarization(weights=objective_weights, Y=Y)
    else:
        obj_tf = get_objective_weights_transform(objective_weights)

    # pyre-fixme[53]: Captured variable `obj_tf` is not annotated.
    def objective(samples: Tensor, X: Tensor | None = None) -> Tensor:
        return obj_tf(samples)

    mc_objective_kwargs = {} if mc_objective_kwargs is None else mc_objective_kwargs
    objective = mc_objective(objective=objective, **mc_objective_kwargs)

    if outcome_constraints is None:
        con_tfs = None
    else:
        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # All acquisition functions registered in BoTorch's `get_acquisition_function`
        # except qSR and qUCB support a principled treatment of the constraints by
        # directly passing them to the constructor.
        if acquisition_function_name == "qSR":
            if constrained_mc_objective is None:
                raise ValueError(
                    "constrained_mc_objective cannot be set to None "
                    "when applying outcome constraints."
                )

            inf_cost = get_infeasible_cost(
                X=X_observed, model=model, objective=objective
            )
            objective = constrained_mc_objective(
                objective=objective, constraints=con_tfs or [], infeasible_cost=inf_cost
            )

    return get_acquisition_function(
        acquisition_function_name=acquisition_function_name,
        model=model,
        objective=objective,
        X_observed=X_observed,
        X_pending=X_pending,
        prune_baseline=prune_baseline,
        mc_samples=mc_samples,
        seed=randint(1, 10000),
        marginalize_dim=marginalize_dim,
        constraints=con_tfs,
    )


def scipy_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    fixed_features: dict[int, float] | None = None,
    rounding_func: Callable[[Tensor], Tensor] | None = None,
    *,
    num_restarts: int = 20,
    raw_samples: int | None = None,
    joint_optimization: bool = False,
    options: dict[str, bool | float | int | str] | None = None,
) -> tuple[Tensor, Tensor]:
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

    sequential = not joint_optimization
    optimize_acqf_options: dict[str, bool | float | int | str] = {
        "batch_limit": 5,
        "init_batch_limit": 32,
    }
    if options is not None:
        optimize_acqf_options.update(options)
    X, expected_acquisition_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=50 * num_restarts if raw_samples is None else raw_samples,
        options=optimize_acqf_options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        post_processing_func=rounding_func,
    )
    return X, expected_acquisition_value


def recommend_best_observed_point(
    model: TorchModel,
    bounds: list[tuple[float, float]],
    objective_weights: Tensor,
    outcome_constraints: tuple[Tensor, Tensor] | None = None,
    linear_constraints: tuple[Tensor, Tensor] | None = None,
    fixed_features: dict[int, float] | None = None,
    model_gen_options: TConfig | None = None,
    target_fidelities: dict[int, float] | None = None,
) -> Tensor | None:
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
            model-specific options. See `TorchOptConfig` for details.
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


def _get_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: int | None = None,
    fidelity_features: list[int] | None = None,
    use_input_warping: bool = False,
    covar_module: Kernel | None = None,
    prior: dict[str, Any] | None = None,
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
        covar_module: Optional. A data kernel of GP model.
        prior: Optional[Dict]. A dictionary that contains the specification of
            GP model prior. Currently, the keys include:
            - covar_module_prior: prior on covariance matrix e.g.
                {"lengthscale_prior": GammaPrior(3.0, 6.0)}.
            - type: type of prior on task covariance matrix e.g.`LKJCovariancePrior`.
            - sd_prior: A scalar prior over nonnegative numbers, which is used for the
                default LKJCovariancePrior task_covar_prior.
            - eta: The eta parameter on the default LKJ task_covar_prior.

    Returns:
        A GPyTorchModel (unfitted).
    """
    Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)
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
        if Y.shape[-1] > 1 and X.ndim > 2:
            raise UnsupportedError(
                "Input warping is not supported for batched multi output models."
            )
        warp_tf = get_warping_transform(
            d=X.shape[-1],
            task_feature=task_feature,
            batch_shape=X.shape[:-2],
        )
    else:
        warp_tf = None
    if fidelity_features is None:
        fidelity_features = []
    if len(fidelity_features) == 0:
        # only pass linear_truncated arg if there are fidelities
        kwargs = {k: v for k, v in kwargs.items() if k != "linear_truncated"}
    # construct kernel based on customized prior if covar_module is None
    prior_dict = prior or {}
    covar_module_prior_dict = prior_dict.pop("covar_module_prior", None)
    if (covar_module_prior_dict is not None) and (covar_module is None):
        covar_module = _get_customized_covar_module(
            covar_module_prior_dict=covar_module_prior_dict,
            ard_num_dims=X.shape[-1],
            aug_batch_shape=_get_aug_batch_shape(X, Y),
            task_feature=task_feature,
        )

    if len(fidelity_features) > 0:
        if task_feature:
            raise NotImplementedError(
                "multi-task multi-fidelity models not yet available"
            )
        # at this point we can assume that there is only a single fidelity parameter
        gp = SingleTaskMultiFidelityGP(
            train_X=X,
            train_Y=Y,
            data_fidelities=fidelity_features[:1],
            input_transform=warp_tf,
            **kwargs,
        )
    elif task_feature is None:
        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=None if all_nan_Yvar else Yvar,
            covar_module=covar_module,
            input_transform=warp_tf,
            **{"outcome_transform": None, **kwargs},
        )
    else:
        # instantiate multitask GP
        all_tasks, _, _ = MultiTaskGP.get_all_tasks(X, task_feature)
        num_tasks = len(all_tasks)
        task_covar_prior = None
        if len(prior_dict) > 0:
            prior_type = prior_dict.get("type", None)
            if issubclass(prior_type, LKJCovariancePrior):
                sd_prior = prior_dict.get("sd_prior", GammaPrior(1.0, 0.15))
                sd_prior._event_shape = torch.Size([num_tasks])
                eta = prior_dict.get("eta", 0.5)
                if not isinstance(eta, float) and not isinstance(eta, int):
                    raise ValueError(f"eta must be a real number, your eta was {eta}")
                task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

            else:
                raise NotImplementedError(
                    "Currently only LKJ prior is supported,"
                    f"your prior type was {prior_type}."
                )

        gp = MultiTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=None if all_nan_Yvar else Yvar,
            task_feature=task_feature,
            covar_module=covar_module,
            rank=kwargs.get("rank"),
            task_covar_prior=task_covar_prior,
            input_transform=warp_tf,
            # specify output_tasks so that model.num_outputs
            # is 1, since the model is only modeling
            # a since metric.
            output_tasks=all_tasks[:1],
        )
    return gp


def _get_customized_covar_module(
    covar_module_prior_dict: dict[str, Prior],
    ard_num_dims: int,
    aug_batch_shape: torch.Size,
    task_feature: int | None = None,
) -> Kernel:
    """Construct a GP kernel based on customized prior dict.

    Args:
        covar_module_prior_dict: Dict. The keys are the names of the prior and values
            are the priors. e.g. {"lengthscale_prior": GammaPrior(3.0, 6.0)}.
        ard_num_dims: The dimension of the inputs, including task features.
        aug_batch_shape: The output dimension augmented batch shape of the model
            (different from the batch shape for batched multi-output models).
        task_feature: The index of the task feature.
    """
    # TODO: add more checks of covar_module_prior_dict
    if task_feature is not None:
        ard_num_dims -= 1
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=aug_batch_shape,
            lengthscale_prior=covar_module_prior_dict.get(
                "lengthscale_prior", GammaPrior(3.0, 6.0)
            ),
        ),
        batch_shape=aug_batch_shape,
        outputscale_prior=covar_module_prior_dict.get(
            "outputscale_prior", GammaPrior(2.0, 0.15)
        ),
    )


def _get_aug_batch_shape(X: Tensor, Y: Tensor) -> torch.Size:
    """Obtain the output-augmented batch shape of GP model.

    Args:
        X: A `(input_batch_shape) x n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.

    Returns:
        The output-augmented batch shape: `input_batch_shape x (m)`
    """
    batch_shape = X.shape[:-2]
    num_outputs = Y.shape[-1]
    if num_outputs > 1:
        batch_shape += torch.Size([num_outputs])  # pyre-ignore
    return batch_shape


def get_warping_transform(
    d: int,
    batch_shape: torch.Size | None = None,
    task_feature: int | None = None,
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
    # Legacy Ax models operate in the unit cube
    bounds = torch.zeros(2, d, dtype=torch.double)
    bounds[1] = 1
    # Note: this currently uses the same warping functions for all tasks
    tf = Warp(
        d=d,
        indices=indices,
        # prior with a median of 1
        concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
        concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
        batch_shape=batch_shape,
        # Legacy Ax models operate in the unit cube
        bounds=bounds,
    )
    return tf
