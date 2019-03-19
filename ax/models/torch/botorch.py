#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from ax.core.types.types import TConfig
from ax.models.model_utils import (
    best_observed_point,
    filter_constraints_and_fixed_features,
    get_observed,
)
from ax.models.torch.utils import _get_model
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from botorch.acquisition.objective import ConstrainedMCObjective, LinearMCObjective
from botorch.acquisition.utils import get_acquisition_function, get_infeasible_cost
from botorch.fit import fit_model
from botorch.models import MultiOutputGP
from botorch.optim.optimize import joint_optimize, sequential_optimize
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


class BotorchModel(TorchModel):
    """
    Class implementing a single task multi-outcome model, assuming independence
    across outcomes.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    model: MultiOutputGP
    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: List[Tensor]

    def __init__(
        self,
        acquisition_function_name: str = "qNEI",
        acquisition_function_kwargs: Optional[Dict[str, Union[float, int]]] = None,
        refit_on_cv: bool = False,
    ) -> None:
        """
        Args:
            acquisition_function_name: a string representing the acquisition
                function name.
            acquisition_function_kwargs: A map containing extra arguments for
                initializing the acquisition function module. For instance, for
                UCB this should include the `beta` parameter.
            refit_on_cv: Re-fit hyperparameters during cross validation.
        """
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_kwargs = acquisition_function_kwargs or {}
        # pyre-fixme[8]: Attribute has type `MultiOutputGP`; used as `None`.
        self.model = None
        self.Xs = []
        self.Ys = []
        self.Yvars = []
        self.refit_on_cv = refit_on_cv
        self.dtype = None
        self.device = None

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
    ) -> None:
        if len(task_features) > 0:
            raise ValueError("Task features not supported.")
        self.dtype = Xs[0].dtype  # pyre-ignore
        self.device = Xs[0].device
        self.Xs = Xs
        self.Ys = Ys
        self.Yvars = Yvars
        self.model = _get_and_fit_model(Xs=Xs, Ys=Ys, Yvars=Yvars)

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return _model_predict(model=self.model, X=X)

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate new candidates.

        An initialized acquisition function can be passed in as
        model_gen_options["acquisition_function"].

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b. (Not used by single task models)
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b. (Not used by single task models)
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature tensors X
                for m outcomes and k_i pending observations for outcome i.
                (Only used if n > 1).
            model_gen_options: A config dictionary that can contain
                model-specific options.
            rounding_func: A function that rounds an optimization result
                appropriately (i.e., according to `round-trip` transformations).
        Returns:
            2-element tuple containing

            - (n x d) tensor of generated points.
            - n-tensor of weights for each point.
        """
        options: TConfig = model_gen_options or {}
        if pending_observations is not None:
            # Get points observed for all objective and constraint outcomes
            X_pending = get_observed(
                Xs=pending_observations,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )
            # Filter to those that satisfy constraints.
            X_pending = filter_constraints_and_fixed_features(
                X=X_pending,
                bounds=bounds,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
            )
            if len(X_pending) == 0:
                X_pending = None
        else:
            X_pending = None

        # Get points observed for all objective and constraint outcomes
        X_observed = get_observed(
            Xs=self.Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )
        # Filter to those that satisfy constraints.
        X_observed = filter_constraints_and_fixed_features(
            X=X_observed,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
        if len(X_observed) == 0:
            raise ValueError("There are no feasible observed points.")
        # construct Objective module
        if outcome_constraints is None:
            objective = LinearMCObjective(weights=objective_weights)
        else:
            obj_tf = get_objective_weights_transform(objective_weights)
            con_tfs = get_outcome_constraint_transforms(outcome_constraints)
            if not torch.is_tensor(X_observed):  # this is just to shut up pyre
                X_observed = torch.tensor(X_observed)
            inf_cost = get_infeasible_cost(
                X=X_observed, model=self.model, objective=obj_tf
            )
            objective = ConstrainedMCObjective(
                objective=obj_tf,
                constraints=con_tfs or [],  # shut up pyre
                infeasible_cost=inf_cost,
            )
        # get the AcquisitionFunction
        acquisition_function = get_acquisition_function(
            acquisition_function_name=self.acquisition_function_name,
            model=self.model,
            objective=objective,
            X_observed=X_observed,
            X_pending=X_pending,
            mc_samples=self.acquisition_function_kwargs.get("mc_samples", 500),
            qmc=self.acquisition_function_kwargs.get("qmc", True),
            seed=torch.randint(1, 10000, (1,)).item(),
            **self.acquisition_function_kwargs
        )

        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

        # TODO: implement better random restart heuristic
        # pyre-fixme[9]: opts has type `Dict[str, Union[bool, float, int]]`; used as ...
        opts: Dict[str, Union[bool, float, int]] = options
        # pyre-fixme[9]: num_restarts has type `int`; used as `Union[float, str]`.
        num_restarts: int = options.get("num_restarts", 20)
        # pyre-fixme[9]: raw_samples has type `int`; used as `Union[float, str]`.
        raw_samples: int = options.get("num_raw_samples", 50 * num_restarts)
        # pyre-fixme[9]: joint_optimization has type `bool`; used as `Union[float, st...
        joint_optimization: bool = options.get("joint_optimization", False)
        optimize = joint_optimize if joint_optimization else sequential_optimize
        candidates = optimize(
            acq_function=acquisition_function,
            bounds=bounds_,
            n=n,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=opts,
            fixed_features=fixed_features,
            post_processing_func=rounding_func,
        )
        return candidates.detach().cpu(), torch.ones(n, dtype=self.dtype)

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[Tensor]:
        x_best = best_observed_point(
            model=self,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=model_gen_options,
        )
        if x_best is None:
            return None
        else:
            return x_best.to(dtype=self.dtype, device=torch.device("cpu"))

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[Tensor],
        Ys_train: List[Tensor],
        Yvars_train: List[Tensor],
        X_test: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        state_dict = None if self.refit_on_cv else self.model.state_dict()
        model = _get_and_fit_model(
            Xs=Xs_train, Ys=Ys_train, Yvars=Yvars_train, state_dict=state_dict
        )
        return _model_predict(model=model, X=X_test)


def _get_and_fit_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    state_dict: Optional[Dict[str, Tensor]] = None,
) -> MultiOutputGP:
    """Instantiate a model with the given data.

    Args:
        Xs: List of X data, one tensor per outcome
        Ys: List of Y data, one tensor per outcome
        Yvars: List of observed variance of Ys.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.

    Returns:
        A MultiOutputGP.
    """
    models = [_get_model(X=X, Y=Y, Yvar=Yvar) for X, Y, Yvar in zip(Xs, Ys, Yvars)]
    model = MultiOutputGP(gp_models=models)  # pyre-ignore: [16]
    model.to(dtype=Xs[0].dtype, device=Xs[0].device)  # pyre-ignore
    if state_dict is None:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        mll = fit_model(mll, bounds=bounds)
    else:
        model.load_state_dict(state_dict)
    return model


def _model_predict(model: MultiOutputGP, X: Tensor) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        posterior = model.posterior(X)
    mean = posterior.mean.cpu().detach()
    variance = posterior.variance.cpu().detach()
    cov = variance.unsqueeze(-1) * torch.eye(variance.shape[-1], dtype=variance.dtype)
    return mean, cov
