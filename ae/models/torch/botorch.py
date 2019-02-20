#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.models.model_utils import best_observed_point
from ae.lazarus.ae.models.torch.utils import (
    MIN_INFERRED_NOISE_LEVEL,
    _get_model,
    _joint_optimize,
    _sequential_optimize,
)
from ae.lazarus.ae.models.torch_base import TorchModel
from ae.lazarus.ae.utils.common.docutils import copy_doc
from botorch.acquisition.utils import get_acquisition_function
from botorch.fit import fit_model
from botorch.models import Model
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


class BotorchModel(TorchModel):
    """
    Class implementing a single task multi-outcome model, assuming independence
    across outcomes.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    models: List[Model]
    train_X: List[Tensor]

    def __init__(
        self,
        acquisition_function_name: str = "qNEI",
        acquisition_function_args: Optional[Dict[str, Union[float, int]]] = None,
        refit_on_cv: bool = False,
    ) -> None:
        """
        Args:
            acquisition_function_name: a string representing the acquisition
                function name
            acquisition_function_args: A map containing extra arguments for initializing
                the acquisition function module. For UCB, this should include beta.
            refit_on_cv: Re-fit hyperparameters during cross validation
        """
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_args = acquisition_function_args
        self.models = []
        self.Xs = []
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
        self.dtype = Xs[0].dtype
        self.device = Xs[0].device
        self.Xs = Xs
        for X, Y, Yvar in zip(Xs, Ys, Yvars):
            self.models.append(_get_and_fit_model(X=X, Y=Y, Yvar=Yvar))

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return _gp_predict(models=self.models, X=X)

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: Optional[Tensor],
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
            X: An (n x d) tensor of generated points.
            w: An n-tensor of weights for each point.
        """
        options: TConfig = model_gen_options or {}
        if pending_observations is not None:
            X_pending = pending_observations[0]

            if not all(torch.equal(p, X_pending) for p in pending_observations[1:]):
                raise ValueError(
                    "Model requires pending observations to be the same "
                    "for all outcomes."
                )
        else:
            X_pending = None
        if objective_weights is None:
            raise ValueError("This model requires objective weights.")
        if len(torch.nonzero(objective_weights)) != 1:
            raise ValueError("Objective must be a single outcome.")
        objective_indx = int(torch.nonzero(objective_weights))
        objective_transform = get_objective_weights_transform(
            objective_weights=objective_weights
        )
        outcome_constraint_transforms = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        if linear_constraints is not None or outcome_constraints is not None:
            raise ValueError("This model does not support constraints.")

        acquisition_function: Optional[torch.nn.Module] = options.get(
            "acquisition_function"
        )
        if acquisition_function is None:
            acquisition_function = get_acquisition_function(
                acquisition_function_name=self.acquisition_function_name,
                model=self.models[objective_indx],
                X_observed=self.Xs[objective_indx],
                objective=objective_transform,
                constraints=outcome_constraint_transforms,
                X_pending=X_pending,
                acquisition_function_args=self.acquisition_function_args,
                seed=torch.randint(1, 10000, (1,)).item(),
            )

        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

        # TODO: implement better heuristic

        opts: Dict[str, Union[bool, float, int]] = options
        num_restarts: int = options.get("num_restarts", 20)
        raw_samples: int = options.get(
            "num_raw_samples", 1000 if self.device == torch.device("cpu") else 5000
        )
        joint_optimization: bool = options.get("joint_optimization", False)
        optimize = _joint_optimize if joint_optimization else _sequential_optimize
        candidates = optimize(
            acq_function=acquisition_function,
            bounds=bounds_,
            n=n,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            model=self.models[objective_indx],
            options=opts,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
        )
        return candidates.detach().cpu(), torch.ones(n, dtype=self.dtype)

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Optional[Tensor],
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
        cv_models: List[Model] = []
        for i, m in enumerate(self.models):
            state_dict = None if self.refit_on_cv else m.state_dict()
            cv_models.append(
                _get_and_fit_model(
                    X=Xs_train[i],
                    Y=Ys_train[i],
                    Yvar=Yvars_train[i],
                    state_dict=state_dict,
                )
            )
        return _gp_predict(models=cv_models, X=X_test)


def _get_and_fit_model(
    X: Tensor, Y: Tensor, Yvar: Tensor, state_dict: Optional[Dict[str, Tensor]] = None
) -> Model:
    """Instantiate a model with the given data.

    Args:
        X: X data
        Y: Y data
        Yvar: Observed variance of Y. If all zero, assume noiseless data.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.

    Returns: A HeteroskedasticSingleTaskGP.
    """
    model = _get_model(X=X, Y=Y, Yvar=Yvar)
    if state_dict is None:
        # get bound on noise level (avoid numerical issues in gpytorch)
        if isinstance(model, HeteroskedasticSingleTaskGP):
            # This is the only model with which we infer the noise level
            noise_covar = (
                model.likelihood.noise_covar.noise_model.likelihood.noise_covar
            )
            lb = noise_covar._inv_param_transform(
                torch.tensor(MIN_INFERRED_NOISE_LEVEL)
            ).item()
            nc = "likelihood.noise_covar.noise_model.likelihood.noise_covar.raw_noise"
            bounds = {nc: (lb, None)}
        else:
            bounds = {}
        mll = ExactMarginalLogLikelihood(model.likelihood, model)  # pyre-ignore: [16]
        mll = fit_model(mll, bounds=bounds)
    else:
        model.load_state_dict(state_dict)  # pyre-ignore: [16]
    return model


def _gp_predict(models: List[Model], X: Tensor) -> Tuple[Tensor, Tensor]:
    n, t = X.shape[0], len(models)
    mean = torch.zeros(n, t, dtype=X.dtype)
    cov = torch.zeros(n, t, t, dtype=X.dtype)
    for i, model in enumerate(models):
        with torch.no_grad():
            posterior = model.posterior(X)
        mean[:, i].copy_(posterior.mean.cpu().detach())
        cov[:, i, i].copy_(posterior.variance.cpu().detach())
    return mean, cov
