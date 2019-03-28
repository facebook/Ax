#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TConfig
from ax.models.model_utils import best_observed_point
from ax.models.torch.botorch_defaults import (  # pyre-ignore [21]
    get_and_fit_model,
    get_NEI,
    predict_from_model,
    scipy_optimizer,
)
from ax.models.torch.utils import _get_X_pending_and_observed
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import Model
from torch import Tensor


TModelConstuctor = Callable[
    [List[Tensor], List[Tensor], List[Tensor], Optional[Dict[str, Tensor]], Any], Model
]
TModelPredictor = Callable[[Model, Tensor], Tuple[Tensor, Tensor]]
TAcqfConstructor = Callable[
    [
        Model,
        Tensor,
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tensor],
        Optional[Tensor],
        Any,
    ],
    AcquisitionFunction,
]
TOptimizer = Callable[
    [
        AcquisitionFunction,
        Tensor,
        int,
        Optional[Dict[int, float]],
        Optional[Callable[[Tensor], Tensor]],
        Any,
    ],
    Tensor,
]


class BotorchModel(TorchModel):
    """
    Customizable botorch model.

    By default, this uses a noisy Expected Improvement acquisition funciton on
    top of a model made up of separate GPs, one for each outcome. This behavior
    can be modified by providing custom implementations of the following
    components:
        - a `model_constructor` that instantiates and fits a model on data
        - a `model_predictor` that predicts using the fitted model
        - a `acqf_constructor` that creates an acquisition function from a fitted model
        - a `acqf_optimizer` that optimizes the acquisition function

    Args:
        model_constructor: A callable with the following signature:
            ```
            model_constructor(Xs, Ys, Yvars, state_dict, **kwargs) -> model
            ```
            where `Xs`, `Ys`, `Yvars` are lists of tensors (one element per
            outcome), `state_dict` is a pytorch module state dict, and `model` is
            a botorch `Model`. Optional kwargs are being passed through from the
            `BotorchModel` constructor. This callable is assumed to return a
            fitted botorch model that has the same dtype and lives on the same
            device as the input tensors.
        model_predictor: A callable with the following signature:
            ```
            model_predictor(model, X) -> [mean, cov]
            ````
            where `model` is a fitted botorch model, `X` is a tensor of
            candidate points, and `mean` and `cov` are the posterior mean and
            covariance, respectively.
        acqf_constructor: A callable with the following signature:
            ```
            acqf_constructor(
                model,
                objective_weights,
                outcome_constraints,
                X_observed,
                X_pending,
                **kwargs,
            ) -> acq_function
            ```
            where `model` is a botorch `Model`, `objective_weights` is a tensor
            of weights for the model outputs, `outcome_constraints` is a tuple
            of tensors describing the (linear) outcome constraints, `X_observed`
            are previously observed points, and `X_pending` are points whose
            evaluation is pending. `acq_function` is a botorch acquisition
            function crafted from these inputs.
            For additional details on the arguments, see `get_NEI`.
        acqf_optimizer: A callable with the following signature:
            ```
            acqf_optimizer(
                acq_function,
                bounds,
                n,
                fixed_features,
                rounding_func,
                **kwargs,
            ) -> candidates
            ```
            where `acq_function` is a botorch `AcquisitionFunciton`, `bounds` is
            a tensor containing bounds on the parameters, `n` is the number of
            candidates to be generated, `fixed_features` specifies features that
            should be fixed during generation, and `rounding_func` is a callback
            that rounds an optimization result appropriately. `candidates` is
            a tensor of generated candidates.
            For additional details on the arguments, see `scipy_optimizer`.
        refit_on_cv: If True, refit the model for each fold when performing
            cross-validation.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: List[Tensor]

    def __init__(
        self,
        model_constructor: TModelConstuctor = get_and_fit_model,
        model_predictor: TModelPredictor = predict_from_model,
        acqf_constructor: TAcqfConstructor = get_NEI,
        acqf_optimizer: TOptimizer = scipy_optimizer,
        refit_on_cv: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_constructor = model_constructor
        self.model_predictor = model_predictor
        self.acqf_constructor = acqf_constructor
        self.acqf_optimizer = acqf_optimizer
        self.refit_on_cv = refit_on_cv
        self.model = None
        self.Xs = []
        self.Ys = []
        self.Yvars = []
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
        self.dtype = Xs[0].dtype  # pyre-ignore [16]
        self.device = Xs[0].device
        self.Xs = Xs
        self.Ys = Ys
        self.Yvars = Yvars
        self.model = self.model_constructor(  # pyre-ignore [28]
            Xs=Xs, Ys=Ys, Yvars=Yvars
        )

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return self.model_predictor(model=self.model, X=X)  # pyre-ignore [28]

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
        """Generate new candidates.

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
            Tensor: `n x d`-dim Tensor of generated points.
            Tensor: `n`-dim Tensor of weights for each point.
        """
        options = model_gen_options or {}
        acf_options = options.get("acqiusition_function_kwargs", {})
        optimizer_options = options.get("optimizer_kwargs", {})

        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            pending_observations=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )

        acquisition_function = self.acqf_constructor(  # pyre-ignore: [28]
            model=self.model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            X_pending=X_pending,
            **acf_options,
        )

        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

        candidates = self.acqf_optimizer(  # pyre-ignore: [28]
            acq_function=checked_cast(AcquisitionFunction, acquisition_function),
            bounds=bounds_,
            n=n,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            **optimizer_options,
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
        return x_best.to(dtype=self.dtype, device=torch.device("cpu"))

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[Tensor],
        Ys_train: List[Tensor],
        Yvars_train: List[Tensor],
        X_test: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.model is None:
            raise RuntimeError("Cannot cross-validate model that has not been fitted")
        state_dict = (
            None if self.refit_on_cv else self.model.state_dict()  # pyre-ignore [16]
        )
        model = self.model_constructor(  # pyre-ignore [28]
            Xs=Xs_train, Ys=Ys_train, Yvars=Yvars_train, state_dict=state_dict
        )
        return self.model_predictor(model=model, X=X_test)  # pyre-ignore [28]
