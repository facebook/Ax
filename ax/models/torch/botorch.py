#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.exceptions.core import DataRequiredError
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    get_NEI,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.utils import (
    _datasets_to_legacy_inputs,
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    normalize_indices,
    predict_from_model,
    subset_model,
)
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.models.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor

logger: Logger = get_logger(__name__)


# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
TModelConstructor = Callable[
    [
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[int],
        List[int],
        List[str],
        Optional[Dict[str, Tensor]],
        Any,
    ],
    Model,
]
TModelPredictor = Callable[[Model, Tensor], Tuple[Tensor, Tensor]]
# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
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
# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
TOptimizer = Callable[
    [
        AcquisitionFunction,
        Tensor,
        int,
        Optional[List[Tuple[Tensor, Tensor, float]]],
        Optional[List[Tuple[Tensor, Tensor, float]]],
        Optional[Dict[int, float]],
        Optional[Callable[[Tensor], Tensor]],
        Any,
    ],
    Tuple[Tensor, Tensor],
]
TBestPointRecommender = Callable[
    [
        TorchModel,
        List[Tuple[float, float]],
        Tensor,
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
        Optional[Dict[int, float]],
        Optional[TConfig],
        Optional[Dict[int, float]],
    ],
    Optional[Tensor],
]


class BotorchModel(TorchModel):
    r"""
    Customizable botorch model.

    By default, this uses a noisy Expected Improvement acquisition function on
    top of a model made up of separate GPs, one for each outcome. This behavior
    can be modified by providing custom implementations of the following
    components:

    - a `model_constructor` that instantiates and fits a model on data
    - a `model_predictor` that predicts outcomes using the fitted model
    - a `acqf_constructor` that creates an acquisition function from a fitted model
    - a `acqf_optimizer` that optimizes the acquisition function
    - a `best_point_recommender` that recommends a current "best" point (i.e.,
        what the model recommends if the learning process ended now)

    Args:
        model_constructor: A callable that instantiates and fits a model on data,
            with signature as described below.
        model_predictor: A callable that predicts using the fitted model, with
            signature as described below.
        acqf_constructor: A callable that creates an acquisition function from a
            fitted model, with signature as described below.
        acqf_optimizer: A callable that optimizes the acquisition function, with
            signature as described below.
        best_point_recommender: A callable that recommends the best point, with
            signature as described below.
        refit_on_cv: If True, refit the model for each fold when performing
            cross-validation.
        refit_on_update: If True, refit the model after updating the training
            data using the `update` method.
        warm_start_refitting: If True, start model refitting from previous
            model parameters in order to speed up the fitting process.


    Call signatures:

    ::

        model_constructor(
            Xs,
            Ys,
            Yvars,
            task_features,
            fidelity_features,
            metric_names,
            state_dict,
            **kwargs,
        ) -> model

    Here `Xs`, `Ys`, `Yvars` are lists of tensors (one element per outcome),
    `task_features` identifies columns of Xs that should be modeled as a task,
    `fidelity_features` is a list of ints that specify the positions of fidelity
    parameters in 'Xs', `metric_names` provides the names of each `Y` in `Ys`,
    `state_dict` is a pytorch module state dict, and `model` is a BoTorch `Model`.
    Optional kwargs are being passed through from the `BotorchModel` constructor.
    This callable is assumed to return a fitted BoTorch model that has the same
    dtype and lives on the same device as the input tensors.

    ::

        model_predictor(model, X) -> [mean, cov]

    Here `model` is a fitted botorch model, `X` is a tensor of candidate points,
    and `mean` and `cov` are the posterior mean and covariance, respectively.

    ::

        acqf_constructor(
            model,
            objective_weights,
            outcome_constraints,
            X_observed,
            X_pending,
            **kwargs,
        ) -> acq_function


    Here `model` is a botorch `Model`, `objective_weights` is a tensor of weights
    for the model outputs, `outcome_constraints` is a tuple of tensors describing
    the (linear) outcome constraints, `X_observed` are previously observed points,
    and `X_pending` are points whose evaluation is pending. `acq_function` is a
    BoTorch acquisition function crafted from these inputs. For additional
    details on the arguments, see `get_NEI`.

    ::

        acqf_optimizer(
            acq_function,
            bounds,
            n,
            inequality_constraints,
            equality_constraints,
            fixed_features,
            rounding_func,
            **kwargs,
        ) -> candidates

    Here `acq_function` is a BoTorch `AcquisitionFunction`, `bounds` is a tensor
    containing bounds on the parameters, `n` is the number of candidates to be
    generated, `inequality_constraints` are inequality constraints on parameter
    values, `fixed_features` specifies features that should be fixed during
    generation, and `rounding_func` is a callback that rounds an optimization
    result appropriately. `candidates` is a tensor of generated candidates.
    For additional details on the arguments, see `scipy_optimizer`.

    ::

        best_point_recommender(
            model,
            bounds,
            objective_weights,
            outcome_constraints,
            linear_constraints,
            fixed_features,
            model_gen_options,
            target_fidelities,
        ) -> candidates

    Here `model` is a TorchModel, `bounds` is a list of tuples containing bounds
    on the parameters, `objective_weights` is a tensor of weights for the model outputs,
    `outcome_constraints` is a tuple of tensors describing the (linear) outcome
    constraints, `linear_constraints` is a tuple of tensors describing constraints
    on the design, `fixed_features` specifies features that should be fixed during
    generation, `model_gen_options` is a config dictionary that can contain
    model-specific options, and `target_fidelities` is a map from fidelity feature
    column indices to their respective target fidelities, used for multi-fidelity
    optimization problems. % TODO: refer to an example.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: List[Tensor]
    model: Optional[Model]

    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model,
        model_predictor: TModelPredictor = predict_from_model,
        # pyre-fixme[9]: acqf_constructor has type `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor], Any],
        #  AcquisitionFunction]`; used as `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor],
        #  **(Any)], AcquisitionFunction]`.
        acqf_constructor: TAcqfConstructor = get_NEI,
        # pyre-fixme[9]: acqf_optimizer declared/used type mismatch
        acqf_optimizer: TOptimizer = scipy_optimizer,
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = True,
        use_input_warping: bool = False,
        use_loocv_pseudo_likelihood: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_constructor = model_constructor
        self.model_predictor = model_predictor
        self.acqf_constructor = acqf_constructor
        self.acqf_optimizer = acqf_optimizer
        self.best_point_recommender = best_point_recommender
        # pyre-fixme[4]: Attribute must be annotated.
        self._kwargs = kwargs
        self.refit_on_cv = refit_on_cv
        self.refit_on_update = refit_on_update
        self.warm_start_refitting = warm_start_refitting
        self.use_input_warping = use_input_warping
        self.use_loocv_pseudo_likelihood = use_loocv_pseudo_likelihood
        self.model: Optional[Model] = None
        self.Xs = []
        self.Ys = []
        self.Yvars = []
        self.dtype = None
        self.device = None
        self.task_features: List[int] = []
        self.fidelity_features: List[int] = []
        self.metric_names: List[str] = []

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        if len(datasets) == 0:
            raise DataRequiredError("BotorchModel.fit requires non-empty data sets.")
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        self.metric_names = metric_names
        self.Xs, self.Ys, self.Yvars = Xs, Ys, Yvars
        self.dtype = self.Xs[0].dtype
        self.device = self.Xs[0].device
        self.task_features = normalize_indices(
            search_space_digest.task_features, d=self.Xs[0].size(-1)
        )
        self.fidelity_features = normalize_indices(
            search_space_digest.fidelity_features, d=self.Xs[0].size(-1)
        )
        self.model = self.model_constructor(  # pyre-ignore [28]
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            task_features=self.task_features,
            fidelity_features=self.fidelity_features,
            metric_names=self.metric_names,
            use_input_warping=self.use_input_warping,
            use_loocv_pseudo_likelihood=self.use_loocv_pseudo_likelihood,
            **self._kwargs,
        )

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return self.model_predictor(model=self.model, X=X)  # pyre-ignore [28]

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        options = torch_opt_config.model_gen_options or {}
        acf_options = options.get(Keys.ACQF_KWARGS, {})
        optimizer_options = options.get(Keys.OPTIMIZER_KWARGS, {})

        if search_space_digest.target_fidelities:
            raise NotImplementedError(
                "target_fidelities not implemented for base BotorchModel"
            )
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )
        model = self.model
        # subset model only to the outcomes we need for the optimization	357
        if options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=model,  # pyre-ignore [6]
                objective_weights=torch_opt_config.objective_weights,
                outcome_constraints=torch_opt_config.outcome_constraints,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
        else:
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints

        bounds_ = torch.tensor(
            search_space_digest.bounds, dtype=self.dtype, device=self.device
        )
        bounds_ = bounds_.transpose(0, 1)

        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)

        from botorch.exceptions.errors import UnsupportedError

        # pyre-fixme[53]: Captured variable `X_observed` is not annotated.
        # pyre-fixme[53]: Captured variable `X_pending` is not annotated.
        # pyre-fixme[53]: Captured variable `acf_options` is not annotated.
        # pyre-fixme[53]: Captured variable `botorch_rounding_func` is not annotated.
        # pyre-fixme[53]: Captured variable `bounds_` is not annotated.
        # pyre-fixme[53]: Captured variable `model` is not annotated.
        # pyre-fixme[53]: Captured variable `objective_weights` is not annotated.
        # pyre-fixme[53]: Captured variable `optimizer_options` is not annotated.
        # pyre-fixme[53]: Captured variable `outcome_constraints` is not annotated.
        def make_and_optimize_acqf(override_qmc: bool = False) -> Tuple[Tensor, Tensor]:
            add_kwargs = {"qmc": False} if override_qmc else {}
            acquisition_function = self.acqf_constructor(  # pyre-ignore: [28]
                model=model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
                **acf_options,
                **add_kwargs,
            )
            acquisition_function = checked_cast(
                AcquisitionFunction, acquisition_function
            )
            # pyre-ignore: [28]
            candidates, expected_acquisition_value = self.acqf_optimizer(
                acq_function=checked_cast(AcquisitionFunction, acquisition_function),
                bounds=bounds_,
                n=n,
                inequality_constraints=_to_inequality_constraints(
                    linear_constraints=torch_opt_config.linear_constraints
                ),
                fixed_features=torch_opt_config.fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
            return candidates, expected_acquisition_value

        try:
            candidates, expected_acquisition_value = make_and_optimize_acqf()
        except UnsupportedError as e:  # untested
            if "SobolQMCSampler only supports dimensions" in str(e):
                # dimension too large for Sobol, let's use IID
                candidates, expected_acquisition_value = make_and_optimize_acqf(
                    override_qmc=True
                )
            else:
                raise e

        gen_metadata = {}
        if expected_acquisition_value.numel() > 0:
            gen_metadata[
                "expected_acquisition_value"
            ] = expected_acquisition_value.tolist()

        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=torch.ones(n, dtype=self.dtype),
            gen_metadata=gen_metadata,
        )

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Optional[Tensor]:
        if torch_opt_config.is_moo:
            raise NotImplementedError(
                "Best observed point is incompatible with MOO problems."
            )
        return self.best_point_recommender(  # pyre-ignore [28]
            model=self,
            bounds=search_space_digest.bounds,
            objective_weights=torch_opt_config.objective_weights,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            model_gen_options=torch_opt_config.model_gen_options,
            target_fidelities=search_space_digest.target_fidelities,
        )

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(  # pyre-ignore [14]: `search_space_digest` arg not needed here
        self,
        datasets: List[SupervisedDataset],
        X_test: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        if self.model is None:
            raise RuntimeError("Cannot cross-validate model that has not been fitted")
        if self.refit_on_cv:
            state_dict = None
        else:
            state_dict = deepcopy(self.model.state_dict())
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        model = self.model_constructor(  # pyre-ignore: [28]
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            task_features=self.task_features,
            state_dict=state_dict,
            fidelity_features=self.fidelity_features,
            metric_names=self.metric_names,
            refit_model=self.refit_on_cv,
            use_input_warping=self.use_input_warping,
            use_loocv_pseudo_likelihood=self.use_loocv_pseudo_likelihood,
            **self._kwargs,
        )
        return self.model_predictor(model=model, X=X_test)  # pyre-ignore: [28]

    @copy_doc(TorchModel.update)
    def update(  # pyre-ignore [14]: `search_space_digest` arg not needed here
        self,
        datasets: List[SupervisedDataset],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        **kwargs: Any,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Cannot update model that has not been fitted")
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        self.Xs = Xs
        self.Ys = Ys
        self.Yvars = Yvars
        if self.refit_on_update and not self.warm_start_refitting:
            state_dict = None  # pragma: no cover
        else:
            state_dict = deepcopy(not_none(self.model).state_dict())
        self.model = self.model_constructor(  # pyre-ignore: [28]
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            task_features=self.task_features,
            state_dict=state_dict,
            fidelity_features=self.fidelity_features,
            metric_names=self.metric_names,
            refit_model=self.refit_on_update,
            use_input_warping=self.use_input_warping,
            use_loocv_pseudo_likelihood=self.use_loocv_pseudo_likelihood,
            **self._kwargs,
        )

    def feature_importances(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Cannot calculate feature_importances without a fitted model"
            )
        elif isinstance(self.model, ModelListGP):
            models = self.model.models
        else:
            models = [self.model]
        lengthscales = []
        for m in models:
            ls = m.covar_module.base_kernel.lengthscale
            if ls.ndim == 2:
                ls = ls.unsqueeze(0)
            lengthscales.append(ls)
        lengthscales = torch.cat(lengthscales, dim=0)
        # pyre-fixme[16]: `float` has no attribute `detach`.
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        return (1 / lengthscales).detach().cpu().numpy()


def get_rounding_func(
    rounding_func: Optional[Callable[[Tensor], Tensor]]
) -> Optional[Callable[[Tensor], Tensor]]:
    if rounding_func is None:
        botorch_rounding_func = rounding_func
    else:
        # make sure rounding_func is properly applied to q- and t-batches
        def botorch_rounding_func(X: Tensor) -> Tensor:
            batch_shape, d = X.shape[:-1], X.shape[-1]
            X_round = torch.stack(
                [rounding_func(x) for x in X.view(-1, d)]  # pyre-ignore: [16]
            )
            return X_round.view(*batch_shape, d)

    return botorch_rounding_func
