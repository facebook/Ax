#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.exceptions.core import AxError
from ax.models.torch.botorch import (
    BotorchModel,
    TAcqfConstructor,
    TBestPointRecommender,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
    get_rounding_func,
)
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    predict_from_model,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.botorch_moo_defaults import (
    get_NEHVI,
    pareto_frontier_evaluator,
    scipy_optimizer_list,
    infer_objective_thresholds,
)
from ax.models.torch.frontier_utils import TFrontierEvaluator
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    randomize_objective_weights,
    subset_model,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


logger = get_logger(__name__)

TOptimizerList = Callable[
    [
        List[AcquisitionFunction],
        Tensor,
        Optional[List[Tuple[Tensor, Tensor, float]]],
        Optional[Dict[int, float]],
        Optional[Callable[[Tensor], Tensor]],
        Any,
    ],
    Tuple[Tensor, Tensor],
]


class MultiObjectiveBotorchModel(BotorchModel):
    r"""
    Customizable multi-objective model.

    By default, this uses an Expected Hypervolume Improvment function to find the
    pareto frontier of a function with multiple outcomes. This behavior
    can be modified by providing custom implementations of the following
    components:

    - a `model_constructor` that instantiates and fits a model on data
    - a `model_predictor` that predicts outcomes using the fitted model
    - a `acqf_constructor` that creates an acquisition function from a fitted model
    - a `acqf_optimizer` that optimizes the acquisition function

    Args:
        model_constructor: A callable that instantiates and fits a model on data,
            with signature as described below.
        model_predictor: A callable that predicts using the fitted model, with
            signature as described below.
        acqf_constructor: A callable that creates an acquisition function from a
            fitted model, with signature as described below.
        acqf_optimizer: A callable that optimizes an acquisition
            function, with signature as described below.



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

        frontier_evaluator(
            model,
            objective_weights,
            objective_thresholds,
            X,
            Y,
            Yvar,
            outcome_constraints,
        )

    Here `model` is a botorch `Model`, `objective_thresholds` is used in hypervolume
    evaluations, `objective_weights` is a tensor of weights applied to the  objectives
    (sign represents direction), `X`, `Y`, `Yvar` are tensors, `outcome_constraints` is
    a tuple of tensors describing the (linear) outcome constraints.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: List[Tensor]

    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model,
        model_predictor: TModelPredictor = predict_from_model,
        # pyre-fixme[9]: acqf_constructor has type `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor], Any],
        #  AcquisitionFunction]`; used as `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor],
        #  **(Any)], AcquisitionFunction]`.
        acqf_constructor: TAcqfConstructor = get_NEHVI,
        # pyre-fixme[9]: acqf_optimizer has type `Callable[[AcquisitionFunction,
        #  Tensor, int, Optional[Dict[int, float]], Optional[Callable[[Tensor],
        #  Tensor]], Any], Tensor]`; used as `Callable[[AcquisitionFunction, Tensor,
        #  int, Optional[Dict[int, float]], Optional[Callable[[Tensor], Tensor]],
        #  **(Any)], Tensor]`.
        acqf_optimizer: TOptimizer = scipy_optimizer,
        # TODO: Remove best_point_recommender for botorch_moo. Used in modelbridge._gen.
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        frontier_evaluator: TFrontierEvaluator = pareto_frontier_evaluator,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = False,
        use_input_warping: bool = False,
        use_loocv_pseudo_likelihood: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_constructor = model_constructor
        self.model_predictor = model_predictor
        self.acqf_constructor = acqf_constructor
        self.acqf_optimizer = acqf_optimizer
        self.best_point_recommender = best_point_recommender
        self.frontier_evaluator = frontier_evaluator
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

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,  # objective_directions
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        objective_thresholds: Optional[Tensor] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[Tensor, Tensor, TGenMetadata, Optional[List[TCandidateMetadata]]]:
        options = model_gen_options or {}
        acf_options = options.get("acquisition_function_kwargs", {})
        optimizer_options = options.get("optimizer_kwargs", {})

        if target_fidelities:
            raise NotImplementedError(
                "target_fidelities not implemented for base BotorchModel"
            )
        if (
            objective_thresholds is not None
            and objective_weights.shape[0] != objective_thresholds.shape[0]
        ):
            raise AxError(
                "Objective weights and thresholds most both contain an element for"
                " each modeled metric."
            )

        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            pending_observations=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )

        model = not_none(self.model)
        full_objective_thresholds = objective_thresholds
        full_objective_weights = objective_weights
        full_outcome_constraints = outcome_constraints
        # subset model only to the outcomes we need for the optimization
        if options.get(Keys.SUBSET_MODEL, True):
            full_objective_weights
            subset_model_results = subset_model(
                model=model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                objective_thresholds=objective_thresholds,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
            objective_thresholds = subset_model_results.objective_thresholds
            idcs = subset_model_results.indices
        else:
            idcs = None
        if objective_thresholds is None:
            full_objective_thresholds = infer_objective_thresholds(
                model=model,
                X_observed=not_none(X_observed),
                objective_weights=full_objective_weights,
                outcome_constraints=full_outcome_constraints,
                subset_idcs=idcs,
            )
            # subset the objective thresholds
            objective_thresholds = (
                full_objective_thresholds
                if idcs is None
                else full_objective_thresholds[idcs].clone()
            )

        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)
        botorch_rounding_func = get_rounding_func(rounding_func)
        if acf_options.get("random_scalarization", False) or acf_options.get(
            "chebyshev_scalarization", False
        ):
            # If using a list of acquisition functions, the algorithm to generate
            # that list is configured by acquisition_function_kwargs.
            objective_weights_list = [
                randomize_objective_weights(objective_weights, **acf_options)
                for _ in range(n)
            ]
            acquisition_function_list = [
                self.acqf_constructor(  # pyre-ignore: [28]
                    model=model,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    X_observed=X_observed,
                    X_pending=X_pending,
                    **acf_options,
                )
                for objective_weights in objective_weights_list
            ]
            acquisition_function_list = [
                checked_cast(AcquisitionFunction, acq_function)
                for acq_function in acquisition_function_list
            ]
            # Multiple acquisition functions require a sequential optimizer
            # always use scipy_optimizer_list.
            # TODO(jej): Allow any optimizer.
            candidates, expected_acquisition_value = scipy_optimizer_list(
                acq_function_list=acquisition_function_list,
                bounds=bounds_,
                inequality_constraints=_to_inequality_constraints(
                    linear_constraints=linear_constraints
                ),
                fixed_features=fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
        else:
            acquisition_function = self.acqf_constructor(  # pyre-ignore: [28]
                model=model,
                objective_weights=objective_weights,
                objective_thresholds=objective_thresholds,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
                **acf_options,
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
                    linear_constraints=linear_constraints
                ),
                fixed_features=fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
        gen_metadata = {
            "expected_acquisition_value": expected_acquisition_value.tolist(),
            "objective_thresholds": not_none(full_objective_thresholds).cpu(),
        }
        return (
            candidates.detach().cpu(),
            torch.ones(n, dtype=self.dtype),
            gen_metadata,
            None,
        )
