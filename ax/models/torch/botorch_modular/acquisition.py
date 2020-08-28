#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ax.core.types import TConfig
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective,
    subset_model,
)
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.equality import Base
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.utils.containers import TrainingData
from torch import Tensor


class Optimizer:  # NOTE: Stub for future BoTorch optimizer class.
    pass


class Acquisition(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch `AcquisitionFunction`, subcomponent
    of `BoTorchModel` and is not meant to be used outside of it.

    Args:
        surrogate: Surrogate model, with which this acquisition function
            will be used.
        bounds: A list of (lower, upper) tuples for each column of X in
            the training data of the surrogate model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        botorch_acqf_class: Type of BoTorch `AcquistitionFunction` that
            should be used. Subclasses of `Acquisition` often specify
            these via `default_botorch_acqf_class` attribute, in which
            case specifying one here is not required.
        options: Optional mapping of kwargs to the underlying `Acquisition
            Function` in BoTorch.
        pending_observations: A list of tensors, each of which contains
            points whose evaluation is pending (i.e. that have been
            submitted for evaluation) for a given outcome. A list
            of m (k_i x d) feature tensors X for m outcomes and k_i,
            pending observations for outcome i.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        target_fidelities: Optional mapping from parameter name to its
            target fidelity, applicable to fidelity parameters only.
    """

    surrogate: Surrogate
    acqf: AcquisitionFunction
    # BoTorch `AcquisitionFunction` class associated with this `Acquisition`
    # class by default. `None` for the base `Acquisition` class, but can be
    # specified in subclasses.
    default_botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None

    def __init__(
        self,
        surrogate: Surrogate,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        options: Optional[Dict[str, Any]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> None:
        if not botorch_acqf_class and not self.default_botorch_acqf_class:
            raise ValueError(
                f"Acquisition class {self.__class__} does not specify a default "
                "BoTorch `AcquisitionFunction`, so `botorch_acqf_class` "
                "argument must be specified."
            )
        botorch_acqf_class = not_none(
            botorch_acqf_class or self.default_botorch_acqf_class
        )
        self.surrogate = surrogate
        self.options = options or {}
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.surrogate.training_data.Xs,
            pending_observations=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )

        # Subset model only to the outcomes we need for the optimization.
        if self.options.get(Keys.SUBSET_MODEL, True):
            model, objective_weights, outcome_constraints, _ = subset_model(
                self.surrogate.model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )
        else:
            model = self.surrogate.model

        objective = get_botorch_objective(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            use_scalarized_objective=issubclass(
                botorch_acqf_class, AnalyticAcquisitionFunction
            ),
        )
        # NOTE: Computing model dependencies might be handled entirely on
        # BoTorch side.
        model_deps = self.compute_model_dependencies(
            surrogate=surrogate,
            bounds=bounds,
            objective_weights=objective_weights,
            pending_observations=pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            options=self.options,
        )
        data_deps = self.compute_data_dependencies(
            training_data=self.surrogate.training_data
        )
        # pyre-ignore[28]: Some kwargs are not expected in base `Model`
        # but are expected in its subclasses.
        self.acqf = botorch_acqf_class(
            model=model,
            objective=objective,
            X_pending=X_pending,
            X_baseline=X_observed,
            **self.options,
            **model_deps,
            **data_deps,
        )

    def optimize(
        self,
        bounds: Tensor,
        n: int,
        optimizer_class: Optional[Optimizer] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.
        """
        optimizer_options = optimizer_options or {}
        # TODO: make use of `optimizer_class` when its added to BoTorch.
        return optimize_acqf(
            self.acqf,
            bounds=bounds,
            q=n,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            post_processing_func=rounding_func,
            **optimizer_options,
        )

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Argument `bounds` expected.
    @copy_doc(Surrogate.best_in_sample_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, float]:
        return self.surrogate.best_in_sample_point(
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=options,
        )

    @classmethod
    def compute_model_dependencies(
        cls,
        surrogate: Surrogate,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Computes inputs to acquisition function class based on the given
        surrogate model.

        NOTE: May not be needed if model dependencies are handled entirely on
        the BoTorch side.
        """
        return {}

    @classmethod
    def compute_data_dependencies(cls, training_data: TrainingData) -> Dict[str, Any]:
        """Computes inputs to acquisition function class based on the given
        data in model's training data.

        NOTE: May not be needed if model dependencies are handled entirely on
        the BoTorch side.
        """
        return {}
