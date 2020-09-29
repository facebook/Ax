#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ax.core.types import TConfig
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective,
    subset_model,
)
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.equality import Base
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.model import Model
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
    # BoTorch `AcquisitionFunction` class associated with this `Acquisition`
    # instance. Determined during `__init__`, do not set manually.
    _botorch_acqf_class: Type[AcquisitionFunction]

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
        self._botorch_acqf_class = not_none(
            botorch_acqf_class or self.default_botorch_acqf_class
        )
        self.surrogate = surrogate
        self.options = options or {}
        trd = self._extract_training_data(surrogate=surrogate)
        Xs = [trd.X] if isinstance(trd, TrainingData) else [i.X for i in trd.values()]
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
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

        objective = self._get_botorch_objective(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )
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
        # pyre-ignore[28]: Some kwargs are not expected in base `Model`
        # but are expected in its subclasses.
        self.acqf = self._botorch_acqf_class(
            model=model,
            objective=objective,
            X_pending=X_pending,
            X_baseline=X_observed,
            **self.options,
            **model_deps,
            **self.compute_data_dependencies(training_data=trd),
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
        # NOTE: Could make use of `optimizer_class` when it's added to BoTorch.
        return optimize_acqf(
            self.acqf,
            bounds=bounds,
            q=n,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            post_processing_func=rounding_func,
            **optimizer_options,
        )

    def evaluate(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of acquisition values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        # NOTE: `AcquisitionFunction.__call__` calls `forward`,
        # so below is equivalent to `self.acqf.forward(X=X)`.
        return self.acqf(X=X)

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

        NOTE: When subclassing `Acquisition` from a superclass where this
        method returns a non-empty dictionary of kwargs to `AcquisitionFunction`,
        call `super().compute_model_dependencies` and then update that
        dictionary of options with the options for the subclass you are creating
        (unless the superclass' model dependencies should not be propagated to
        the subclass). See `MultiFidelityKnowledgeGradient.compute_model_dependencies`
        for an example.

        Args:
            surrogate: The surrogate object containing the BoTorch `Model`,
                with which this `Acquisition` is to be used.
            bounds: A list of (lower, upper) tuples for each column of X in
                the training data of the surrogate model.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
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
            options: The `options` kwarg dict, passed on initialization of
                the `Acquisition` object.

        Returns: A dictionary of surrogate model-dependent options, to be passed
            as kwargs to BoTorch`AcquisitionFunction` constructor.
        """
        return {}

    @classmethod
    def compute_data_dependencies(
        cls, training_data: Union[TrainingData, Dict[str, TrainingData]]
    ) -> Dict[str, Any]:
        """Computes inputs to acquisition function class based on the given
        data in model's training data.

        NOTE: May not be needed if model dependencies are handled entirely on
        the BoTorch side.

        Args:
            training_data: Either a `TrainingData` for 1 outcome, or a mapping of
                outcome name to respective `TrainingData` (if `ListSurrogate` is used).

        Returns: A dictionary of training data-dependent options, to be passed
            as kwargs to BoTorch`AcquisitionFunction` constructor.
        """
        return {}

    def _get_botorch_objective(
        self,
        model: Model,
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
    ) -> AcquisitionObjective:
        return get_botorch_objective(
            model=model,
            objective_weights=objective_weights,
            use_scalarized_objective=issubclass(
                self._botorch_acqf_class, AnalyticAcquisitionFunction
            ),
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )

    @classmethod
    def _extract_training_data(
        cls, surrogate: Surrogate
    ) -> Union[TrainingData, Dict[str, TrainingData]]:
        if isinstance(surrogate, ListSurrogate):
            return checked_cast(dict, surrogate.training_data_per_outcome)
        else:
            return checked_cast(TrainingData, surrogate.training_data)
