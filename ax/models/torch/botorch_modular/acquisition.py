#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TConfig
from ax.models.model_utils import (
    mk_discrete_choices,
    enumerate_discrete_combinations,
)
from ax.models.torch.botorch_modular.default_options import (
    get_default_optimizer_options,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective,
    subset_model,
)
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.optim.optimize import optimize_acqf_mixed, optimize_acqf_discrete
from torch import Tensor


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
        search_space_digest: A SearchSpaceDigest object containing
            metadata about the search space (e.g. bounds, parameter types).
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
    """

    surrogate: Surrogate
    acqf: AcquisitionFunction

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        objective_thresholds: Optional[Tensor] = None,
    ) -> None:
        self.surrogate = surrogate
        self.options = options or {}
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.surrogate.training_data.Xs,
            objective_weights=objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
        # store objective thresholds for all outcomes (including non-objectives)
        self._objective_thresholds = objective_thresholds
        full_objective_weights = objective_weights
        full_outcome_constraints = outcome_constraints
        # Subset model only to the outcomes we need for the optimization.
        if self.options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=self.surrogate.model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                objective_thresholds=objective_thresholds,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
            objective_thresholds = subset_model_results.objective_thresholds
            subset_idcs = subset_model_results.indices
        else:
            model = self.surrogate.model
            subset_idcs = None
        # If objective weights suggest multiple objectives but objective
        # thresholds are not specified, infer them using the model that
        # has already been subset to avoid re-subsetting it within
        # `inter_objective_thresholds`.
        if (
            objective_weights.nonzero().numel() > 1  # pyre-ignore [16]
            and self._objective_thresholds is None
        ):
            self._objective_thresholds = infer_objective_thresholds(
                model=model,
                objective_weights=full_objective_weights,
                outcome_constraints=full_outcome_constraints,
                X_observed=X_observed,
                subset_idcs=subset_idcs,
            )
            objective_thresholds = (
                not_none(self._objective_thresholds)[subset_idcs]
                if subset_idcs is not None
                else self._objective_thresholds
            )
        objective = self.get_botorch_objective(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )
        model_deps = self.compute_model_dependencies(
            surrogate=surrogate,
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            pending_observations=pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=self.options,
        )
        input_constructor_kwargs = {
            "X_baseline": X_observed,
            "X_pending": X_pending,
            "objective_thresholds": objective_thresholds,
            "outcome_constraints": outcome_constraints,
            **model_deps,
            **self.options,
        }
        input_constructor = get_acqf_input_constructor(botorch_acqf_class)
        acqf_inputs = input_constructor(
            model=model,
            training_data=self.surrogate.training_data,
            objective=objective,
            **input_constructor_kwargs,
        )
        self.acqf = botorch_acqf_class(**acqf_inputs)  # pyre-ignore [45]

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class underlying this ``Acquisition``."""
        return self.acqf.__class__

    @property
    def dtype(self) -> torch.dtype:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.dtype

    @property
    def device(self) -> torch.device:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.device

    @property
    def objective_thresholds(self) -> Optional[Tensor]:
        """The objective thresholds for all outcomes.

        For non-objective outcomes, the objective thresholds are nans.
        """
        return self._objective_thresholds

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.
        """
        # NOTE: Could make use of `optimizer_class` when its added to BoTorch
        # instead of calling `optimizer_acqf` or `optimize_acqf_discrete` etc.

        optimizer_options_with_defaults = {
            **get_default_optimizer_options(acqf_class=self.botorch_acqf_class),
            **(optimizer_options or {}),
        }
        ssd = search_space_digest
        tkwargs = {"dtype": self.dtype, "device": self.device}
        # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
        #  `Union[torch.device, torch.dtype]`.
        bounds = torch.tensor(ssd.bounds, **tkwargs).transpose(0, 1)
        discrete_features = sorted(ssd.ordinal_features + ssd.categorical_features)

        if fixed_features is not None:
            for i in fixed_features:
                if not 0 <= i < len(ssd.feature_names):
                    raise ValueError(f"Invalid fixed_feature index: {i}")

        # 1. Handle the fully continuous search space.
        if not discrete_features:
            return optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=n,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=rounding_func,
                **optimizer_options_with_defaults,
            )

        # 2. Handle search spaces with discrete features.
        discrete_choices = mk_discrete_choices(ssd=ssd, fixed_features=fixed_features)

        # 2a. Handle the fully discrete search space.
        if len(discrete_choices) == len(ssd.feature_names):
            # For now we just enumerate all possible discrete combinations. This is not
            # scalable and and only works for a reasonably small number of choices.
            all_choices = (discrete_choices[i] for i in range(len(discrete_choices)))
            # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
            #  `Union[torch.device, torch.dtype]`.
            all_choices = torch.tensor(list(itertools.product(*all_choices)), **tkwargs)

            return optimize_acqf_discrete(
                acq_function=self.acqf,
                q=n,
                choices=all_choices,
                **optimizer_options_with_defaults,
            )

        # 2b. Handle mixed search spaces that have discrete and continuous features.
        return optimize_acqf_mixed(
            acq_function=self.acqf,
            bounds=bounds,
            q=n,
            # For now we just enumerate all possible discrete combinations. This is not
            # scalable and and only works for a reasonably small number of choices. A
            # slowdown warning is logged in `enumerate_discrete_combinations` if needed.
            fixed_features_list=enumerate_discrete_combinations(
                discrete_choices=discrete_choices
            ),
            inequality_constraints=inequality_constraints,
            post_processing_func=rounding_func,
            **optimizer_options_with_defaults,
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

    @copy_doc(Surrogate.best_in_sample_point)
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, float]:
        return self.surrogate.best_in_sample_point(
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=options,
        )

    def compute_model_dependencies(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
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
            search_space_digest: A SearchSpaceDigest object containing
                metadata about the search space (e.g. bounds, parameter types).
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
            options: The `options` kwarg dict, passed on initialization of
                the `Acquisition` object.

        Returns: A dictionary of surrogate model-dependent options, to be passed
            as kwargs to BoTorch`AcquisitionFunction` constructor.
        """
        return {}

    def get_botorch_objective(
        self,
        botorch_acqf_class: Type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
    ) -> AcquisitionObjective:
        return get_botorch_objective(
            model=model,
            objective_weights=objective_weights,
            use_scalarized_objective=issubclass(
                botorch_acqf_class, AnalyticAcquisitionFunction
            ),
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            X_observed=X_observed,
        )
