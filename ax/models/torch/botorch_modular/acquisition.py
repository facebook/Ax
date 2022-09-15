#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import operator
import warnings
from functools import partial, reduce
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, SearchSpaceExhausted
from ax.models.model_utils import enumerate_discrete_combinations, mk_discrete_choices
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
    subset_model,
)
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.models.model import Model
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_mixed,
)
from torch import Tensor


DUPLICATE_TOL = 1e-6
MAX_CHOICES_ENUMERATE = 100_000


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
        search_space_digest: A SearchSpaceDigest object containing metadata
            about the search space (e.g. bounds, parameter types).
        torch_opt_config: A TorchOptConfig object containing optimization
            arguments (e.g., objective weights, constraints).
        botorch_acqf_class: Type of BoTorch `AcquistitionFunction` that
            should be used. Subclasses of `Acquisition` often specify
            these via `default_botorch_acqf_class` attribute, in which
            case specifying one here is not required.
        options: Optional mapping of kwargs to the underlying `Acquisition
            Function` in BoTorch.
    """

    surrogate: Surrogate
    acqf: AcquisitionFunction

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.surrogate = surrogate
        # pyre-fixme[4]: Attribute must be annotated.
        self.options = options or {}
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.surrogate.Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )
        # Store objective thresholds for all outcomes (including non-objectives).
        self._objective_thresholds: Optional[
            Tensor
        ] = torch_opt_config.objective_thresholds
        self._full_objective_weights: Tensor = torch_opt_config.objective_weights
        full_outcome_constraints = torch_opt_config.outcome_constraints
        # Subset model only to the outcomes we need for the optimization.
        if self.options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=self.surrogate.model,
                objective_weights=torch_opt_config.objective_weights,
                outcome_constraints=torch_opt_config.outcome_constraints,
                objective_thresholds=torch_opt_config.objective_thresholds,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
            objective_thresholds = subset_model_results.objective_thresholds
            subset_idcs = subset_model_results.indices
        else:
            model = self.surrogate.model
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints
            objective_thresholds = torch_opt_config.objective_thresholds
            subset_idcs = None
        # If objective weights suggest multiple objectives but objective
        # thresholds are not specified, infer them using the model that
        # has already been subset to avoid re-subsetting it within
        # `inter_objective_thresholds`.
        if (
            objective_weights.nonzero().numel() > 1
            and self._objective_thresholds is None
        ):
            if torch_opt_config.risk_measure is not None:
                # TODO[T131759263]: modify the heuristic to support risk measures.
                raise NotImplementedError(  # pragma: no cover
                    "Objective thresholds must be provided when using risk measures."
                )
            self._objective_thresholds = infer_objective_thresholds(
                model=model,
                objective_weights=self._full_objective_weights,
                outcome_constraints=full_outcome_constraints,
                X_observed=X_observed,
                subset_idcs=subset_idcs,
            )
            objective_thresholds = (
                not_none(self._objective_thresholds)[subset_idcs]
                if subset_idcs is not None
                else self._objective_thresholds
            )
        objective, posterior_transform = self.get_botorch_objective_and_transform(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            risk_measure=torch_opt_config.risk_measure,
        )
        model_deps = self.compute_model_dependencies(
            surrogate=surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                torch_opt_config,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                objective_thresholds=objective_thresholds,
            ),
            options=self.options,
        )
        input_constructor_kwargs = {
            "X_baseline": X_observed,
            "X_pending": X_pending,
            "objective_thresholds": objective_thresholds,
            "outcome_constraints": outcome_constraints,
            "target_fidelities": search_space_digest.target_fidelities,
            "bounds": search_space_digest.bounds,
            **model_deps,
            **self.options,
        }
        input_constructor = get_acqf_input_constructor(botorch_acqf_class)
        # Handle multi-dataset surrogates - TODO: Improve this
        if len(self.surrogate.training_data) == 1:
            training_data = self.surrogate.training_data[0]
        else:
            training_data = dict(
                zip(not_none(self.surrogate._outcomes), self.surrogate.training_data)
            )
        acqf_inputs = input_constructor(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            **input_constructor_kwargs,
        )
        self.acqf = botorch_acqf_class(**acqf_inputs)  # pyre-ignore [45]
        self.X_pending: Optional[Tensor] = X_pending
        self.X_observed: Tensor = not_none(X_observed)

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class underlying this ``Acquisition``."""
        return self.acqf.__class__

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.dtype

    @property
    def device(self) -> Optional[torch.device]:
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

    @property
    def objective_weights(self) -> Optional[Tensor]:
        """The objective weights for all outcomes."""
        return self._full_objective_weights

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
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
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
        # NOTE: Could make use of `optimizer_class` when it's added to BoTorch
        # instead of calling `optimizer_acqf` or `optimize_acqf_discrete` etc.
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            bounds=bounds,
            q=n,
            optimizer_options=optimizer_options,
        )

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
            X_observed = self.X_observed
            if self.X_pending is not None:
                X_observed = torch.cat([X_observed, self.X_pending], dim=0)

            # Special handling for search spaces with a large number of choices
            total_choices = reduce(
                operator.mul, [float(len(c)) for c in discrete_choices.values()]
            )
            if total_choices > MAX_CHOICES_ENUMERATE:
                discrete_choices = [
                    torch.tensor(c, device=self.device, dtype=self.dtype)
                    for c in discrete_choices.values()
                ]
                return optimize_acqf_discrete_local_search(
                    acq_function=self.acqf,
                    q=n,
                    discrete_choices=discrete_choices,
                    inequality_constraints=inequality_constraints,
                    X_avoid=X_observed,
                    **optimizer_options_with_defaults,
                )

            # Enumerate all possible choices
            all_choices = (discrete_choices[i] for i in range(len(discrete_choices)))
            all_choices = _tensorize(tuple(product(*all_choices)))

            # This can be vectorized, but using a for-loop to avoid memory issues
            for x in X_observed:
                all_choices = all_choices[
                    (all_choices - x).abs().max(dim=-1).values > DUPLICATE_TOL
                ]

            # Filter out candidates that violate the constraints
            # TODO: It will be more memory-efficient to do this filtering before
            # converting the generator into a tensor. However, if we run into memory
            # issues we are likely better off being smarter in how we optimize the
            # acquisition function.
            inequality_constraints = inequality_constraints or []
            is_feasible = torch.ones(all_choices.shape[0], dtype=torch.bool)
            for (inds, weights, bound) in inequality_constraints:
                is_feasible &= (all_choices[..., inds] * weights).sum(dim=-1) >= bound
            all_choices = all_choices[is_feasible]

            num_choices = all_choices.size(dim=0)
            if num_choices == 0:
                raise SearchSpaceExhausted(
                    "No more feasible choices in a fully discrete search space."
                )
            if num_choices < n:
                warnings.warn(
                    (
                        f"Requested n={n} candidates from fully discrete search "
                        f"space, but only {num_choices} possible choices remain."
                    ),
                    AxWarning,
                )
                n = num_choices

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
        if isinstance(self.acqf, qKnowledgeGradient):
            return self.acqf.evaluate(X=X)
        else:
            # NOTE: `AcquisitionFunction.__call__` calls `forward`,
            # so below is equivalent to `self.acqf.forward(X=X)`.
            return self.acqf(X=X)

    def compute_model_dependencies(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
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
            search_space_digest: A SearchSpaceDigest object containing metadata
                about the search space (e.g. bounds, parameter types).
            torch_opt_config: A TorchOptConfig object containing optimization
                arguments (e.g., objective weights, constraints).
            options: The `options` kwarg dict, passed on initialization of
                the `Acquisition` object.

        Returns: A dictionary of surrogate model-dependent options, to be passed
            as kwargs to BoTorch`AcquisitionFunction` constructor.
        """
        return {}

    def get_botorch_objective_and_transform(
        self,
        botorch_acqf_class: Type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
        risk_measure: Optional[RiskMeasureMCObjective] = None,
    ) -> Tuple[Optional[MCAcquisitionObjective], Optional[PosteriorTransform]]:
        return get_botorch_objective_and_transform(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            X_observed=X_observed,
            risk_measure=risk_measure,
        )
