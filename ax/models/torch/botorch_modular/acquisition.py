#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from functools import partial, reduce
from itertools import product
from logging import Logger
from typing import Any

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError, SearchSpaceExhausted
from ax.models.model_utils import (
    all_ordinal_features_are_integer_valued,
    enumerate_discrete_combinations,
    mk_discrete_choices,
)
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
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.exceptions.errors import InputDataError
from botorch.models.model import Model
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_mixed,
)
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from botorch.utils.constraints import get_outcome_constraint_transforms
from pyre_extensions import none_throws
from torch import Tensor


# For fully discrete search spaces.
MAX_CHOICES_ENUMERATE = 100_000
MAX_CARDINALITY_FOR_LOCAL_SEARCH = 100
# For mixed search spaces.
ALTERNATING_OPTIMIZER_THRESHOLD = 10

logger: Logger = get_logger(__name__)


class Acquisition(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch `AcquisitionFunction`, subcomponent
    of `BoTorchModel` and is not meant to be used outside of it.

    Args:
        surrogate: The Surrogate model, with which this acquisition
            function will be used.
        search_space_digest: A SearchSpaceDigest object containing metadata
            about the search space (e.g. bounds, parameter types).
        torch_opt_config: A TorchOptConfig object containing optimization
            arguments (e.g., objective weights, constraints).
        botorch_acqf_class: Type of BoTorch `AcquisitionFunction` that
            should be used.
        options: Optional mapping of kwargs to the underlying `Acquisition
            Function` in BoTorch.
    """

    surrogate: Surrogate
    acqf: AcquisitionFunction
    options: dict[str, Any]

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.options = options or {}

        # Extract pending and observed points.
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=surrogate.Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )

        # Store objective thresholds for all outcomes (including non-objectives).
        self._objective_thresholds: Tensor | None = (
            torch_opt_config.objective_thresholds
        )
        self._full_objective_weights: Tensor = torch_opt_config.objective_weights
        full_outcome_constraints = torch_opt_config.outcome_constraints

        # Subset model only to the outcomes we need for the optimization.
        if self.options.pop(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=surrogate.model,
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
            model = surrogate.model
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints
            objective_thresholds = torch_opt_config.objective_thresholds
            subset_idcs = None

        # If MOO and some objective thresholds are not specified, infer them using
        # the model that has already been subset to avoid re-subsetting it within
        # `infer_objective_thresholds`.
        if (
            torch_opt_config.is_moo
            and (
                self._objective_thresholds is None
                or self._objective_thresholds[torch_opt_config.objective_weights != 0]
                .isnan()
                .any()
            )
            and X_observed is not None
        ):
            if torch_opt_config.risk_measure is not None:
                raise NotImplementedError(
                    "Objective thresholds must be provided when using risk measures."
                )
            self._objective_thresholds = infer_objective_thresholds(
                model=model,
                objective_weights=self._full_objective_weights,
                outcome_constraints=full_outcome_constraints,
                X_observed=X_observed,
                subset_idcs=subset_idcs,
                objective_thresholds=self._objective_thresholds,
            )
            objective_thresholds = (
                none_throws(self._objective_thresholds)[subset_idcs]
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
        target_fidelities = {
            k: v
            for k, v in search_space_digest.target_values.items()
            if k in search_space_digest.fidelity_features
        }
        input_constructor_kwargs = {
            "model": model,
            "X_baseline": X_observed,
            "X_pending": X_pending,
            "objective_thresholds": objective_thresholds,
            "constraints": get_outcome_constraint_transforms(
                outcome_constraints=outcome_constraints
            ),
            "objective": objective,
            "posterior_transform": posterior_transform,
            **self.options,
        }

        if len(target_fidelities) > 0:
            input_constructor_kwargs["target_fidelities"] = target_fidelities

        input_constructor = get_acqf_input_constructor(botorch_acqf_class)

        # Extract the training data from the surrogate.
        # If there is a single dataset, this will be the dataset itself.
        # If there are multiple datasets, this will be a dict mapping the outcome names
        # to the corresponding datasets.
        training_data = surrogate.training_data
        if len(training_data) == 1:
            training_data = training_data[0]
        else:
            training_data = dict(zip(none_throws(surrogate._outcomes), training_data))

        acqf_inputs = input_constructor(
            training_data=training_data,
            bounds=search_space_digest.bounds,
            **{k: v for k, v in input_constructor_kwargs.items() if v is not None},
        )
        self.acqf = botorch_acqf_class(**acqf_inputs)  # pyre-ignore [45]
        self.X_pending: Tensor | None = X_pending
        self.X_observed: Tensor | None = X_observed

    @property
    def botorch_acqf_class(self) -> type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class underlying this ``Acquisition``."""
        return self.acqf.__class__

    @property
    def dtype(self) -> torch.dtype | None:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.dtype

    @property
    def device(self) -> torch.device | None:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.device

    @property
    def objective_thresholds(self) -> Tensor | None:
        """The objective thresholds for all outcomes.

        For non-objective outcomes, the objective thresholds are nans.
        """
        return self._objective_thresholds

    @property
    def objective_weights(self) -> Tensor | None:
        """The objective weights for all outcomes."""
        return self._full_objective_weights

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
                result appropriately. This is typically passed down from
                `ModelBridge` to ensure compatibility of the candidates with
                with Ax transforms. For additional post processing, use
                `post_processing_func` option in `optimizer_options`.
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``. This can also include a `post_processing_func`
                which is applied to the candidates before the `rounding_func`.
                `post_processing_func` can be used to support more customized options
                that typically only exist in MBM, such as BoTorch transforms.
                See the docstring of `TorchOptConfig` for more information on passing
                down these options while constructing a generation strategy.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()
        discrete_features = sorted(ssd.ordinal_features + ssd.categorical_features)
        discrete_choices = mk_discrete_choices(ssd=ssd, fixed_features=fixed_features)

        if len(discrete_features) == 0:
            optimizer = "optimize_acqf"
        else:
            fully_discrete = len(discrete_choices) == len(ssd.feature_names)
            if fully_discrete:
                # One of the three optimizers may be used depending on the number of
                # discrete choices and the cardinality of individual parameters.
                # If there are less than `MAX_CHOICES_ENUMERATE` choices, we will
                # evaluate all of them and pick the best.
                # If there are less than `MAX_CARDINALITY_FOR_LOCAL_SEARCH` choices
                # for all parameters, if there are any categorical features, or if
                # any of the parameters has non-integer valued choices, we will use
                # local search.  Otherwise, we will use the mixed alternating optimizer,
                # which may use continuous relaxation for the high cardinality
                # parameters, while using local search for the remaining parameters.
                cardinalities = [len(c) for c in discrete_choices.values()]
                max_cardinality = max(cardinalities)
                total_discrete_choices = reduce(operator.mul, cardinalities)
                if total_discrete_choices > MAX_CHOICES_ENUMERATE:
                    if (
                        max_cardinality <= MAX_CARDINALITY_FOR_LOCAL_SEARCH
                        or len(ssd.categorical_features) > 0
                        or not all_ordinal_features_are_integer_valued(ssd=ssd)
                    ):
                        optimizer = "optimize_acqf_discrete_local_search"
                    else:
                        optimizer = "optimize_acqf_mixed_alternating"
                else:
                    optimizer = "optimize_acqf_discrete"
                    # `raw_samples` and `num_restarts` are not supported by
                    # `optimize_acqf_discrete`.
                    if optimizer_options is not None:
                        optimizer_options.pop("raw_samples", None)
                        optimizer_options.pop("num_restarts", None)
            else:
                n_combos = math.prod([len(v) for v in discrete_choices.values()])
                # If there are
                # - any categorical features (except for those handled by transforms),
                # - any ordinal features with non-integer choices,
                # - or less than `ALTERNATING_OPTIMIZER_THRESHOLD` combinations
                # of discrete choices, we will use `optimize_acqf_mixed`, which
                # enumerates all discrete combinations and optimizes the continuous
                # features with discrete features being fixed. Otherwise, we will
                # use `optimize_acqf_mixed_alternating`, which alternates between
                # continuous and discrete optimization steps.
                if (
                    n_combos <= ALTERNATING_OPTIMIZER_THRESHOLD
                    or len(ssd.categorical_features) > 0
                    or not all_ordinal_features_are_integer_valued(ssd=ssd)
                ):
                    optimizer = "optimize_acqf_mixed"
                else:
                    optimizer = "optimize_acqf_mixed_alternating"

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            optimizer_options=optimizer_options,
            optimizer=optimizer,
        )
        if fixed_features is not None:
            for i in fixed_features:
                if not 0 <= i < len(ssd.feature_names):
                    raise ValueError(f"Invalid fixed_feature index: {i}")
        # Return a weight of 1 for each arm by default. This can be
        # customized in subclasses if necessary.
        arm_weights = torch.ones(n, dtype=self.dtype)
        # 1. Handle the fully continuous search space.
        if optimizer == "optimize_acqf":
            candidates, acqf_values = optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=n,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=rounding_func,
                **optimizer_options_with_defaults,
            )
            return candidates, acqf_values, arm_weights

        # 2. Handle fully discrete search spaces.
        if optimizer in (
            "optimize_acqf_discrete",
            "optimize_acqf_discrete_local_search",
        ):
            X_observed = self.X_observed
            if self.X_pending is not None:
                if X_observed is None:
                    X_observed = self.X_pending
                else:
                    X_observed = torch.cat([X_observed, self.X_pending], dim=0)

            # Special handling for search spaces with a large number of choices
            if optimizer == "optimize_acqf_discrete_local_search":
                discrete_choices = [
                    torch.tensor(c, device=self.device, dtype=self.dtype)
                    for c in discrete_choices.values()
                ]
                candidates, acqf_values = optimize_acqf_discrete_local_search(
                    acq_function=self.acqf,
                    q=n,
                    discrete_choices=discrete_choices,
                    inequality_constraints=inequality_constraints,
                    X_avoid=X_observed,
                    **optimizer_options_with_defaults,
                )
                return candidates, acqf_values, arm_weights

            # Else, optimizer is `optimize_acqf_discrete`
            # Enumerate all possible choices
            all_choices = (discrete_choices[i] for i in range(len(discrete_choices)))
            all_choices = _tensorize(tuple(product(*all_choices)))
            try:
                candidates, acqf_values = optimize_acqf_discrete(
                    acq_function=self.acqf,
                    q=n,
                    choices=all_choices,
                    X_avoid=X_observed,
                    inequality_constraints=inequality_constraints,
                    **optimizer_options_with_defaults,
                )
            except InputDataError:
                raise SearchSpaceExhausted(
                    "No more feasible choices in a fully discrete search space."
                )
            return candidates, acqf_values, arm_weights

        # 3. Handle mixed search spaces that have discrete and continuous features.
        if optimizer == "optimize_acqf_mixed":
            candidates, acqf_values = optimize_acqf_mixed(
                acq_function=self.acqf,
                bounds=bounds,
                q=n,
                fixed_features_list=enumerate_discrete_combinations(
                    discrete_choices=discrete_choices
                ),
                inequality_constraints=inequality_constraints,
                post_processing_func=rounding_func,
                **optimizer_options_with_defaults,
            )
        elif optimizer == "optimize_acqf_mixed_alternating":
            candidates, acqf_values = optimize_acqf_mixed_alternating(
                acq_function=self.acqf,
                bounds=bounds,
                discrete_dims=search_space_digest.ordinal_features,
                q=n,
                post_processing_func=rounding_func,
                fixed_features=fixed_features,
                inequality_constraints=inequality_constraints,
                **optimizer_options_with_defaults,
            )
        else:
            raise AxError(  # pragma: no cover
                f"Unknown optimizer: {optimizer}. This code should be unreachable."
            )
        return candidates, acqf_values, arm_weights

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

    def get_botorch_objective_and_transform(
        self,
        botorch_acqf_class: type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Tensor | None = None,
        outcome_constraints: tuple[Tensor, Tensor] | None = None,
        X_observed: Tensor | None = None,
        risk_measure: RiskMeasureMCObjective | None = None,
    ) -> tuple[MCAcquisitionObjective | None, PosteriorTransform | None]:
        return get_botorch_objective_and_transform(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            risk_measure=risk_measure,
        )
