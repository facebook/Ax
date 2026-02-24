#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
References

.. [Daulton2026bonsai]
    S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. BONSAI: Bayesian
    Optimization with Natural Simplicity and Interpretability. ArXiv, 2026.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
from logging import Logger
from typing import Any

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError, DataRequiredError, SearchSpaceExhausted
from ax.generators.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.generators.torch.botorch_modular.surrogate import Surrogate
from ax.generators.torch.botorch_modular.utils import (
    _fix_map_key_to_target,
    _objective_threshold_to_outcome_constraints,
)
from ax.generators.torch.botorch_moo_utils import infer_objective_thresholds
from ax.generators.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
    subset_model,
)
from ax.generators.torch_base import TorchOptConfig
from ax.generators.utils import enumerate_discrete_combinations, mk_discrete_choices
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogProbabilityOfFeasibility
from botorch.acquisition.multioutput_acquisition import MultiOutputAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import BotorchError, InputDataError
from botorch.generation.sampling import SamplingStrategy
from botorch.models.model import Model
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_mixed,
)
from botorch.optim.optimize_mixed import (
    MAX_CARDINALITY_FOR_LOCAL_SEARCH,
    MAX_CHOICES_ENUMERATE,
    optimize_acqf_mixed_alternating,
    should_use_mixed_alternating_optimizer,
)
from botorch.optim.parameter_constraints import evaluate_feasibility
from botorch.utils.constraints import get_outcome_constraint_transforms
from pyre_extensions import none_throws
from torch import Tensor

try:
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

except ImportError:
    optimize_with_nsgaii = None


logger: Logger = get_logger(__name__)


def determine_optimizer(
    search_space_digest: SearchSpaceDigest,
    acqf: AcquisitionFunction | None = None,
    discrete_choices: Mapping[int, Sequence[float]] | None = None,
) -> str:
    """Determine the optimizer to use for a given search space.

    Args:
        search_space_digest: A SearchSpaceDigest object containing search space
            properties, e.g. ``bounds`` for optimization.
        acqf: The acquisition function to be used.
        discrete_choices: A dictionary mapping indices of discrete (ordinal
            or categorical) parameters to their respective sets of values
            provided as a list. This excludes fixed features.
    Returns:
        The name of the optimizer to use for the given search space.
    """
    if acqf is not None and isinstance(acqf, MultiOutputAcquisitionFunction):
        return "optimize_with_nsgaii"
    ssd = search_space_digest
    discrete_features = sorted(ssd.ordinal_features + ssd.categorical_features)
    if discrete_choices is None:
        discrete_choices = {}

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
            # for all parameters, we will use local search. Otherwise, we will use
            # the mixed alternating optimizer, which may use continuous relaxation
            # for the high cardinality parameters, while using local search for
            # the remaining parameters.
            cardinalities = [len(c) for c in discrete_choices.values()]
            max_cardinality = max(cardinalities)
            total_discrete_choices = reduce(operator.mul, cardinalities)
            if total_discrete_choices > MAX_CHOICES_ENUMERATE:
                if max_cardinality <= MAX_CARDINALITY_FOR_LOCAL_SEARCH:
                    optimizer = "optimize_acqf_discrete_local_search"
                else:
                    optimizer = "optimize_acqf_mixed_alternating"
            else:
                optimizer = "optimize_acqf_discrete"
        else:
            # For mixed (not fully discrete) search spaces, use the shared utility
            # from BoTorch to determine whether to use mixed alternating optimizer.
            if should_use_mixed_alternating_optimizer(discrete_dims=discrete_choices):
                optimizer = "optimize_acqf_mixed_alternating"
            else:
                optimizer = "optimize_acqf_mixed"
    return optimizer


class Acquisition(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch `AcquisitionFunction`, subcomponent
    of `BoTorchGenerator` and is not meant to be used outside of it.

    Args:
        surrogate: The Surrogate model, with which this acquisition
            function will be used.
        search_space_digest: A SearchSpaceDigest object containing metadata
            about the search space (e.g. bounds, parameter types).
        torch_opt_config: A TorchOptConfig object containing optimization
            arguments (e.g., objective weights, constraints).
        botorch_acqf_class: Type of BoTorch `AcquisitionFunction` that
            should be used.
        botorch_acqf_options: Optional mapping of kwargs to the underlying
            `AcquisitionFunction` in BoTorch.
        botorch_acqf_classes_with_options: A list of tuples of botorch
            `AcquisitionFunction` classes and dicts of kwargs, passed to
            the botorch `AcquisitionFunction`. This is used to specify
            multiple acquisition functions to be used with MultiAcquisition.
        n: The number of candidates that will be generated by this acquisition.
        options: Optional mapping of kwargs to the underlying `Acquisition
            Function` in BoTorch.
    """

    surrogate: Surrogate
    acqf: AcquisitionFunction
    _model: Model
    _objective_weights: Tensor
    _objective_thresholds: Tensor | None
    _outcome_constraints: tuple[Tensor, Tensor] | None
    _learned_objective_preference_model: Model | None
    _subset_idcs: Tensor | None
    _pruning_target_point: Tensor | None
    num_pruned_dims: list[int] | None

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction] | None,
        botorch_acqf_options: dict[str, Any] | None = None,
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ]
        | None = None,
        n: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.options: dict[str, Any] = options or {}
        botorch_acqf_options = botorch_acqf_options or {}
        self.search_space_digest = search_space_digest
        self.n = n
        self._should_subset_model: bool = self.options.get(Keys.SUBSET_MODEL, True)
        self._learned_objective_preference_model = None
        self._subset_idcs = None
        self._pruning_target_point = torch_opt_config.pruning_target_point
        self.num_pruned_dims = None

        # Extract pending and observed points.
        # We fix MAP_KEY to the fixed value (if given) to avoid points getting
        # discarded due to metrics being observed at different progressions.
        Xs = _fix_map_key_to_target(
            Xs=surrogate.Xs,
            feature_names=search_space_digest.feature_names,
            fixed_features=torch_opt_config.fixed_features,
        )
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )
        self.X_pending: Tensor | None = X_pending
        self.X_observed: Tensor | None = X_observed

        # Store objective thresholds for all outcomes (including non-objectives).
        self._full_objective_thresholds: Tensor | None = (
            torch_opt_config.objective_thresholds
        )
        self._full_objective_weights: Tensor = torch_opt_config.objective_weights

        (
            self._model,
            self._objective_weights,
            self._outcome_constraints,
            self._objective_thresholds,
        ) = self._subset_model(
            model=surrogate.model,
            objective_weights=torch_opt_config.objective_weights,
            outcome_constraints=torch_opt_config.outcome_constraints,
            objective_thresholds=torch_opt_config.objective_thresholds,
        )
        self._update_objective_thresholds(torch_opt_config=torch_opt_config)
        self._set_preference_model(torch_opt_config=torch_opt_config)

        if not botorch_acqf_classes_with_options:
            if botorch_acqf_class is None:
                raise AxError(
                    "One of botorch_acqf_class or botorch_acqf_classes_with_options"
                    " is required."
                )
            botorch_acqf_classes_with_options = [
                (botorch_acqf_class, botorch_acqf_options)
            ]
        self.botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ] = none_throws(botorch_acqf_classes_with_options)
        self._instantiate_acquisition()

    def _subset_model(
        self,
        model: Model,
        objective_weights: Tensor,
        outcome_constraints: tuple[Tensor, Tensor] | None = None,
        objective_thresholds: Tensor | None = None,
    ) -> tuple[Model, Tensor, tuple[Tensor, Tensor] | None, Tensor | None]:
        if not self._should_subset_model:
            return model, objective_weights, outcome_constraints, objective_thresholds
        # Otherwise, subset
        subset_model_results = subset_model(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
        )
        if self._subset_idcs is None:
            self._subset_idcs = subset_model_results.indices
        elif not torch.equal(subset_model_results.indices, self._subset_idcs):
            raise ValueError(
                "Subsequent model subsetting inconsistent with earlier."
            )  # pragma: no cover
        return (
            subset_model_results.model,
            subset_model_results.objective_weights,
            subset_model_results.outcome_constraints,
            subset_model_results.objective_thresholds,
        )

    def _update_objective_thresholds(self, torch_opt_config: TorchOptConfig) -> None:
        """If MOO and some objective thresholds are not specified, infer them using
        the model that has already been subset to avoid re-subsetting it within
        `infer_objective_thresholds`.

        If risk measures are used, objective thresholds must be provided. If not,
        this will error out.

        If `infer_objective_thresholds` errors out, e.g., due to no feasible point,
        this will log an error and let the optimization continue. Not all acquisition
        functions require objective thresholds, so this is not necessarily a problem.
        """
        if not (
            torch_opt_config.is_moo
            and (
                self._full_objective_thresholds is None
                or self._full_objective_thresholds[
                    torch_opt_config.outcome_mask
                ]
                .isnan()
                .any()
            )
            and self.X_observed is not None
        ):
            return
        try:
            self._full_objective_thresholds = infer_objective_thresholds(
                model=self._model,
                objective_weights=self._full_objective_weights,
                X_observed=self.X_observed,
                outcome_constraints=torch_opt_config.outcome_constraints,
                subset_idcs=self._subset_idcs,
                objective_thresholds=self._full_objective_thresholds,
            )
            self._objective_thresholds = (
                none_throws(self._full_objective_thresholds)[self._subset_idcs]
                if self._subset_idcs is not None
                else self._full_objective_thresholds
            )
        except (AxError, BotorchError) as e:
            logger.warning(
                "Failed to infer objective thresholds. Resuming optimization "
                "without objective thresholds, which may or may not work depending "
                f"on the acquisition function. Original error: {e}."
            )

    def _set_preference_model(self, torch_opt_config: TorchOptConfig) -> None:
        if torch_opt_config.use_learned_objective:
            if (Keys.PAIRWISE_PREFERENCE_QUERY.value,) not in self.surrogate._submodels:
                raise DataRequiredError(
                    "PreferenceOptimizationConfig is used but missing "
                    "preference objective model. Double check if the preference "
                    "exploration auxiliary experiment has data."
                )
            self._learned_objective_preference_model = self.surrogate._submodels[
                (Keys.PAIRWISE_PREFERENCE_QUERY.value,)
            ]

    def _instantiate_acquisition(self) -> None:
        """Constructs the acquisition function based on the provided
        botorch_acqf_classes_with_options.

        """
        if len(self.botorch_acqf_classes_with_options) != 1:
            raise ValueError("Only one botorch_acqf_class is supported.")
        botorch_acqf_class, botorch_acqf_options = (
            self.botorch_acqf_classes_with_options[0]
        )
        self.acqf = self._construct_botorch_acquisition(
            botorch_acqf_class=botorch_acqf_class,
            botorch_acqf_options=botorch_acqf_options,
            model=self._model,
        )
        self.models_used = [self.surrogate.model_name_by_metric]
        self.acq_function_sequence = None

    def _construct_botorch_acquisition(
        self,
        botorch_acqf_class: type[AcquisitionFunction],
        botorch_acqf_options: dict[str, Any],
        model: Model,
    ) -> AcquisitionFunction:
        objective, posterior_transform = self.get_botorch_objective_and_transform(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=self._objective_weights,
            outcome_constraints=self._outcome_constraints,
            X_observed=self.X_observed,
            learned_objective_preference_model=self._learned_objective_preference_model,
        )
        # Build constraint transforms, combining outcome constraints with
        # objective threshold-derived constraints when using
        # qLogProbabilityOfFeasibility for MOO.
        outcome_constraints = self._outcome_constraints
        constraint_transforms = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        if (
            issubclass(botorch_acqf_class, qLogProbabilityOfFeasibility)
            and self._objective_thresholds is not None
        ):
            threshold_constraints = _objective_threshold_to_outcome_constraints(
                objective_weights=self._objective_weights,
                objective_thresholds=self._objective_thresholds,
            )
            threshold_transforms = get_outcome_constraint_transforms(
                outcome_constraints=threshold_constraints
            )
            if constraint_transforms is not None and threshold_transforms is not None:
                constraint_transforms = constraint_transforms + threshold_transforms
            elif threshold_transforms is not None:
                constraint_transforms = threshold_transforms
        input_constructor_kwargs = {
            "model": model,
            "X_baseline": self.X_observed,
            "X_pending": self.X_pending,
            "objective_thresholds": self._objective_thresholds,
            "constraints": constraint_transforms,
            "constraints_tuple": self._outcome_constraints,
            "objective": objective,
            "posterior_transform": posterior_transform,
            **botorch_acqf_options,
        }
        target_fidelities = {
            k: v
            for k, v in self.search_space_digest.target_values.items()
            if k in self.search_space_digest.fidelity_features
        }
        if len(target_fidelities) > 0:
            input_constructor_kwargs["target_fidelities"] = target_fidelities
        input_constructor = get_acqf_input_constructor(botorch_acqf_class)

        # Extract the training data from the surrogate.
        # If there is a single dataset, this will be the dataset itself.
        # If there are multiple datasets, this will be a dict mapping the outcome names
        # to the corresponding datasets.
        training_data = self.surrogate.training_data
        if len(training_data) == 1:
            training_data = training_data[0]
        else:
            training_data = dict(
                zip(none_throws(self.surrogate._outcomes), training_data)
            )
        acqf_inputs = input_constructor(
            training_data=training_data,
            bounds=self.search_space_digest.bounds,
            **{k: v for k, v in input_constructor_kwargs.items() if v is not None},
        )
        return botorch_acqf_class(**acqf_inputs)  # pyre-ignore [45]

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
        return self._full_objective_thresholds

    @property
    def objective_weights(self) -> Tensor | None:
        """The objective weights for all outcomes."""
        return self._full_objective_weights

    def select_from_candidate_set(
        self,
        n: int,
        candidate_set: Tensor,
        sampling_strategy: SamplingStrategy | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select n candidates from a discrete set with optional weight allocation.

        This method selects candidates from ``candidate_set`` using either a
        ``SamplingStrategy`` (e.g., Thompson Sampling with win-counting for weight
        allocation) or greedy acquisition function optimization.

        ``candidate_set`` is the stable interface for any candidate generation
        method. Any method that produces candidates (in-sample training data,
        pathwise TS optimization, user-provided sets, etc.) feeds into this
        parameter. The selection/weight-allocation logic is agnostic to how
        candidates were generated.

        Args:
            n: The number of candidates to select.
            candidate_set: A ``(num_choices, d)`` tensor of discrete candidate
                points to select from.
            sampling_strategy: An optional BoTorch ``SamplingStrategy`` instance
                (e.g., ``MaxPosteriorSampling`` for Thompson Sampling, or
                ``BoltzmannSampling`` for acquisition-weighted sampling). When
                provided, candidates are selected by sampling from ``candidate_set``
                according to the strategy. When ``num_samples > n``, win-counting
                mode is used: many posterior samples are drawn, wins are counted
                per candidate, and the top-n candidates are returned with weights
                proportional to their win probability (normalized to sum to 1).
                If not provided, greedy acquisition function selection is used via
                ``optimize_acqf_discrete``.

        Returns:
            A three-element tuple containing an ``n x d``-dim tensor of selected
            candidates, a tensor with the associated acquisition values, and a
            tensor with the weight for each candidate (normalized to sum to 1
            for win-counting mode, or uniform for direct/greedy selection).

        Raises:
            ValueError: If ``candidate_set`` is empty or has fewer points than
                ``n``.
        """
        if candidate_set.shape[0] == 0:
            raise ValueError(
                "`candidate_set` is empty. Provide a non-empty set of candidates."
            )
        if candidate_set.shape[0] < n:
            raise ValueError(
                f"`candidate_set` has {candidate_set.shape[0]} candidates, "
                f"but {n} were requested. Provide at least {n} candidates."
            )

        if sampling_strategy is not None:
            # Check if this is a win-counting strategy (e.g., Thompson Sampling)
            # or a direct selection strategy (e.g., Boltzmann Sampling).
            # If num_samples is explicitly set and > n, use win-counting mode.
            # Otherwise, use direct selection mode.
            num_samples_attr = getattr(sampling_strategy, "num_samples", None)
            num_samples: int | None = (
                int(num_samples_attr) if num_samples_attr is not None else None
            )

            if num_samples is not None and num_samples > n:
                # Win-counting mode: sample many times, count wins, return top-n
                # with weights proportional to win counts (normalized to sum to 1).
                sampled_candidates = sampling_strategy(
                    candidate_set.unsqueeze(0), num_samples=num_samples
                ).squeeze(0)  # (num_samples, d)

                # Count wins for each unique candidate
                unique_candidates, inverse_indices = torch.unique(
                    sampled_candidates, dim=0, return_inverse=True
                )
                counts = torch.bincount(
                    inverse_indices, minlength=unique_candidates.shape[0]
                )

                # Select top-n candidates by win count.
                # When num_unique < n (fewer unique winners than requested),
                # we return all unique winners. The caller should handle
                # candidates.shape[0] <= n, consistent with
                # optimize_acqf_discrete which may also return fewer than n.
                num_unique = unique_candidates.shape[0]
                top_n = min(n, num_unique)
                top_counts, top_indices = torch.topk(counts, top_n)

                candidates = unique_candidates[top_indices]
                arm_weights = top_counts.to(dtype=self.dtype, device=self.device)
                arm_weights = arm_weights / arm_weights.sum()
            else:
                # Direct selection mode: sample exactly n candidates with equal
                # weights. Used for strategies like BoltzmannSampling where
                # weighting is built into the selection process.
                sampled_candidates = sampling_strategy(
                    candidate_set.unsqueeze(0), num_samples=n
                ).squeeze(0)  # (n, d)
                candidates = sampled_candidates
                arm_weights = torch.ones(n, dtype=self.dtype, device=self.device)

            acqf_values = self.evaluate(candidates.unsqueeze(1)).view(-1)
            return candidates, acqf_values, arm_weights

        # Greedy selection from provided discrete candidate set via acqf.
        # optimize_acqf_discrete may return fewer than n candidates when
        # there are fewer feasible choices; arm_weights matches actual count.
        candidates, acqf_values = optimize_acqf_discrete(
            acq_function=self.acqf,
            q=n,
            choices=candidate_set,
            unique=True,
        )
        arm_weights = torch.ones(
            candidates.shape[0], dtype=self.dtype, device=self.device
        )
        return candidates, acqf_values, arm_weights

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
        candidate_set: Tensor | None = None,
        sampling_strategy: SamplingStrategy | None = None,
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
                `Adapter` to ensure compatibility of the candidates with
                with Ax transforms. For additional post processing, use
                `post_processing_func` option in `optimizer_options`.
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``. This can also include a `post_processing_func`
                which is applied to the candidates before the `rounding_func`.
                `post_processing_func` can be used to support more customized options
                that typically only exist in MBM, such as BoTorch transforms.
                See the docstring of `TorchOptConfig` for more information on passing
                down these options while constructing a generation strategy.
            candidate_set: An optional tensor of shape `(num_choices, d)` containing
                discrete candidate points to select from instead of optimizing over
                the search space. When provided, selection is delegated to
                ``select_from_candidate_set``. This enables in-sample candidate
                generation when set to the training data (X_observed).
            sampling_strategy: An optional BoTorch ``SamplingStrategy`` instance
                (e.g., ``MaxPosteriorSampling`` for Thompson Sampling, or
                ``BoltzmannSampling`` for acquisition-weighted sampling).
                Passed to ``select_from_candidate_set`` when ``candidate_set``
                is provided. Requires ``candidate_set`` to be provided.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        # Dispatch to candidate set selection if candidate_set or
        # sampling_strategy is provided.
        if sampling_strategy is not None or candidate_set is not None:
            if candidate_set is None:
                raise ValueError(
                    "`candidate_set` is required when using `sampling_strategy`. "
                    "Provide the discrete set of candidates to sample from."
                )
            return self.select_from_candidate_set(
                n=n,
                candidate_set=candidate_set,
                sampling_strategy=sampling_strategy,
            )

        # Options that would need to be passed in the transformed space are
        # disallowed, since this would be very difficult for an end user to do
        # directly, and someone who uses BoTorch at this level of detail would
        # probably be better off using BoTorch directly.
        # `return_best_only` and `return_full_tree` are disallowed because
        # Ax expects `optimize_acqf` to return tensors of a certain shape.
        if optimizer_options is not None:
            forbidden_optimizer_options = [
                "equality_constraints",
                "inequality_constraints",  # These should be constructed by Ax
                "batch_initial_conditions",
                "return_best_only",
                "return_full_tree",
            ]

            for kw in optimizer_options:
                if kw in forbidden_optimizer_options:
                    raise ValueError(
                        f"Argument {kw} is not allowed in `optimizer_options`."
                    )

        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()
        discrete_choices = mk_discrete_choices(ssd=ssd, fixed_features=fixed_features)

        optimizer = determine_optimizer(
            search_space_digest=ssd,
            discrete_choices=discrete_choices,
            acqf=self.acqf,
        )
        # `raw_samples` and `num_restarts` are not supported by
        # `optimize_acqf_discrete`.
        if (
            optimizer in ("optimize_acqf_discrete", "optimize_with_nsgaii")
            and optimizer_options is not None
        ):
            optimizer_options.pop("raw_samples", None)
            optimizer_options.pop("num_restarts", None)

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
                acq_function_sequence=self.acq_function_sequence,
                **optimizer_options_with_defaults,
            )

        # 2. Handle fully discrete search spaces.
        elif optimizer in (
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
                n_candidates = candidates.shape[0]
                return (
                    candidates,
                    acqf_values,
                    arm_weights[:n_candidates] * n_candidates / n,
                )

            # Else, optimizer is `optimize_acqf_discrete`
            # Enumerate all possible choices
            all_choices = (discrete_choices[i] for i in range(len(discrete_choices)))
            all_choices = _tensorize(tuple(product(*all_choices)))
            try:
                max_batch_size = optimizer_options_with_defaults.pop(
                    "max_batch_size", 2048
                )
                try:
                    # Adapt max batch size for batched models to reduce peak memory.
                    max_batch_size = (
                        max_batch_size // self.surrogate.model.batch_shape.numel()
                    )
                except Exception:  # pragma: no cover
                    pass  # Do not error out if the model does not have a batch shape.
                candidates, acqf_values = optimize_acqf_discrete(
                    acq_function=self.acqf,
                    q=n,
                    choices=all_choices,
                    max_batch_size=max_batch_size,
                    X_avoid=X_observed,
                    inequality_constraints=inequality_constraints,
                    **optimizer_options_with_defaults,
                )
            except InputDataError:
                raise SearchSpaceExhausted(
                    "No more feasible choices in a fully discrete search space."
                )

        # 3. Handle mixed search spaces that have discrete and continuous features.
        elif optimizer == "optimize_acqf_mixed":
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
            # NOTE: We intentially use `ssd.discrete_choices` as opposed to
            # `discrete_choices`. This optimizer checks whether `discrete_dims` and
            # `cat_dims` match with the bounds. However, `discrete_choices` has
            # overridden some values in `ssd.discrete_choices` using `fixed_features`,
            # which would fail the check. This optimizer is able to handle fixed
            # features itself. No need for overriding.
            candidates, acqf_values = optimize_acqf_mixed_alternating(
                acq_function=self.acqf,
                bounds=bounds,
                discrete_dims={
                    k: list(v)
                    for k, v in ssd.discrete_choices.items()
                    if k in ssd.ordinal_features
                },
                cat_dims={
                    k: list(v)
                    for k, v in ssd.discrete_choices.items()
                    if k in ssd.categorical_features
                },
                q=n,
                post_processing_func=rounding_func,
                fixed_features=fixed_features,
                inequality_constraints=inequality_constraints,
                **optimizer_options_with_defaults,
            )
        elif optimizer == "optimize_with_nsgaii":
            if optimize_with_nsgaii is not None:
                # TODO: support post_processing_func
                candidates, acqf_values = optimize_with_nsgaii(
                    acq_function=self.acqf,
                    bounds=bounds,
                    q=n,
                    fixed_features=fixed_features,
                    # We use pyre-ignore here to avoid a circular import.
                    # pyre-ignore [6]: Incompatible parameter type [6]: In call `len`,
                    # for 1st positional argument, expected
                    # `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor,
                    # Module]`.
                    num_objectives=len(self.acqf.acqfs),
                    **optimizer_options_with_defaults,
                )
            else:
                raise AxError(
                    "optimize_with_nsgaii requires botorch to be installed with "
                    "the pymoo."
                )
        else:
            raise AxError(  # pragma: no cover
                f"Unknown optimizer: {optimizer}. This code should be unreachable."
            )
        # prune irrelevant parameters post-hoc
        if self.options.get("prune_irrelevant_parameters", False) and not isinstance(
            self.acqf, qLogProbabilityOfFeasibility
        ):
            if self._pruning_target_point is None:
                logger.info(
                    "Must specify pruning_target_point to prune irrelevant "
                    "parameters. Skipping pruning irrelevant parameters."
                )
            else:
                candidates, acqf_values = self._prune_irrelevant_parameters(
                    candidates=candidates,
                    search_space_digest=search_space_digest,
                    inequality_constraints=inequality_constraints,
                    fixed_features=fixed_features,
                )
        n_candidates = candidates.shape[0]
        return candidates, acqf_values, arm_weights[:n_candidates] * n_candidates / n

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
        outcome_constraints: tuple[Tensor, Tensor] | None = None,
        X_observed: Tensor | None = None,
        learned_objective_preference_model: Model | None = None,
    ) -> tuple[MCAcquisitionObjective | None, PosteriorTransform | None]:
        return get_botorch_objective_and_transform(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            learned_objective_preference_model=learned_objective_preference_model,
        )

    def _condition_on_prev_candidates(
        self, prev_candidates: Tensor, initial_X_pending: Tensor | None
    ) -> None:
        """Condition the acquisition function on the candidates.

        Args:
            prev_candidates: A `q' x d`-dim Tensor of candidates.
            initial_X_pending: A `q'' x d`-dim Tensor of pending points. These are
                points that are the initial pending evaluations and should be added
                to the `X_pending` of the acquisition function.
        """
        if prev_candidates.shape[0] > 0:
            if initial_X_pending is not None and initial_X_pending.shape[0]:
                self.X_pending = torch.cat([initial_X_pending, prev_candidates], dim=0)
            else:
                self.X_pending = prev_candidates
            self._instantiate_acquisition()

    def _compute_baseline_acqf_value(
        self, last_candidate: Tensor, fixed_features: dict[int, float] | None = None
    ) -> float:
        r"""Compute the baseline acquisition function value.

        If this is the first point, the baseline AF value is the max AF value
        the previously evaluated points. The otherwise the baseline value is the
        acquisition value of the last point after conditioning on the previously
        selected points including the last candidate. For incremental AFs, this is
        will be near zero. For non-incremental AFs, this will be the AF value of the
        the batch consisting of the previously selected points (since they are
        pending).

        Args:
            last_candidate: A `1 x d`-dim Tensor containing the last candidate.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.

        Returns:
            The baseline acquisition function value.
        """
        if last_candidate.shape[0] > 0:
            X = last_candidate
        elif self.X_observed is not None:
            # if this is the first candidate, compute the acquisition value for
            # the previously evaluated set of points. Take the max and compute
            # the incremental value with respect to that baseline value.
            X = self.X_observed.unsqueeze(1)
            if fixed_features is not None:
                for idx, v in fixed_features.items():
                    X[..., idx] = v
        else:
            return 0.0
        with torch.no_grad():
            base_af_val = self.evaluate(X=X)
        base_af_val = base_af_val.exp() if self.acqf._log else base_af_val
        return base_af_val.max().item()

    def _prune_irrelevant_parameters(
        self,
        candidates: Tensor,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Prune irrelevant parameters from the candidates using BONSAI.

        See [Daulton2026bonsai]_ for details.

        The method involves first optimizing the AF without any notion of irrelevance.
        Then, the irrelevant parameters are pruned via a sequential greedy algorithm
        that sets each dimension to the target point value, and recomputes the AF value.
        The dimension that corresponds to the smallest reduction in AF value compared to
        the AF value of the dense original candidate is then set to the target value,
        and the algorithm proceeds until removing any dimension causes the reduction in
        the AF value of the proposed point to be less than some tolerance, relative to
        the original dense point.

        For `q > 1` cases, this is repeated for each point in the batch, after adding
        the previously pruned points as pending points. The incremental AF value is
        used in the stopping rule.

        Args:
            candidates: A `q x d`-dim Tensor of candidates.
            search_space_digest: The SearchSpaceDigest.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
                `coefficients` should be torch tensors. See the docstring of
                `make_scipy_linear_constraints` for an example. When q=1, or when
                applying the same constraint to each candidate in the batch
                (intra-point constraint), `indices` should be a 1-d tensor.
                For inter-point constraints, in which the constraint is applied to the
                whole batch of candidates, `indices` must be a 2-d tensor, where
                in each row `indices[i] =(k_i, l_i)` the first index `k_i` corresponds
                to the `k_i`-th element of the `q`-batch and the second index `l_i`
                corresponds to the `l_i`-th feature of that element.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.

        Returns:
            A two-element tuple containing an `q x d`-dim tensor of generated
            candidates and a `q`-dim tensor with the associated acquisition values.
        """
        target_point = none_throws(
            self._pruning_target_point,
            message="Must specify pruning_target_point to prune irrelevant parameters",
        )
        irrelevance_pruning_rtol = self.options.get("irrelevance_pruning_rtol", 0.2)
        initial_X_pending = self.X_pending
        pruned_af_vals = []
        excluded_indices = set(search_space_digest.fidelity_features)
        excluded_indices.update(set(search_space_digest.task_features))
        if fixed_features is not None:
            excluded_indices.update(set(fixed_features.keys()))
        orig_acqf = self.acqf
        # iterate over points in the batch and prune each point
        initial_active_dims = (candidates != target_point).sum(dim=-1)
        for i in range(candidates.shape[0]):
            # condition on previous pruned points
            self._condition_on_prev_candidates(
                prev_candidates=candidates[:i], initial_X_pending=initial_X_pending
            )
            # compute the AF value for the previous candidates so that we can compute
            # the incremental EI
            base_af_val = self._compute_baseline_acqf_value(
                last_candidate=candidates[i - 1 : i], fixed_features=fixed_features
            )
            # compute the AF value of the dense candidate (before pruning)
            with torch.no_grad():
                dense_af_val = self.evaluate(X=candidates[i : i + 1])
            # compute the incremental AF value of the dense candidate, relative to the
            # base AF value
            dense_incremental_af_val = (
                ((dense_af_val.exp() if self.acqf._log else dense_af_val) - base_af_val)
                .clamp_min(0.0)
                .item()
            )
            # In the base case that no parameters are pruned, the final AF val is the
            # dense AF val
            final_af_val = dense_af_val
            # If the current incremental AF value is zero, then we skip pruning
            if dense_incremental_af_val > 0.0:
                remaining_indices = set(range(candidates.shape[-1])) - excluded_indices
                # remove features that are already set to target_point
                remaining_indices -= set(
                    (candidates[i] == target_point).nonzero().view(-1).tolist()
                )
                # len(remaining_indices) - 1 is used here so that we do not prune
                # every dimension
                for _ in range(len(remaining_indices) - 1):
                    indices = torch.tensor(
                        list(remaining_indices),
                        dtype=torch.long,
                        device=candidates.device,
                    )
                    # create a `b x 1 x d`-dim tensor, where each batch has the
                    # original candidate with a different dimension set to the
                    # corresponding target value
                    pruned_candidates = _expand_and_set_single_feature_to_target(
                        X=candidates[i : i + 1],
                        indices=indices,
                        targets=target_point[indices],
                    )
                    # remove candidates that violate constraints after pruning
                    pruned_candidates, indices = _remove_infeasible_candidates(
                        candidates=pruned_candidates,
                        indices=indices,
                        inequality_constraints=inequality_constraints,
                    )
                    if pruned_candidates.shape[0] == 0:
                        # no feasible points, continue to
                        # next candidate
                        break
                    # determine which dimension to prune based on the reduction
                    # in incremental AF value relative to original dense candidate
                    min_idx, best_af_val = self._get_best_pruned_candidate(
                        pruned_candidates=pruned_candidates,
                        irrelevance_pruning_rtol=irrelevance_pruning_rtol,
                        dense_incremental_af_val=dense_incremental_af_val,
                        base_af_val=base_af_val,
                    )

                    if min_idx is not None:
                        candidates[i] = pruned_candidates[min_idx]
                        remaining_indices.remove(
                            int(none_throws(indices)[min_idx].item())
                        )
                        final_af_val = best_af_val
                    else:
                        # if min_idx is None that means that no candidate met
                        # the relative tolerance threshold, so we stop pruning
                        # this candidate
                        break
            pruned_af_vals.append(final_af_val)
        self.acqf = orig_acqf
        # store the number of pruned dimensions for each candidate to return
        # this via gen_metadata
        self.num_pruned_dims = (
            initial_active_dims - (candidates != target_point).sum(dim=-1)
        ).tolist()

        return candidates, torch.cat(pruned_af_vals, dim=0)

    def _get_best_pruned_candidate(
        self,
        pruned_candidates: Tensor,
        irrelevance_pruning_rtol: float,
        dense_incremental_af_val: float,
        base_af_val: float,
    ) -> tuple[int | None, Tensor | None]:
        """Determine which pruned candidate is best.

        This computes the relative change in incremental acquisition value for each
        pruned candidate and returns the index and candidate with the smallest
        reduction. If no candidate meets the relative tolerance-based stopping rule,
        then Nones are returned instead of an index and candidate.

        Args:
            pruned_candidates: A `b x 1 x d`-dim tensor of pruned candidates.
            irrelevance_pruning_rtol: The relative tolerance on the reduction in
                incremental AF value, relative to the dense candidate.
            dense_incremental_af_val: The incremental AF value for the dense
                candidate.
            base_af_val: The baseline AF value that should be subtracted from the AF
                value for each pruned candidate to determine the incremental value.

        Returns:
            A two-element tuple containing the index and corresponding candidate.
        """
        with torch.no_grad():
            new_af_val = torch.cat(
                [self.evaluate(X=X_) for X_ in pruned_candidates.split(1024)], dim=0
            )
        if self.acqf._log:
            non_log_new_af_val = new_af_val.exp()
        else:
            non_log_new_af_val = new_af_val
        af_reduction = (
            dense_incremental_af_val - (non_log_new_af_val - base_af_val).clamp_min(0.0)
        ) / dense_incremental_af_val

        min_af_reduction, min_idx = af_reduction.min(dim=0)
        if min_af_reduction <= irrelevance_pruning_rtol:
            return min_idx.item(), new_af_val[min_idx].view(1)
        return None, None


def _expand_and_set_single_feature_to_target(
    X: Tensor, indices: Tensor, targets: Tensor
) -> Tensor:
    """Return an expanded copy of X, using target values at the specified indices.

    In the returned copy X2, for each `i` in `indices`, `X2[i, 0, indices[i]]` is
    set to the corresponding target value in `targets`.

    Args:
        X: A `1 x d`-dim tensor of points.
        indices: A `k`-dim tensor of indices with values in [0, d).
        targets: A `k`-dim tensor of target values.

    Returns:
        A `k x 1 x d`-dim tensor copy of X updated to have the target values
            at the specified indices.
    """
    d = X.shape[1]
    k = indices.shape[0]
    X2 = X.unsqueeze(0).expand(k, 1, d).clone()
    q_idxr = torch.zeros((k, 1), dtype=torch.long, device=X.device)
    # indices expanded to (k, 1)
    indices_exp = indices.unsqueeze(1)
    # targets expanded to (k, 1)
    targets_exp = targets.unsqueeze(1)
    # Assign targets to X2 at the specified indices
    X2[torch.arange(k, device=X.device).unsqueeze(1), q_idxr, indices_exp] = targets_exp
    return X2


def _remove_infeasible_candidates(
    candidates: Tensor,
    indices: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Filter out infeasible candidates, based on the parameter constraints.

    Args:
        candidates: A `b x 1 x d`-dim tensor of candidates.
        indices: A `b`-dim tensor of indices, where the values are integers
            in [0, d-1).
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example. When q=1, or when
            applying the same constraint to each candidate in the batch
            (intra-point constraint), `indices` should be a 1-d tensor.
            For inter-point constraints, in which the constraint is applied to the
            whole batch of candidates, `indices` must be a 2-d tensor, where
            in each row `indices[i] =(k_i, l_i)` the first index `k_i` corresponds
            to the `k_i`-th element of the `q`-batch and the second index `l_i`
            corresponds to the `l_i`-th feature of that element.

    Returns:
        A two-element tuple containing the filter candidates and indices.
    """
    if inequality_constraints is not None:
        is_feasible = evaluate_feasibility(
            X=candidates,
            inequality_constraints=inequality_constraints,
        )
        candidates = candidates[is_feasible]
        indices = indices[is_feasible]
    return candidates, indices
