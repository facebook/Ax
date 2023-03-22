#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
import operator
import warnings
from functools import partial, reduce
from itertools import product
from logging import Logger
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, SearchSpaceExhausted
from ax.models.model_utils import enumerate_discrete_combinations, mk_discrete_choices
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    _tensor_difference,
    get_post_processing_func,
)
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
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.models.model import Model, ModelDict
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_mixed,
)
from torch import Tensor


DUPLICATE_TOL = 1e-6
MAX_CHOICES_ENUMERATE = 100_000

logger: Logger = get_logger(__name__)


class Acquisition(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch `AcquisitionFunction`, subcomponent
    of `BoTorchModel` and is not meant to be used outside of it.

    Args:
        surrogates: Dict of name => Surrogate model pairs, with which this acquisition
            function will be used.
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

    surrogates: Dict[str, Surrogate]
    acqf: AcquisitionFunction
    options: Dict[str, Any]

    def __init__(
        self,
        # If using multiple Surrogates, must label primary Surrogate (typically the
        # regression Surrogate) Keys.PRIMARY_SURROGATE
        surrogates: Dict[str, Surrogate],
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.surrogates = surrogates
        self.options = options or {}

        # Compute pending and observed points for each surrogate
        Xs_pending_and_observed = {
            name: _get_X_pending_and_observed(
                Xs=surrogate.Xs,
                objective_weights=torch_opt_config.objective_weights,
                bounds=search_space_digest.bounds,
                pending_observations=torch_opt_config.pending_observations,
                outcome_constraints=torch_opt_config.outcome_constraints,
                linear_constraints=torch_opt_config.linear_constraints,
                fixed_features=torch_opt_config.fixed_features,
            )
            for name, surrogate in self.surrogates.items()
        }

        Xs_pending_list = [
            Xs_pending
            for Xs_pending, _ in Xs_pending_and_observed.values()
            if Xs_pending is not None
        ]
        unique_Xs_pending = (
            torch.unique(
                input=torch.cat(
                    tensors=Xs_pending_list,
                    dim=0,
                ),
                dim=0,
            )
            if len(Xs_pending_list) > 0
            else None
        )

        # This tensor may have some Xs that are also in pending (because they are
        # observed for some models but not others)
        Xs_observed_maybe_pending_list = [
            Xs_observed
            for _, Xs_observed in Xs_pending_and_observed.values()
            if Xs_observed is not None
        ]
        unique_Xs_observed_maybe_pending = (
            torch.unique(
                input=torch.cat(
                    tensors=Xs_observed_maybe_pending_list,
                    dim=0,
                ),
                dim=0,
            )
            if len(Xs_observed_maybe_pending_list) > 0
            else None
        )

        # If a point is pending on any model do not count it as observed.
        # Do this by stacking pending on top of observed, filtering repeats, then
        # removing pending points.
        # TODO[sdaulton] Is this a sound approach? Should we be doing something more
        # sophisticated here?
        if unique_Xs_pending is None and unique_Xs_observed_maybe_pending is None:
            unique_Xs_observed = None
        elif unique_Xs_pending is None:
            unique_Xs_observed = unique_Xs_observed_maybe_pending
        else:
            unique_Xs_observed = _tensor_difference(
                A=unique_Xs_pending, B=unique_Xs_observed_maybe_pending
            )

            if torch.numel(unique_Xs_observed_maybe_pending) != torch.numel(
                unique_Xs_observed
            ):
                logger.warning(
                    "Encountered Xs pending for some Surrogates but observed for "
                    "others. Considering these points to be pending."
                )

        # Store objective thresholds for all outcomes (including non-objectives).
        self._objective_thresholds: Optional[
            Tensor
        ] = torch_opt_config.objective_thresholds
        self._full_objective_weights: Tensor = torch_opt_config.objective_weights
        full_outcome_constraints = torch_opt_config.outcome_constraints

        # TODO[mpolson64] Handle more elegantly in the future. Since right now we
        # only use one objective and posterior_transform this should be fine.
        primary_surrogate = (
            self.surrogates[Keys.PRIMARY_SURROGATE]
            if len(self.surrogates) > 1
            else next(iter(self.surrogates.values()))
        )

        primary_Xs_pending, primary_Xs_observed = Xs_pending_and_observed[
            Keys.PRIMARY_SURROGATE
            if len(self.surrogates) > 1
            else next(iter(Xs_pending_and_observed.keys()))
        ]

        # Subset model only to the outcomes we need for the optimization.
        if self.options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=primary_surrogate.model,
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
            model = primary_surrogate.model
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints
            objective_thresholds = torch_opt_config.objective_thresholds
            subset_idcs = None
        # If objective weights suggest multiple objectives but objective
        # thresholds are not specified, infer them using the model that
        # has already been subset to avoid re-subsetting it within
        # `inter_objective_thresholds`.
        if objective_weights.nonzero().numel() > 1 and (
            self._objective_thresholds is None
            or self._objective_thresholds[torch_opt_config.objective_weights != 0]
            .isnan()
            .any()
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
                X_observed=primary_Xs_observed,
                subset_idcs=subset_idcs,
                objective_thresholds=self._objective_thresholds,
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
            X_observed=primary_Xs_observed,
            risk_measure=torch_opt_config.risk_measure,
        )

        model_deps = self.compute_model_dependencies(
            surrogates=self.surrogates,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                torch_opt_config,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                objective_thresholds=objective_thresholds,
            ),
            options=self.options,
        )

        acqf_model_kwarg = (
            {
                "model_dict": ModelDict(
                    **{
                        name: surrogate.model
                        for name, surrogate in self.surrogates.items()
                    }
                )
            }
            if len(self.surrogates) > 1
            else {"model": model}
        )

        input_constructor_kwargs = {
            "X_baseline": unique_Xs_observed,
            "X_pending": unique_Xs_pending,
            "objective_thresholds": objective_thresholds,
            "outcome_constraints": outcome_constraints,
            "target_fidelities": search_space_digest.target_fidelities,
            "bounds": search_space_digest.bounds,
            **acqf_model_kwarg,
            **model_deps,
            **self.options,
        }
        input_constructor = get_acqf_input_constructor(botorch_acqf_class)
        # Handle multi-dataset surrogates - TODO: Improve this
        # If there is only one SupervisedDataset return it alone
        if (
            len(self.surrogates) == 1
            and len(next(iter(self.surrogates.values())).training_data) == 1
        ):
            training_data = next(iter(self.surrogates.values())).training_data[0]
        else:
            tdicts = (
                dict(zip(not_none(surrogate._outcomes), surrogate.training_data))
                for surrogate in self.surrogates.values()
            )
            # outcome_name => Dataset
            training_data = functools.reduce(lambda x, y: {**x, **y}, tdicts)

        acqf_inputs = input_constructor(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            **input_constructor_kwargs,
        )
        self.acqf = botorch_acqf_class(**acqf_inputs)  # pyre-ignore [45]
        self.X_pending: Optional[Tensor] = unique_Xs_pending
        self.X_observed: Tensor = not_none(unique_Xs_observed)

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class underlying this ``Acquisition``."""
        return self.acqf.__class__

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        dtypes = {
            label: surrogate.dtype for label, surrogate in self.surrogates.items()
        }

        dtypes_list = list(dtypes.values())
        if dtypes_list.count(dtypes_list[0]) != len(dtypes_list):
            raise ValueError(  # pragma: no cover
                f"Expected all Surrogates to have same dtype, found {dtypes}"
            )

        return dtypes_list[0]

    @property
    def device(self) -> Optional[torch.device]:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """

        devices = {
            label: surrogate.device for label, surrogate in self.surrogates.items()
        }

        devices_list = list(devices.values())
        if devices_list.count(devices_list[0]) != len(devices_list):
            raise ValueError(  # pragma: no cover
                f"Expected all Surrogates to have same device, found {devices}"
            )

        return devices_list[0]

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
            A two-element tuple containing an `n x d`-dim tensor of generated candidates
            and a tensor with the associated acquisition value.
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
        post_processing_func = get_post_processing_func(
            rounding_func=rounding_func,
            optimizer_options=optimizer_options_with_defaults,
        )
        discrete_features = sorted(ssd.ordinal_features + ssd.categorical_features)
        if fixed_features is not None:
            for i in fixed_features:
                if not 0 <= i < len(ssd.feature_names):
                    raise ValueError(f"Invalid fixed_feature index: {i}")

        # 1. Handle the fully continuous search space.
        if not discrete_features or optimizer_options_with_defaults.pop(
            "force_use_optimize_acqf", False
        ):
            return optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=n,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
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

            discrete_opt_options = optimizer_argparse(
                self.acqf,
                bounds=bounds,
                q=n,
                optimizer_options=optimizer_options,
                optimizer_is_discrete=True,
            )
            return optimize_acqf_discrete(
                acq_function=self.acqf, q=n, choices=all_choices, **discrete_opt_options
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
            post_processing_func=post_processing_func,
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
        surrogates: Mapping[str, Surrogate],
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
            surrogates: Mapping from names to Surrogate objects containing BoTorch
                `Model`s, with which this `Acquisition` is to be used.
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
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            risk_measure=risk_measure,
        )
