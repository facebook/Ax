#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.modelbridge_utils import (
    get_weighted_mc_objective_and_objective_thresholds,
)
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_moo_defaults import DEFAULT_EHVI_MC_SAMPLES
from ax.models.torch.botorch_moo_defaults import get_default_partitioning_alpha
from ax.utils.common.typeutils import not_none
from botorch.acquisition import monte_carlo  # noqa F401
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import IIDNormalSampler
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils import get_outcome_constraint_transforms
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from torch import Tensor


class MOOAcquisition(Acquisition):
    default_botorch_acqf_class: Optional[
        Type[AcquisitionFunction]
    ] = qExpectedHypervolumeImprovement

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor],
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        options: Optional[Dict[str, Any]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> None:
        botorch_acqf_class = not_none(
            botorch_acqf_class or self.default_botorch_acqf_class
        )
        if not issubclass(botorch_acqf_class, qExpectedHypervolumeImprovement):
            raise UnsupportedError(
                "Only qExpectedHypervolumeImprovement is currently supported as "
                f"a MOOAcquisition botorch_acqf_class. Got: {botorch_acqf_class}."
            )

        super().__init__(
            surrogate=surrogate,
            botorch_acqf_class=botorch_acqf_class,
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
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
        return {
            "outcome_constraints": outcome_constraints,
        }

    def _get_botorch_objective(
        self,
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
    ) -> AcquisitionObjective:
        objective, _ = get_weighted_mc_objective_and_objective_thresholds(
            objective_weights=objective_weights,
            # pyre-ignore [6]: expected `Tensor` but got `Optional[Tensor]`
            objective_thresholds=objective_thresholds,
        )
        return objective

    def _instantiate_acqf(
        self,
        model: Model,
        objective: AcquisitionObjective,
        model_dependent_kwargs: Dict[str, Any],
        objective_thresholds: Optional[Tensor] = None,
        X_pending: Optional[Tensor] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:
        # Extract model dependent kwargs
        outcome_constraints = model_dependent_kwargs.pop("outcome_constraints")
        # Replicate `get_EHVI` transformation code
        X_observed = X_baseline
        if X_observed is None:
            raise ValueError("There are no feasible observed points.")
        if objective_thresholds is None:
            raise ValueError("Objective Thresholds required")
        with torch.no_grad():
            Y = model.posterior(X_observed).mean

        # For EHVI acquisition functions we pass the constraint transform directly.
        if outcome_constraints is None:
            cons_tfs = None
        else:
            cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        num_objectives = objective_thresholds.shape[0]

        mc_samples = self.options.get("mc_samples", DEFAULT_EHVI_MC_SAMPLES)
        qmc = self.options.get("qmc", True)
        alpha = self.options.get(
            "alpha",
            get_default_partitioning_alpha(num_objectives=num_objectives),
        )
        # this selects the objectives (a subset of the outcomes) and set each
        # objective threhsold to have the proper optimization direction
        ref_point = objective(objective_thresholds).tolist()

        # initialize the sampler
        seed = int(torch.randint(1, 10000, (1,)).item())
        if qmc:
            sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
        else:
            sampler = IIDNormalSampler(
                num_samples=mc_samples, seed=seed
            )  # pragma: nocover
        if not ref_point:
            raise ValueError(
                "`ref_point` must be specified in kwargs for qEHVI"
            )  # pragma: nocover
        # get feasible points
        if cons_tfs is not None:
            # pyre-ignore [16]: `Tensor` has no attribute `all`.
            feas = torch.stack([c(Y) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
            Y = Y[feas]
        obj = objective(Y)
        partitioning = NondominatedPartitioning(
            ref_point=torch.as_tensor(ref_point, dtype=Y.dtype, device=Y.device),
            Y=obj,
            alpha=alpha,
        )
        self.acqf = self._botorch_acqf_class(  # pyre-ignore[28]: Some kwargs are
            # not expected in base `AcquisitionFunction` but are expected in
            # its subclasses.
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=cons_tfs,
            X_pending=X_pending,
        )
