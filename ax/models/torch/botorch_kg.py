#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.models.torch.botorch import BotorchModel, get_rounding_func
from ax.models.torch.botorch_defaults import recommend_best_out_of_sample_point
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    get_botorch_objective,
    get_out_of_sample_best_point_acqf,
    subset_model,
)
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.objective import AcquisitionObjective, MCAcquisitionObjective
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from torch import Tensor


class KnowledgeGradient(BotorchModel):
    r"""The Knowledge Gradient with one shot optimization.

    Args:
        cost_intercept: The cost intercept for the affine cost of the form
            `cost_intercept + n`, where `n` is the number of generated points.
            Only used for multi-fidelity optimzation (i.e., if fidelity_features
            are present).
        linear_truncated: If `False`, use an alternate downsampling + exponential
            decay Kernel instead of the default `LinearTruncatedFidelityKernel`
            (only relevant for multi-fidelity optimization).
        kwargs: Model-specific kwargs.
    """

    def __init__(
        self,
        cost_intercept: float = 1.0,
        linear_truncated: bool = True,
        use_input_warping: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            best_point_recommender=recommend_best_out_of_sample_point,
            linear_truncated=linear_truncated,
            use_input_warping=use_input_warping,
            **kwargs,
        )
        self.cost_intercept = cost_intercept

    def gen(
        self,
        n: int,
        bounds: List,
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[Tensor, Tensor, TGenMetadata, Optional[List[TCandidateMetadata]]]:
        r"""Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature tensors X
                for m outcomes and k_i pending observations for outcome i.
            model_gen_options: A config dictionary that can contain
                model-specific options.
            rounding_func: A function that rounds an optimization result
                appropriately (i.e., according to `round-trip` transformations).
            target_fidelities: A map {feature_index: value} of fidelity feature
                column indices to their respective target fidelities. Used for
                multi-fidelity optimization.

        Returns:
            3-element tuple containing

            - (n x d) tensor of generated points.
            - n-tensor of weights for each point.
            - Dictionary of model-specific metadata for the given
                generation candidates.
        """
        options = model_gen_options or {}
        acf_options = options.get("acquisition_function_kwargs", {})
        optimizer_options = options.get("optimizer_kwargs", {})

        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            pending_observations=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )

        # subset model only to the outcomes we need for the optimization
        model = not_none(self.model)
        if options.get("subset_model", True):
            subset_model_results = subset_model(
                model=model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints

        objective = get_botorch_objective(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )

        inequality_constraints = _to_inequality_constraints(linear_constraints)
        # TODO: update optimizers to handle inequality_constraints
        if inequality_constraints is not None:
            raise UnsupportedError(
                "Inequality constraints are not yet supported for KnowledgeGradient!"
            )

        # extract a few options
        n_fantasies = acf_options.get("num_fantasies", 64)
        qmc = acf_options.get("qmc", True)
        seed_inner = acf_options.get("seed_inner", None)
        num_restarts = optimizer_options.get("num_restarts", 40)
        raw_samples = optimizer_options.get("raw_samples", 1024)

        # get current value
        current_value = self._get_current_value(
            model=model,
            bounds=bounds,
            X_observed=not_none(X_observed),
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            seed_inner=seed_inner,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            target_fidelities=target_fidelities,
            qmc=qmc,
        )

        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

        # get acquisition function
        acq_function = _instantiate_KG(
            model=model,
            objective=objective,
            qmc=qmc,
            n_fantasies=n_fantasies,
            num_trace_observations=options.get("num_trace_observations", 0),
            mc_samples=acf_options.get("mc_samples", 256),
            seed_inner=seed_inner,
            seed_outer=acf_options.get("seed_outer", None),
            X_pending=X_pending,
            target_fidelities=target_fidelities,
            fidelity_weights=options.get("fidelity_weights"),
            current_value=current_value,
            cost_intercept=self.cost_intercept,
        )

        # optimize and get new points
        new_x = _optimize_and_get_candidates(
            acq_function=acq_function,
            bounds_=bounds_,
            n=n,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            optimizer_options=optimizer_options,
            rounding_func=rounding_func,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
        )

        return new_x, torch.ones(n, dtype=self.dtype), {}, None

    def _get_best_point_acqf(
        self,
        X_observed: Tensor,
        objective_weights: Tensor,
        mc_samples: int = 512,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        seed_inner: Optional[int] = None,
        qmc: bool = True,
        **kwargs: Any,
    ) -> Tuple[AcquisitionFunction, Optional[List[int]]]:
        return get_out_of_sample_best_point_acqf(
            model=not_none(self.model),
            Xs=self.Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=not_none(X_observed),
            seed_inner=seed_inner,
            fixed_features=fixed_features,
            fidelity_features=self.fidelity_features,
            target_fidelities=target_fidelities,
            qmc=qmc,
        )

    def _get_current_value(
        self,
        model: Model,
        bounds: List,
        X_observed: Tensor,
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]],
        linear_constraints: Optional[Tuple[Tensor, Tensor]],
        seed_inner: Optional[int],
        fixed_features: Optional[Dict[int, float]],
        model_gen_options: Optional[TConfig],
        target_fidelities: Optional[Dict[int, float]],
        qmc: bool,
    ) -> Tensor:
        r"""Computes the value of the current best point. This is the current_value
        passed to KG.

        NOTE: The current value is computed as the current value of the 'best point
        acquisition function' (typically `PosteriorMean` or `qSimpleRegret`), not of
        the Knowledge Gradient acquisition function.
        """
        best_point_acqf, non_fixed_idcs = get_out_of_sample_best_point_acqf(
            model=model,
            Xs=self.Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            seed_inner=seed_inner,
            fixed_features=fixed_features,
            fidelity_features=self.fidelity_features,
            target_fidelities=target_fidelities,
            qmc=qmc,
        )

        # solution from previous iteration
        recommended_point = self.best_point(
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            target_fidelities=target_fidelities,
        )
        # pyre-fixme[16]: `Optional` has no attribute `detach`.
        recommended_point = recommended_point.detach().unsqueeze(0)
        # ensure correct device (`best_point` always returns a CPU tensor)
        recommended_point = recommended_point.to(device=self.device)
        # Extract acquisition value (TODO: Make this less painful and repetitive)
        if non_fixed_idcs is not None:
            recommended_point = recommended_point[..., non_fixed_idcs]
        current_value = best_point_acqf(recommended_point).max()
        return current_value


def _instantiate_KG(
    model: Model,
    objective: AcquisitionObjective,
    qmc: bool = True,
    n_fantasies: int = 64,
    mc_samples: int = 256,
    num_trace_observations: int = 0,
    seed_inner: Optional[int] = None,
    seed_outer: Optional[int] = None,
    X_pending: Optional[Tensor] = None,
    current_value: Optional[Tensor] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
) -> qKnowledgeGradient:
    r"""Instantiate either a `qKnowledgeGradient` or `qMultiFidelityKnowledgeGradient`
    acquisition function depending on whether `target_fidelities` is defined.
    """
    sampler_cls = SobolQMCNormalSampler if qmc else IIDNormalSampler
    fantasy_sampler = sampler_cls(num_samples=n_fantasies, seed=seed_outer)
    if isinstance(objective, MCAcquisitionObjective):
        inner_sampler = sampler_cls(num_samples=mc_samples, seed=seed_inner)
    else:
        inner_sampler = None
    if target_fidelities:
        if fidelity_weights is None:
            fidelity_weights = {f: 1.0 for f in target_fidelities}
        if not set(target_fidelities) == set(fidelity_weights):
            raise RuntimeError(
                "Must provide the same indices for target_fidelities "
                f"({set(target_fidelities)}) and fidelity_weights "
                f" ({set(fidelity_weights)})."
            )
        cost_model = AffineFidelityCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=cost_intercept
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        def project(X: Tensor) -> Tensor:
            return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

        def expand(X: Tensor) -> Tensor:
            return expand_trace_observations(
                X=X,
                fidelity_dims=sorted(target_fidelities),  # pyre-ignore: [6]
                num_trace_obs=num_trace_observations,
            )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=n_fantasies,
            sampler=fantasy_sampler,
            objective=objective,
            inner_sampler=inner_sampler,
            X_pending=X_pending,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=project,
            expand=expand,
        )

    return qKnowledgeGradient(
        model=model,
        num_fantasies=n_fantasies,
        sampler=fantasy_sampler,
        objective=objective,
        inner_sampler=inner_sampler,
        X_pending=X_pending,
        current_value=current_value,
    )


def _optimize_and_get_candidates(
    acq_function: qKnowledgeGradient,
    bounds_: Tensor,
    n: int,
    num_restarts: int,
    raw_samples: int,
    optimizer_options: Dict,
    rounding_func: Optional[Callable[[Tensor], Tensor]],
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]],
    fixed_features: Optional[Dict[int, float]],
) -> Tensor:
    r"""Generates initial conditions for optimization, optimize the acquisition
    function, and return the candidates.
    """
    batch_initial_conditions = gen_one_shot_kg_initial_conditions(
        acq_function=acq_function,
        bounds=bounds_,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            "frac_random": optimizer_options.get("frac_random", 0.1),
            "num_inner_restarts": num_restarts,
            "raw_inner_samples": raw_samples,
        },
    )

    botorch_rounding_func = get_rounding_func(rounding_func)

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds_,
        q=n,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        post_processing_func=botorch_rounding_func,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            "batch_limit": optimizer_options.get("batch_limit", 8),
            "maxiter": optimizer_options.get("maxiter", 200),
            "method": "L-BFGS-B",
            "nonnegative": optimizer_options.get("nonnegative", False),
        },
        batch_initial_conditions=batch_initial_conditions,
    )
    new_x = candidates.detach().cpu()
    return new_x
