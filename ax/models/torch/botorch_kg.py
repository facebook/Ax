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
    subset_model,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import (
    AcquisitionObjective,
    ConstrainedMCObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.acquisition.utils import (
    expand_trace_observations,
    get_infeasible_cost,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.objective import get_objective_weights_transform
from torch import Tensor


class KnowledgeGradient(BotorchModel):
    r""" The Knowledge Gradient with one shot optimization

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
        self, cost_intercept: float = 1.0, linear_truncated: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(
            best_point_recommender=recommend_best_out_of_sample_point,
            linear_truncated=linear_truncated,
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
        """
        Generate new candidates.

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

        model = self.model

        # subset model only to the outcomes we need for the optimization
        if options.get("subset_model", True):
            model, objective_weights, outcome_constraints = subset_model(
                model=model,  # pyre-ignore [6]
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )

        objective = _get_objective(
            model=model,  # pyre-ignore [6]
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )

        # get the acquisition function
        n_fantasies = acf_options.get("num_fantasies", 64)
        qmc = acf_options.get("qmc", True)
        seed_inner = acf_options.get("seed_inner", None)
        num_restarts = optimizer_options.get("num_restarts", 40)
        raw_samples = optimizer_options.get("raw_samples", 1024)

        inequality_constraints = _to_inequality_constraints(linear_constraints)
        # TODO: update optimizers to handle inequality_constraints
        if inequality_constraints is not None:
            raise UnsupportedError(
                "Inequality constraints are not yet supported for KnowledgeGradient!"
            )

        # get current value
        best_point_acqf, non_fixed_idcs = self._get_best_point_acqf(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,  # pyre-ignore: [6]
            seed_inner=seed_inner,
            fixed_features=fixed_features,
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
        recommended_point = recommended_point.detach().unsqueeze(0)  # pyre-ignore: [16]
        # Extract acquisition value (TODO: Make this less painful and repetitive)
        if non_fixed_idcs is not None:
            recommended_point = recommended_point[..., non_fixed_idcs]
        current_value = best_point_acqf(recommended_point).max()

        acq_function = _instantiate_KG(
            model=model,  # pyre-ignore [6]
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
        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

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
        model = self.model

        # subset model only to the outcomes we need for the optimization
        if kwargs.get("subset_model", True):
            model, objective_weights, outcome_constraints = subset_model(
                model=model,  # pyre-ignore [6]
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )

        fixed_features = fixed_features or {}
        target_fidelities = target_fidelities or {}
        objective = _get_objective(
            model=model,  # pyre-ignore [6]
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )
        if isinstance(objective, ScalarizedObjective):
            acq_function = PosteriorMean(
                model=model, objective=objective  # pyre-ignore: [6]
            )
        elif isinstance(objective, MCAcquisitionObjective):
            if qmc:
                sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed_inner)
            else:
                sampler = IIDNormalSampler(num_samples=mc_samples, seed=seed_inner)
            acq_function = qSimpleRegret(
                model=model, sampler=sampler, objective=objective  # pyre-ignore [6]
            )
        else:
            raise UnsupportedError(
                f"Unknown objective type: {objective.__class__}"  # pragma: nocover
            )

        if self.fidelity_features:
            # we need to optimize at the target fidelities
            if any(f in self.fidelity_features for f in fixed_features):
                raise RuntimeError("Fixed features cannot also be fidelity features")
            elif not set(self.fidelity_features) == set(target_fidelities):
                raise RuntimeError(
                    "Must provide a target fidelity for every fidelity feature"
                )
            # make sure to not modify fixed_features in-place
            fixed_features = {**fixed_features, **target_fidelities}
        elif target_fidelities:
            raise RuntimeError(
                "Must specify fidelity_features in fit() when using target fidelities"
            )

        if fixed_features:
            acq_function = FixedFeatureAcquisitionFunction(
                acq_function=acq_function,
                d=X_observed.size(-1),
                columns=list(fixed_features.keys()),
                values=list(fixed_features.values()),
            )
            non_fixed_idcs = [
                i for i in range(self.Xs[0].size(-1)) if i not in fixed_features
            ]
        else:
            non_fixed_idcs = None

        return acq_function, non_fixed_idcs


def _get_objective(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
) -> AcquisitionObjective:
    if outcome_constraints is None:
        objective = ScalarizedObjective(weights=objective_weights)
    else:
        X_observed = torch.as_tensor(X_observed)
        obj_tf = get_objective_weights_transform(objective_weights)
        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=obj_tf)
        objective = ConstrainedMCObjective(
            objective=obj_tf, constraints=con_tfs or [], infeasible_cost=inf_cost
        )
    return objective


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
