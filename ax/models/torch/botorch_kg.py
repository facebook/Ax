#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TConfig, TGenMetadata
from ax.models.model_utils import get_observed
from ax.models.torch.botorch import BotorchModel, get_rounding_func
from ax.models.torch.utils import _get_X_pending_and_observed
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
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
        super().__init__(linear_truncated=linear_truncated, **kwargs)
        self.cost_intercept = cost_intercept

    @copy_doc(TorchModel.gen)
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
    ) -> Tuple[Tensor, Tensor, TGenMetadata]:
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
        objective = _get_objective(
            model=self.model,  # pyre-ignore: [6]
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
        best_point_acqf = _set_best_point_acq(
            model=self.model,  # pyre-ignore: [6]
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            seed_inner=seed_inner,
            fidelity_features=self.fidelity_features,
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
        )
        recommended_point = recommended_point.detach()  # pyre-ignore: [16]
        idxr = [
            i
            for i in range(X_observed.size(-1))  # pyre-ignore: [16]
            if i not in self.fidelity_features
        ]
        acq_value = best_point_acqf(recommended_point.unsqueeze(0)[..., idxr])
        current_value = acq_value.max()

        acq_function = _instantiate_KG(
            model=self.model,  # pyre-ignore: [6]
            objective=objective,
            qmc=qmc,
            n_fantasies=n_fantasies,
            num_trace_observations=options.get("num_trace_observations", 0),
            mc_samples=acf_options.get("mc_samples", 256),
            seed_inner=seed_inner,
            seed_outer=acf_options.get("seed_outer", None),
            X_pending=X_pending,
            fidelity_features=self.fidelity_features,
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
        return new_x, torch.ones(n, dtype=self.dtype), {}

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[Tensor]:
        options = model_gen_options or {}
        acf_options = options.get("acquisition_function_kwargs", {})
        optimizer_options = options.get("optimizer_kwargs", {})

        inequality_constraints = _to_inequality_constraints(linear_constraints)
        # TODO: update optimizers to handle inequality_constraints
        if inequality_constraints is not None:
            raise UnsupportedError("Inequality constraints are not supported!")

        # The only difference between constrained and unconstrained version is the
        # `acq_function` in optimize_acqf.
        X_observed = get_observed(
            Xs=self.Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )

        acq_function = _set_best_point_acq(
            model=self.model,  # pyre-ignore: [6]
            mc_samples=acf_options.get("mc_samples", 512),
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            seed_inner=acf_options.get("seed_inner", None),
            fidelity_features=self.fidelity_features,
            qmc=acf_options.get("qmc", True),
        )
        return_best_only = optimizer_options.get("return_best_only", True)
        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        is_fixed = isinstance(acq_function, FixedFeatureAcquisitionFunction)
        dimension = X_observed.shape[-1] - is_fixed * len(self.fidelity_features)
        bounds_ = bounds_.transpose(0, 1)[..., :dimension]
        candidates, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds_,
            q=1,
            num_restarts=optimizer_options.get("num_restarts", 60),
            raw_samples=optimizer_options.get("raw_samples", 1024),
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            options={
                "batch_limit": optimizer_options.get("batch_limit", 8),
                "maxiter": optimizer_options.get("maxiter", 200),
                "nonnegative": optimizer_options.get("nonnegative", False),
                "method": "L-BFGS-B",
            },
            return_best_only=return_best_only,
        )
        recommended_point = candidates.detach().cpu()
        if is_fixed:
            recommended_point = acq_function._construct_X_full(  # pyre-ignore: [16]
                recommended_point
            )
        if return_best_only:
            recommended_point = recommended_point.view(-1)
        return recommended_point


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
    fidelity_features: Optional[List[int]] = None,
    cost_intercept: float = 1.0,
) -> qKnowledgeGradient:
    sampler_cls = SobolQMCNormalSampler if qmc else IIDNormalSampler
    fantasy_sampler = sampler_cls(num_samples=n_fantasies, seed=seed_outer)
    if isinstance(objective, MCAcquisitionObjective):
        inner_sampler = sampler_cls(num_samples=mc_samples, seed=seed_inner)
    else:
        inner_sampler = None
    if fidelity_features:
        cost_model = AffineFidelityCostModel(
            fidelity_weights={f: 1.0 for f in fidelity_features},
            fixed_cost=cost_intercept,
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        target_fidelities = {i: 1.0 for i in fidelity_features}

        def project(X: Tensor) -> Tensor:
            return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

        def expand(X: Tensor) -> Tensor:
            return expand_trace_observations(
                X=X,
                fidelity_dims=sorted(target_fidelities),
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


def _set_best_point_acq(
    model: Model,
    X_observed: Tensor,
    objective_weights: Tensor,
    mc_samples: int = 512,
    fidelity_features: Optional[List[int]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    seed_inner: Optional[int] = None,
    qmc: bool = True,
) -> AcquisitionFunction:
    objective = _get_objective(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
    )
    if isinstance(objective, ScalarizedObjective):
        acq_function = PosteriorMean(model=model, objective=objective)
    elif isinstance(objective, MCAcquisitionObjective):
        if qmc:
            sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed_inner)
        else:
            sampler = IIDNormalSampler(num_samples=mc_samples, seed=seed_inner)
        acq_function = qSimpleRegret(model=model, sampler=sampler, objective=objective)
    else:
        raise UnsupportedError(
            f"Unknown objective type: {objective.__class__}"  # pragma: nocover
        )
    if fidelity_features:
        acq_function = FixedFeatureAcquisitionFunction(
            acq_function=acq_function,
            d=X_observed.size(-1),
            columns=fidelity_features,
            values=[1.0],
        )
    return acq_function


def _to_inequality_constraints(
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None
) -> Optional[List[Tuple[Tensor, Tensor, float]]]:
    if linear_constraints is not None:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indicies = A[i, :].nonzero().squeeze()
            coefficients = -A[i, indicies]
            rhs = -b[i, 0]
            inequality_constraints.append((indicies, coefficients, rhs))
    else:
        inequality_constraints = None
    return inequality_constraints
