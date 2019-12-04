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
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from torch import Tensor


class MaxValueEntropySearch(BotorchModel):
    r"""Max-value entropy search.

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
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[Tensor, Tensor, TGenMetadata]:
        if linear_constraints is not None or outcome_constraints is not None:
            raise UnsupportedError(
                "Constraints are not yet supported by max-value entropy search!"
            )

        if len(objective_weights) > 1:
            raise UnsupportedError(
                "Models with multiple outcomes are not yet supported by MES!"
            )

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

        # get the acquisition function
        num_fantasies = acf_options.get("num_fantasies", 16)
        num_mv_samples = acf_options.get("num_mv_samples", 10)
        num_y_samples = acf_options.get("num_y_samples", 128)
        candidate_size = acf_options.get("candidate_size", 1000)
        num_restarts = optimizer_options.get("num_restarts", 40)
        raw_samples = optimizer_options.get("raw_samples", 1024)

        # generate the discrete points in the design space to sample max values
        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)

        candidate_set = torch.rand(candidate_size, bounds_.size(1))
        candidate_set = bounds_[0] + (bounds_[1] - bounds_[0]) * candidate_set

        acq_function = _instantiate_MES(
            model=self.model,  # pyre-ignore: [6]
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_trace_observations=options.get("num_trace_observations", 0),
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            X_pending=X_pending,
            maximize=True if objective_weights[0] == 1 else False,
            target_fidelities=target_fidelities,
            fidelity_weights=options.get("fidelity_weights"),
            cost_intercept=self.cost_intercept,
        )

        # optimize and get new points
        botorch_rounding_func = get_rounding_func(rounding_func)
        candidates, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds_,
            q=n,
            inequality_constraints=None,
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
            sequential=True,
        )
        new_x = candidates.detach().cpu()
        return new_x, torch.ones(n, dtype=self.dtype), {}

    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Optional[Tensor]:
        """
        Identify the current best point, satisfying the constraints in the same
        format as to gen.

        Return None if no such point can be identified.

        Args:
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
                should be fixed to a particular value in the best point.
            model_gen_options: A config dictionary that can contain
                model-specific options.
            target_fidelities: A map {feature_index: value} of fidelity feature
                column indices to their respective target fidelities. Used for
                multi-fidelity optimization.

        Returns:
            d-tensor of the best point.
        """
        if linear_constraints is not None or outcome_constraints is not None:
            raise UnsupportedError(
                "Constraints are not yet supported by max-value entropy search!"
            )

        options = model_gen_options or {}
        fixed_features = fixed_features or {}
        optimizer_options = options.get("optimizer_kwargs", {})

        X_observed = get_observed(
            Xs=self.Xs,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )

        acq_function, non_fixed_idcs = self._get_best_point_acqf(
            X_observed=X_observed,  # pyre-ignore: [6]
            objective_weights=objective_weights,
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
        )

        return_best_only = optimizer_options.get("return_best_only", True)
        bounds_ = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        bounds_ = bounds_.transpose(0, 1)
        if non_fixed_idcs is not None:
            bounds_ = bounds_[..., non_fixed_idcs]

        candidates, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds_,
            q=1,
            num_restarts=optimizer_options.get("num_restarts", 60),
            raw_samples=optimizer_options.get("raw_samples", 1024),
            inequality_constraints=None,
            fixed_features=None,  # handled inside the acquisition function
            options={
                "batch_limit": optimizer_options.get("batch_limit", 8),
                "maxiter": optimizer_options.get("maxiter", 200),
                "nonnegative": optimizer_options.get("nonnegative", False),
                "method": "L-BFGS-B",
            },
            return_best_only=return_best_only,
        )
        rec_point = candidates.detach().cpu()
        if isinstance(acq_function, FixedFeatureAcquisitionFunction):
            rec_point = acq_function._construct_X_full(rec_point)
        if return_best_only:
            rec_point = rec_point.view(-1)
        return rec_point

    def _get_best_point_acqf(
        self,
        X_observed: Tensor,
        objective_weights: Tensor,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[AcquisitionFunction, Optional[List[int]]]:
        fixed_features = fixed_features or {}
        target_fidelities = target_fidelities or {}
        objective = ScalarizedObjective(weights=objective_weights)
        acq_function = PosteriorMean(
            model=self.model, objective=objective  # pyre-ignore: [6]
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


def _instantiate_MES(
    model: Model,
    candidate_set: Tensor,
    num_fantasies: int = 16,
    num_mv_samples: int = 10,
    num_y_samples: int = 128,
    use_gumbel: bool = True,
    X_pending: Optional[Tensor] = None,
    maximize: bool = True,
    num_trace_observations: int = 0,
    target_fidelities: Optional[Dict[int, float]] = None,
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
) -> qMaxValueEntropy:
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

        return qMultiFidelityMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            X_pending=X_pending,
            maximize=maximize,
            cost_aware_utility=cost_aware_utility,
            project=project,
            expand=expand,
        )

    return qMaxValueEntropy(
        model=model,
        candidate_set=candidate_set,
        num_fantasies=num_fantasies,
        num_mv_samples=num_mv_samples,
        num_y_samples=num_y_samples,
        X_pending=X_pending,
        maximize=maximize,
    )
