#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.models.cost import AffineFidelityCostModel
from torch import Tensor


class MultiFidelityAcquisition(Acquisition):

    # NOTE: Here, we do not consider using `IIDNormalSampler` and always
    # use the `SobolQMCNormalSampler`.
    @classmethod
    def compute_model_dependencies(
        cls,
        surrogate: Surrogate,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        target_fidelities: Dict[int, float],
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        dependencies = super().compute_model_dependencies(
            surrogate=surrogate,
            bounds=bounds,
            objective_weights=objective_weights,
            pending_observations=pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            options=options,
        )

        options = options or {}

        fidelity_weights = options.get(Keys.FIDELITY_WEIGHTS, None)
        if fidelity_weights is None:
            fidelity_weights = {f: 1.0 for f in target_fidelities}
        if not set(target_fidelities) == set(fidelity_weights):
            raise RuntimeError(
                "Must provide the same indices for target_fidelities "
                f"({set(target_fidelities)}) and fidelity_weights "
                f" ({set(fidelity_weights)})."
            )

        cost_intercept = options.get(Keys.COST_INTERCEPT, 1.0)

        cost_model = AffineFidelityCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=cost_intercept
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        def project(X: Tensor) -> Tensor:
            return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

        def expand(X: Tensor) -> Tensor:
            return expand_trace_observations(
                X=X,
                fidelity_dims=sorted(target_fidelities),
                # pyre-fixme[16]: `Optional` has no attribute `get`.
                num_trace_obs=options.get(Keys.NUM_TRACE_OBSERVATIONS, 0),
            )

        dependencies.update(
            {
                Keys.COST_AWARE_UTILITY: cost_aware_utility,
                Keys.PROJECT: project,
                Keys.EXPAND: expand,
            }
        )
        return dependencies
