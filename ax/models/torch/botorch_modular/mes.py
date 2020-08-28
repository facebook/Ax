#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition, Optimizer
from ax.models.torch.botorch_modular.multi_fidelity import MultiFidelityAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from torch import Tensor


class MaxValueEntropySearch(Acquisition):
    default_botorch_acqf_class = qMaxValueEntropy

    def optimize(
        self,
        bounds: Tensor,
        n: int,
        optimizer_class: Optional[Optimizer] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        optimizer_options = optimizer_options or {}
        optimizer_options[Keys.SEQUENTIAL] = True
        return super().optimize(
            bounds=bounds,
            n=n,
            inequality_constraints=None,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )

    @classmethod
    def compute_model_dependencies(
        cls,
        surrogate: Surrogate,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
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

        bounds_ = torch.tensor(
            bounds, dtype=surrogate.dtype, device=surrogate.device
        ).transpose(0, 1)

        candidate_size = options.get(Keys.CANDIDATE_SIZE, 1000)
        candidate_set = torch.rand(candidate_size, bounds_.size(1))
        candidate_set = bounds_[0] + (bounds_[1] - bounds_[0]) * candidate_set

        maximize = True if objective_weights[0] == 1 else False

        dependencies.update(
            {Keys.CANDIDATE_SET: candidate_set, Keys.MAXIMIZE: maximize}
        )
        return dependencies


class MultiFidelityMaxValueEntropySearch(MaxValueEntropySearch):
    default_botorch_acqf_class = qMultiFidelityMaxValueEntropy

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
        dependencies.update(
            MultiFidelityAcquisition.compute_model_dependencies(
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
        )
        return dependencies
