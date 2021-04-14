#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition, Optimizer
from ax.models.torch.botorch_modular.multi_fidelity import MultiFidelityAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from torch import Tensor


class OneShotAcquisition(Acquisition):
    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        optimizer_class: Optional[Optimizer] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        optimizer_options = optimizer_options or {}
        bounds = torch.tensor(
            search_space_digest.bounds, dtype=self.dtype, device=self.device
        ).transpose(0, 1)
        init_conditions = gen_one_shot_kg_initial_conditions(
            acq_function=checked_cast(qKnowledgeGradient, self.acqf),
            bounds=bounds,
            q=n,
            num_restarts=optimizer_options.get(Keys.NUM_RESTARTS),
            raw_samples=optimizer_options.get(Keys.RAW_SAMPLES),
            options={
                Keys.FRAC_RANDOM: optimizer_options.get(Keys.FRAC_RANDOM, 0.1),
                Keys.NUM_INNER_RESTARTS: optimizer_options.get(Keys.NUM_RESTARTS),
                Keys.RAW_INNER_SAMPLES: optimizer_options.get(Keys.RAW_SAMPLES),
            },
        )
        optimizer_options[Keys.BATCH_INIT_CONDITIONS] = init_conditions
        return super().optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )


class KnowledgeGradient(OneShotAcquisition):
    default_botorch_acqf_class = qKnowledgeGradient


class MultiFidelityKnowledgeGradient(MultiFidelityAcquisition, KnowledgeGradient):
    default_botorch_acqf_class = qMultiFidelityKnowledgeGradient

    def compute_model_dependencies(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        target_fidelities: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Compute generic multi-fidelity dependencies first
        dependencies = super().compute_model_dependencies(
            surrogate=surrogate,
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            pending_observations=pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=options,
        )

        _, best_point_acqf_value = surrogate.best_in_sample_point(
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )

        dependencies.update({Keys.CURRENT_VALUE: best_point_acqf_value})
        return dependencies
