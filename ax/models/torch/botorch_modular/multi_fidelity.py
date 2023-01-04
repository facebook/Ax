#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional

from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
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
    def compute_model_dependencies(
        self,
        surrogates: Mapping[str, Surrogate],
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if torch_opt_config.risk_measure is not None:  # pragma: no cover
            raise UnsupportedError(
                f"{self.__class__.__name__} does not support risk measures."
            )
        target_fidelities = search_space_digest.target_fidelities
        if not target_fidelities:
            raise ValueError(  # pragma: no cover
                "Target fidelities are required for {self.__class__.__name__}."
            )

        dependencies = super().compute_model_dependencies(
            surrogates=surrogates,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
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

        # pyre-fixme[53]: Captured variable `target_fidelities` is not annotated.
        def project(X: Tensor) -> Tensor:
            return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

        # pyre-fixme[53]: Captured variable `target_fidelities` is not annotated.
        def expand(X: Tensor) -> Tensor:
            return expand_trace_observations(
                X=X,
                fidelity_dims=sorted(target_fidelities),
                # pyre-fixme[16]: `Optional` has no attribute `get`.
                num_trace_obs=options.get(Keys.NUM_TRACE_OBSERVATIONS, 0),
            )

        dependencies.update(
            # pyre-fixme[6]: For 1st param expected `SupportsKeysAndGetItem[str,
            #  typing.Any]` but got `Dict[Keys, typing.Callable[[Named(X, Tensor)],
            #  typing.Any]]`.
            {
                Keys.COST_AWARE_UTILITY: cost_aware_utility,
                Keys.PROJECT: project,
                Keys.EXPAND: expand,
            }
        )
        return dependencies
