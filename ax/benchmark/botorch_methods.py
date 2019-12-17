#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.transforms.winsorize import Winsorize
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model import Model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


# ------------------------- Standard model constructors ------------------------


def fixed_noise_gp_model_constructor(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],  # Maybe these should be optional where irrelevant?
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    **kwargs: Any,
) -> Model:
    gp = FixedNoiseGP(train_X=Xs[0], train_Y=Ys[0], train_Yvar=Yvars[0], **kwargs)
    gp.to(Xs[0])
    if state_dict is not None:
        gp.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        fit_gpytorch_model(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


# ----------------- Standard methods (as generation strategies) ----------------


winsorized_fixed_noise_GPEI = GenerationStrategy(
    name="Sobol+fixed_noise_GPEI",
    steps=[
        GenerationStep(model=Models.SOBOL, num_arms=5, min_arms_observed=3),
        GenerationStep(
            model=Models.BOTORCH,  # Note: can use FBModels, like FBModels.GPKG
            num_arms=-1,
            model_kwargs={
                "model_constructor": fixed_noise_gp_model_constructor,
                "transforms": [Winsorize] + Cont_X_trans + Y_trans,
                "transform_configs": {
                    "Winsorize": {
                        f"winsorization_{t}": v
                        for t, v in zip(("lower", "upper"), (0.2, None))
                    }
                },
            },
        ),
    ],
)
