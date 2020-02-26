#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Cont_X_trans, Models, Y_trans
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.utils.common.logger import get_logger
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model import Model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


logger = get_logger(__name__)


ACQF_MODEL_MAP = {
    "NEI": Models.BOTORCH,
    "KG": Models.GPKG,
    "MES": Models.GPMES,
    "Sobol": Models.SOBOL,
    "RND": Models.UNIFORM,
}

# ------------------------- Standard model constructors ------------------------


def fixed_noise_gp_model_constructor(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
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


def singletask_gp_model_constructor(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    **kwargs: Any,
) -> Model:
    gp = SingleTaskGP(train_X=Xs[0], train_Y=Ys[0], **kwargs)
    gp.to(Xs[0])
    if state_dict is not None:
        gp.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        fit_gpytorch_model(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


# ----------------- Generation strategy constructor ----------------------------


def make_basic_generation_strategy(
    name: str,
    acquisition: str,
    num_initial_trials: int = 14,
    surrogate_model_constructor: Callable = singletask_gp_model_constructor,
) -> GenerationStrategy:

    if acquisition not in ACQF_MODEL_MAP:
        acquisition = "Sobol"
        logger.warning(
            f"{acquisition} is not a supported "
            "acquisition function. Defaulting to Sobol."
        )

    return GenerationStrategy(
        name=name,
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=num_initial_trials,
                min_trials_observed=num_initial_trials,
            ),
            GenerationStep(
                model=ACQF_MODEL_MAP[acquisition],
                num_trials=-1,
                model_kwargs={
                    "model_constructor": surrogate_model_constructor,
                    "transforms": Cont_X_trans + Y_trans,
                },
            ),
        ],
    )


# ----------------- Standard methods (as generation strategies) ----------------

# examples
winsorized_fixed_noise_NEI = GenerationStrategy(
    name="Sobol+fixed_noise_NEI",
    steps=[
        GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=3),
        GenerationStep(
            model=Models.BOTORCH,  # Note: can use FBModels, like FBModels.GPKG
            num_trials=-1,
            model_kwargs={
                "model_constructor": fixed_noise_gp_model_constructor,
                "transforms": [Winsorize] + Cont_X_trans + Y_trans,  # pyre-ignore[6]
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

singletask_RND = make_basic_generation_strategy(
    name="RND + SingleTaskGP", acquisition="RND", num_initial_trials=14
)

singletask_NEI = make_basic_generation_strategy(
    name="NEI + SingleTaskGP", acquisition="NEI", num_initial_trials=14
)

singletask_KG = make_basic_generation_strategy(
    name="KG + SingleTaskGP", acquisition="KG", num_initial_trials=14
)

singletask_MES = make_basic_generation_strategy(
    name="MES + SingleTaskGP", acquisition="MES", num_initial_trials=14
)

fixednoise_NEI = make_basic_generation_strategy(
    name="NEI + SingleTaskGP",
    acquisition="NEI",
    num_initial_trials=14,
    surrogate_model_constructor=fixed_noise_gp_model_constructor,
)
