#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP


DEFAULT_ACQUISITION_OPTIONS = {
    "num_fantasies": 16,
    "num_mv_samples": 10,
    "num_y_samples": 128,
    "candidate_size": 1000,
    "best_f": 0.0,
}
DEFAULT_OPTIMIZER_OPTIONS = {"num_restarts": 40, "raw_samples": 1024}


# BoTorch `Model` and Ax `Acquisition` combinations to be benchmarked.


# All of the single-fidelity models:

# Single Task GP + NEI
single_task_NEI_kwargs = {
    "surrogate": Surrogate(SingleTaskGP),
    "botorch_acqf_class": qNoisyExpectedImprovement,
    "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
}
# Fixed Noise GP + EI
fixed_noise_EI_kwargs = {
    "surrogate": Surrogate(FixedNoiseGP),
    "botorch_acqf_class": qExpectedImprovement,
    "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
}

# Gather all of the models:

single_fidelity_name_to_model_kwargs = {
    "Sobol+single_task_NEI": single_task_NEI_kwargs,
    "Sobol+fixed_noise_EI": fixed_noise_EI_kwargs,
    # "Sobol+single_task_KG": single_task_KG_kwargs,
    # "Sobol+fixed_noise_KG": fixed_noise_KG_kwargs,
    # "Sobol+single_task_MES": single_task_MES_kwargs,
    # "Sobol+fixed_noise_MES": fixed_noise_MES_kwargs,
}
multi_fidelity_name_to_model_kwargs = {
    # "Sobol+multi_fidelity_single_task_KG": mf_single_task_KG_kwargs,
    # "Sobol+multi_fidelity_fixed_noise_KG": mf_fixed_noise_KG_kwargs,
    # "Sobol+multi_fidelity_single_task_MES": mf_single_task_MES_kwargs,
    # "Sobol+multi_fidelity_fixed_noise_MES": mf_fixed_noise_MES_kwargs,
}


# NOTE: `name_to_model_kwargs` and `MODULAR_BOTORCH_METHOD_GROUPS` must
# have the same keys.
name_to_model_kwargs = {
    "single_fidelity_models": single_fidelity_name_to_model_kwargs,
    "multi_fidelity_models": multi_fidelity_name_to_model_kwargs,
}
MODULAR_BOTORCH_METHOD_GROUPS = {
    "single_fidelity_models": [],
    "multi_fidelity_models": [],
}
assert name_to_model_kwargs.keys() == MODULAR_BOTORCH_METHOD_GROUPS.keys()

# Populate the lists in `MODULAR_BOTORCH_METHODS_GROUPS`.
for group_name in MODULAR_BOTORCH_METHOD_GROUPS:
    for name, model_kwargs in name_to_model_kwargs[group_name].items():
        MODULAR_BOTORCH_METHOD_GROUPS[group_name].append(
            GenerationStrategy(
                name=name,
                steps=[
                    GenerationStep(
                        model=Models.SOBOL, num_trials=5, min_trials_observed=3
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        model_kwargs=model_kwargs,
                        model_gen_kwargs={
                            "model_gen_options": {
                                Keys.OPTIMIZER_KWARGS: DEFAULT_OPTIMIZER_OPTIONS
                            }
                        },
                    ),
                ],
            )
        )

# TODO: Add commented out methods when they are brought back to modular BotAx
# # Single Task GP + KG
# single_task_KG_kwargs = {
#     "surrogate": Surrogate(SingleTaskGP),
#     "acquisition_class": KnowledgeGradient,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Fixed Noise GP + KG
# fixed_noise_KG_kwargs = {
#     "surrogate": Surrogate(FixedNoiseGP),
#     "acquisition_class": KnowledgeGradient,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Single Task GP + MES
# single_task_MES_kwargs = {
#     "surrogate": Surrogate(SingleTaskGP),
#     "acquisition_class": MaxValueEntropySearch,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Fixed Noise GP + MES
# fixed_noise_MES_kwargs = {
#     "surrogate": Surrogate(FixedNoiseGP),
#     "acquisition_class": MaxValueEntropySearch,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }


# All of the multi-fidelity models:

# Single Task GP + KG
# mf_single_task_KG_kwargs = {
#     "surrogate": Surrogate(SingleTaskMultiFidelityGP),
#     "acquisition_class": MultiFidelityKnowledgeGradient,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Fixed Noise GP + KG
# mf_fixed_noise_KG_kwargs = {
#     "surrogate": Surrogate(FixedNoiseMultiFidelityGP),
#     "acquisition_class": MultiFidelityKnowledgeGradient,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Single Task GP + MES
# mf_single_task_MES_kwargs = {
#     "surrogate": Surrogate(SingleTaskMultiFidelityGP),
#     "acquisition_class": MultiFidelityMaxValueEntropySearch,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
# # Fixed Noise GP + MES
# mf_fixed_noise_MES_kwargs = {
#     "surrogate": Surrogate(FixedNoiseMultiFidelityGP),
#     "acquisition_class": MultiFidelityMaxValueEntropySearch,
#     "acquisition_options": DEFAULT_ACQUISITION_OPTIONS,
# }
