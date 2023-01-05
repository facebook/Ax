# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_sequential_optimization_scheduler_options,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP


def get_sobol_botorch_modular_fixed_noise_gp_qnei() -> BenchmarkMethod:
    model_gen_kwargs = {
        "model_gen_options": {
            Keys.OPTIMIZER_KWARGS: {
                "num_restarts": 50,
                "raw_samples": 1024,
            },
            Keys.ACQF_KWARGS: {
                "prune_baseline": True,
                "qmc": True,
                "mc_samples": 512,
            },
        }
    }

    generation_strategy = GenerationStrategy(
        name="SOBOL+BOTORCH_MODULAR::FixedNoiseGP_qNoisyExpectedImprovement",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(FixedNoiseGP),
                    "botorch_acqf_class": qNoisyExpectedImprovement,
                },
                model_gen_kwargs=model_gen_kwargs,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )


def get_sobol_botorch_modular_fixed_noise_gp_qnehvi() -> BenchmarkMethod:
    model_gen_kwargs = {
        "model_gen_options": {
            Keys.OPTIMIZER_KWARGS: {
                "num_restarts": 50,
                "raw_samples": 1024,
            },
            Keys.ACQF_KWARGS: {
                "prune_baseline": True,
                "qmc": True,
                "mc_samples": 512,
            },
        }
    }

    generation_strategy = GenerationStrategy(
        name="SOBOL+BOTORCH_MODULAR::FixedNoiseGP_qNoisyExpectedHypervolumeImprovement",
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=5,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(FixedNoiseGP),
                    "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
                },
                model_gen_kwargs=model_gen_kwargs,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )


def get_sobol_botorch_modular_saas_fully_bayesian_single_task_gp_qnei() -> BenchmarkMethod:  # noqa
    generation_strategy = GenerationStrategy(
        name="SOBOL+BOTORCH_MODULAR::SaasFullyBayesianSingleTaskGP_qNoisyExpectedImprovement",  # noqa
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=SaasFullyBayesianSingleTaskGP
                    ),
                    "botorch_acqf_class": qNoisyExpectedImprovement,
                },
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )


def get_sobol_botorch_modular_saas_fully_bayesian_single_task_gp_qnehvi() -> BenchmarkMethod:  # noqa
    generation_strategy = GenerationStrategy(
        name="SOBOL+BOTORCH_MODULAR::SaasFullyBayesianSingleTaskGP_qNoisyExpectedHypervolumeImprovement",  # noqa
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=5,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=SaasFullyBayesianSingleTaskGP
                    ),
                    "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
                },
            ),
        ],
    )
    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )


def get_sobol_botorch_modular_default() -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name="SOBOL+BOTORCH_MODULAR::default",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )


def get_sobol_botorch_modular_acquisition(
    acquisition_cls: Type[AcquisitionFunction],
    acquisition_options: Optional[Dict[str, Any]] = None,
) -> BenchmarkMethod:
    generation_strategy = GenerationStrategy(
        name=f"SOBOL+BOTORCH_MODULAR::{acquisition_cls.__name__}",
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=5,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "botorch_acqf_class": acquisition_cls,
                    "acquisition_options": acquisition_options,
                },
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=get_sequential_optimization_scheduler_options(),
    )
