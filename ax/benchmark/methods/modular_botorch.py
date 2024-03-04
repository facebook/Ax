# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Optional, Type, Union

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_sequential_optimization_scheduler_options,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.model import SurrogateSpec

from ax.service.scheduler import SchedulerOptions
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model


model_names_abbrevations: Dict[str, str] = {
    SaasFullyBayesianSingleTaskGP.__name__: "SAAS",
}
acqf_name_abbreviations: Dict[str, str] = {
    qLogNoisyExpectedImprovement.__name__: "qLogNEI",
    qNoisyExpectedHypervolumeImprovement.__name__: "qNEHVI",
    qLogNoisyExpectedHypervolumeImprovement.__name__: "qLogNEHVI",
    LogExpectedImprovement.__name__: "LogEI",
}


def get_sobol_botorch_modular_acquisition(
    model_cls: Type[Model],
    acquisition_cls: Type[AcquisitionFunction],
    distribute_replications: bool,
    scheduler_options: Optional[SchedulerOptions] = None,
    name: Optional[str] = None,
) -> BenchmarkMethod:
    model_kwargs: Dict[
        str, Union[Type[AcquisitionFunction], Dict[str, SurrogateSpec]]
    ] = {
        "botorch_acqf_class": acquisition_cls,
        "surrogate_specs": {"BoTorch": SurrogateSpec(botorch_model_class=model_cls)},
    }

    model_name = model_names_abbrevations.get(model_cls.__name__, model_cls.__name__)
    acqf_name = acqf_name_abbreviations.get(
        acquisition_cls.__name__, acquisition_cls.__name__
    )
    name = f"MBM::{model_name}_{acqf_name}"

    generation_strategy = GenerationStrategy(
        name=name,
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs=model_kwargs,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options
        or get_sequential_optimization_scheduler_options(),
        distribute_replications=distribute_replications,
    )
