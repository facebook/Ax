# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Optional, Type, Union

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_benchmark_scheduler_options,
)
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
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
    num_sobol_trials: int = 5,
    model_gen_kwargs: Optional[Dict[str, Any]] = None,
) -> BenchmarkMethod:
    """Get a `BenchmarkMethod` that uses Sobol followed by MBM.

    Args:
        model_cls: BoTorch model class, e.g. SingleTaskGP
        acquisition_cls: Acquisition function class, e.g.
            `qLogNoisyExpectedImprovement`.
        distribute_replications: Whether to use multiple machines
        scheduler_options: Passed as-is to scheduler. Default:
            `get_benchmark_scheduler_options()`.
        name: Name that will be attached to the `GenerationStrategy`.
        num_sobol_trials: Number of Sobol trials; if the scheduler_options
            specify to use `BatchTrial`s, then this refers to the number of
            `BatchTrial`s.
        model_gen_kwargs: Passed to the BoTorch `GenerationStep` and ultimately
            to the BoTorch `Model`.

    Example:
        >>> # A simple example
        >>> from ax.benchmark.methods.sobol_botorch_modular import (
        ...     get_sobol_botorch_modular_acquisition
        ... )
        >>> from ax.benchmark.benchmark_method import get_benchmark_scheduler_options
        >>>
        >>> method = get_sobol_botorch_modular_acquisition(
        ...     model_cls=SingleTaskGP,
        ...     acquisition_cls=qLogNoisyExpectedImprovement,
        ...     distribute_replications=False,
        ... )
        >>> # Pass sequential=False to BoTorch's optimize_acqf
        >>> batch_method = get_sobol_botorch_modular_acquisition(
        ...     model_cls=SingleTaskGP,
        ...     acquisition_cls=qLogNoisyExpectedImprovement,
        ...     distribute_replications=False,
        ...     scheduler_options=get_benchmark_scheduler_options(
        ...         batch_size=5,
        ...     ),
        ...     model_gen_kwargs=model_gen_kwargs={
        ...             "model_gen_options": {
        ...                 "optimizer_kwargs": {"sequential": False}
        ...             }
        ...         }
        ...     num_sobol_trials=1,
        ... )
    """
    model_kwargs: Dict[
        str, Union[Type[AcquisitionFunction], Dict[str, SurrogateSpec], bool]
    ] = {
        "botorch_acqf_class": acquisition_cls,
        "surrogate_specs": {"BoTorch": SurrogateSpec(botorch_model_class=model_cls)},
    }

    model_name = model_names_abbrevations.get(model_cls.__name__, model_cls.__name__)
    acqf_name = acqf_name_abbreviations.get(
        acquisition_cls.__name__, acquisition_cls.__name__
    )
    # Historically all benchmarks were sequential, so sequential benchmarks
    # don't get anything added to their name, for continuity
    batch_suffix = ""
    if (
        scheduler_options is not None
        and (batch_size := scheduler_options.batch_size) is not None
    ):
        if batch_size > 1:
            batch_suffix = f"_q{batch_size}"

    name = name or f"MBM::{model_name}_{acqf_name}{batch_suffix}"

    generation_strategy = GenerationStrategy(
        name=name,
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=num_sobol_trials,
                min_trials_observed=num_sobol_trials,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs=model_kwargs,
                model_gen_kwargs=model_gen_kwargs,
            ),
        ],
    )

    return BenchmarkMethod(
        name=generation_strategy.name,
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options or get_benchmark_scheduler_options(),
        distribute_replications=distribute_replications,
    )
