# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.adapter.registry import Generators
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
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


model_names_abbrevations: dict[str, str] = {
    SaasFullyBayesianSingleTaskGP.__name__: "SAAS",
}
acqf_name_abbreviations: dict[str, str] = {
    qLogNoisyExpectedImprovement.__name__: "qLogNEI",
    qNoisyExpectedHypervolumeImprovement.__name__: "qNEHVI",
    qLogNoisyExpectedHypervolumeImprovement.__name__: "qLogNEHVI",
    LogExpectedImprovement.__name__: "LogEI",
}


def get_sobol_mbm_generation_strategy(
    model_cls: type[Model],
    acquisition_cls: type[AcquisitionFunction] | None = None,
    name: str | None = None,
    num_sobol_trials: int = 5,
    model_gen_kwargs: dict[str, Any] | None = None,
    batch_size: int = 1,
) -> GenerationStrategy:
    """Get a `BenchmarkMethod` that uses Sobol followed by MBM.

    Args:
        model_cls: BoTorch model class, e.g. SingleTaskGP
        acquisition_cls: Acquisition function class, e.g.
            `qLogNoisyExpectedImprovement`.
        name: Name that will be attached to the `GenerationStrategy`.
        num_sobol_trials: Number of Sobol trials; can refer to the number of
            `BatchTrial`s.
        model_gen_kwargs: Passed to the BoTorch `GenerationStep` and ultimately
            to the BoTorch `Model`.

    Example:
        >>> # A simple example
        >>> from ax.benchmark.methods.sobol_botorch_modular import (
        ...     get_sobol_mbm_generation_strategy
        ... )
        >>> gs = get_sobol_mbm_generation_strategy(
        ...     model_cls=SingleTaskGP,
        ...     acquisition_cls=qLogNoisyExpectedImprovement,
        ... )
    """
    model_kwargs: dict[str, Any] = {
        "surrogate_spec": SurrogateSpec(
            model_configs=[ModelConfig(botorch_model_class=model_cls)]
        ),
    }
    if acquisition_cls is not None:
        model_kwargs["botorch_acqf_class"] = acquisition_cls
        acqf_name = acqf_name_abbreviations.get(
            acquisition_cls.__name__, acquisition_cls.__name__
        )
    else:
        acqf_name = ""

    model_name = model_names_abbrevations.get(model_cls.__name__, model_cls.__name__)
    # Historically all benchmarks were sequential, so sequential benchmarks
    # don't get anything added to their name, for continuity
    batch_suffix = f"_q{batch_size}" if batch_size > 1 else ""

    name = (
        name
        or f"MBM::{model_name}{('_' + acqf_name) if acqf_name else ''}{batch_suffix}"
    )

    generation_strategy = GenerationStrategy(
        name=name,
        steps=[
            GenerationStep(
                generator=Generators.SOBOL,
                num_trials=num_sobol_trials,
                min_trials_observed=num_sobol_trials,
            ),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs=model_kwargs,
                model_gen_kwargs=model_gen_kwargs or {},
            ),
        ],
    )
    return generation_strategy


def get_sobol_botorch_modular_acquisition(
    model_cls: type[Model],
    acquisition_cls: type[AcquisitionFunction] | None = None,
    name: str | None = None,
    num_sobol_trials: int = 5,
    model_gen_kwargs: dict[str, Any] | None = None,
    batch_size: int = 1,
) -> BenchmarkMethod:
    """Get a `BenchmarkMethod` that uses Sobol followed by MBM.

    Args:
        model_cls: BoTorch model class, e.g. SingleTaskGP
        acquisition_cls: Acquisition function class, e.g.
            `qLogNoisyExpectedImprovement`.
        name: Name that will be attached to the `GenerationStrategy`.
        num_sobol_trials: Number of Sobol trials; if the orchestrator_options
            specify to use `BatchTrial`s, then this refers to the number of
            `BatchTrial`s.
        model_gen_kwargs: Passed to the BoTorch `GenerationStep` and ultimately
            to the BoTorch `Model`.
        batch_size: Passed to the created ``BenchmarkMethod``.

    Example:
        >>> # A simple example
        >>> from ax.benchmark.methods.sobol_botorch_modular import (
        ...     get_sobol_botorch_modular_acquisition
        ... )
        >>>
        >>> method = get_sobol_botorch_modular_acquisition(
        ...     model_cls=SingleTaskGP,
        ...     acquisition_cls=qLogNoisyExpectedImprovement,
        ... )
        >>> # Pass sequential=False to BoTorch's optimize_acqf
        >>> batch_method = get_sobol_botorch_modular_acquisition(
        ...     model_cls=SingleTaskGP,
        ...     acquisition_cls=qLogNoisyExpectedImprovement,
        ...     batch_size=5,
        ...     model_gen_kwargs={
        ...         "model_gen_options": {
        ...             "optimizer_kwargs": {"sequential": False}
        ...         }
        ...     },
        ...     num_sobol_trials=1,
        ... )
    """
    generation_strategy = get_sobol_mbm_generation_strategy(
        model_cls=model_cls,
        acquisition_cls=acquisition_cls,
        name=name,
        num_sobol_trials=num_sobol_trials,
        model_gen_kwargs=model_gen_kwargs,
        batch_size=batch_size,
    )

    return BenchmarkMethod(
        generation_strategy=generation_strategy,
        batch_size=batch_size,
    )
