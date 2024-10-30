#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem, create_problem_from_botorch
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_runner import BenchmarkRunner
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.scheduler import SchedulerOptions
from ax.utils.common.constants import Keys
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Branin
from pyre_extensions import assert_is_instance
from torch.utils.data import Dataset


def get_single_objective_benchmark_problem(
    observe_noise_sd: bool = False,
    num_trials: int = 4,
    test_problem_kwargs: dict[str, Any] | None = None,
    report_inference_value_as_trace: bool = False,
    noise_std: float | list[float] = 0.0,
) -> BenchmarkProblem:
    return create_problem_from_botorch(
        test_problem_class=Branin,
        test_problem_kwargs=test_problem_kwargs or {},
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
        report_inference_value_as_trace=report_inference_value_as_trace,
        noise_std=noise_std,
    )


def get_multi_objective_benchmark_problem(
    observe_noise_sd: bool = False,
    num_trials: int = 4,
    test_problem_class: type[BraninCurrin] = BraninCurrin,
    report_inference_value_as_trace: bool = False,
) -> BenchmarkProblem:
    return create_problem_from_botorch(
        test_problem_class=test_problem_class,
        test_problem_kwargs={},
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
        report_inference_value_as_trace=report_inference_value_as_trace,
    )


def get_soo_surrogate_test_function(lazy: bool = True) -> SurrogateTestFunction:
    experiment = get_branin_experiment(with_completed_trial=True)
    surrogate = TorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        model=BoTorchModel(surrogate=Surrogate(botorch_model_class=SingleTaskGP)),
        data=experiment.lookup_data(),
        transforms=[],
    )
    if lazy:
        test_function = SurrogateTestFunction(
            outcome_names=["branin"],
            name="test",
            get_surrogate_and_datasets=lambda: (surrogate, []),
        )
    else:
        test_function = SurrogateTestFunction(
            outcome_names=["branin"],
            name="test",
            _surrogate=surrogate,
            _datasets=[],
        )
    return test_function


def get_soo_surrogate() -> BenchmarkProblem:
    experiment = get_branin_experiment(with_completed_trial=True)
    test_function = get_soo_surrogate_test_function()
    runner = BenchmarkRunner(test_problem=test_function, outcome_names=["branin"])

    observe_noise_sd = True
    objective = Objective(
        metric=BenchmarkMetric(
            name="branin", lower_is_better=True, observe_noise_sd=observe_noise_sd
        ),
    )
    optimization_config = OptimizationConfig(objective=objective)

    return BenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=optimization_config,
        num_trials=6,
        observe_noise_stds=observe_noise_sd,
        optimal_value=0.0,
        runner=runner,
    )


def get_moo_surrogate() -> BenchmarkProblem:
    experiment = get_branin_experiment_with_multi_objective(with_completed_trial=True)
    surrogate = TorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        model=BoTorchModel(surrogate=Surrogate(botorch_model_class=SingleTaskGP)),
        data=experiment.lookup_data(),
        transforms=[],
    )

    outcome_names = ["branin_a", "branin_b"]
    test_function = SurrogateTestFunction(
        name="test",
        outcome_names=outcome_names,
        get_surrogate_and_datasets=lambda: (surrogate, []),
    )
    runner = BenchmarkRunner(test_problem=test_function, outcome_names=outcome_names)
    observe_noise_sd = True
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(
            objectives=[
                Objective(
                    metric=BenchmarkMetric(
                        name="branin_a",
                        lower_is_better=True,
                        observe_noise_sd=observe_noise_sd,
                    ),
                ),
                Objective(
                    metric=BenchmarkMetric(
                        name="branin_b",
                        lower_is_better=True,
                        observe_noise_sd=observe_noise_sd,
                    ),
                ),
            ],
        )
    )
    return BenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=optimization_config,
        num_trials=10,
        observe_noise_stds=True,
        optimal_value=1.0,
        runner=runner,
    )


def get_sobol_gpei_benchmark_method() -> BenchmarkMethod:
    return BenchmarkMethod(
        name="MBO_SOBOL_GPEI",
        generation_strategy=GenerationStrategy(
            name="Modular::Sobol+GPEI",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=3,
                    model_kwargs={"fit_tracking_metrics": False},
                    min_trials_observed=3,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    model_kwargs={
                        "surrogate": Surrogate(SingleTaskGP),
                        # TODO: tests should better reflect defaults and not
                        # re-implement this logic.
                        "botorch_acqf_class": qNoisyExpectedImprovement,
                        "model_kwargs": {"fit_tracking_metrics": False},
                    },
                    model_gen_kwargs={
                        "model_gen_options": {
                            Keys.OPTIMIZER_KWARGS: {
                                "num_restarts": 50,
                                "raw_samples": 1024,
                            },
                            Keys.ACQF_KWARGS: {
                                "prune_baseline": True,
                            },
                        }
                    },
                ),
            ],
        ),
        scheduler_options=SchedulerOptions(
            total_trials=4, init_seconds_between_polls=0
        ),
    )


def get_benchmark_result() -> BenchmarkResult:
    problem = get_single_objective_benchmark_problem()

    return BenchmarkResult(
        name="test_benchmarking_result",
        seed=0,
        experiment=Experiment(
            name="test_benchmarking_experiment",
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            is_test=True,
        ),
        inference_trace=np.ones(4),
        oracle_trace=np.zeros(4),
        optimization_trace=np.array([3, 2, 1, 0.1]),
        score_trace=np.array([3, 2, 1, 0.1]),
        fit_time=0.1,
        gen_time=0.2,
    )


def get_aggregated_benchmark_result() -> AggregatedBenchmarkResult:
    result = get_benchmark_result()
    return AggregatedBenchmarkResult.from_benchmark_results([result, result])


@dataclass(kw_only=True)
class DummyTestFunction(BenchmarkTestFunction):
    num_outcomes: int = 1
    dim: int = 6

    # pyre-fixme[14]: Inconsistent override, as dict[str, float] is not a
    # `TParameterization`
    def evaluate_true(self, params: dict[str, float]) -> torch.Tensor:
        value = sum(elt**2 for elt in params.values())
        return value * torch.ones(self.num_outcomes, dtype=torch.double)


class TestDataset(Dataset):
    def __init__(
        self,
        root: str = "",
        train: bool = True,
        download: bool = True,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        transform: Any = None,
    ) -> None:
        torch.manual_seed(0)
        self.data: torch.Tensor = torch.randint(
            low=0, high=256, size=(32, 1, 28, 28), dtype=torch.float32
        )
        self.targets: torch.Tensor = torch.randint(
            low=0, high=10, size=(32,), dtype=torch.uint8
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        target = assert_is_instance(self.targets[idx].item(), int)
        return self.data[idx], target
