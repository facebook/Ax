#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_metric import (
    BenchmarkMapMetric,
    BenchmarkMapUnavailableWhileRunningMetric,
    BenchmarkMetric,
    BenchmarkTimeVaryingMetric,
)
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    create_problem_from_botorch,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_step_runtime_function import TBenchmarkStepRuntimeFunction
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.benchmark.benchmark_test_functions.synthetic import IdentityTestFunction
from ax.benchmark.problems.synthetic.hss.jenatton import get_jenatton_search_space
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types import TParameterization, TParamValue
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.modelbridge.external_generation_node import ExternalGenerationNode
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
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
    status_quo_params: TParameterization | None = None,
) -> BenchmarkProblem:
    return create_problem_from_botorch(
        test_problem_class=Branin,
        test_problem_kwargs=test_problem_kwargs or {},
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
        report_inference_value_as_trace=report_inference_value_as_trace,
        noise_std=noise_std,
        status_quo_params=status_quo_params,
        baseline_value=3,
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
        baseline_value=0.0,
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
            outcome_names=["branin"], name="test", get_surrogate=lambda: surrogate
        )
    else:
        test_function = SurrogateTestFunction(
            outcome_names=["branin"],
            name="test",
            _surrogate=surrogate,
        )
    return test_function


def get_soo_surrogate() -> BenchmarkProblem:
    experiment = get_branin_experiment(with_completed_trial=True)
    test_function = get_soo_surrogate_test_function()

    optimization_config = get_soo_opt_config(
        outcome_names=test_function.outcome_names,
        observe_noise_sd=True,
    )

    return BenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=optimization_config,
        num_trials=6,
        optimal_value=0.0,
        baseline_value=3.0,
        test_function=test_function,
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
        name="test", outcome_names=outcome_names, get_surrogate=lambda: surrogate
    )
    optimization_config = get_moo_opt_config(
        outcome_names=outcome_names,
        ref_point=[0.0, 0.0],
        observe_noise_sd=True,
    )

    return BenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=optimization_config,
        num_trials=10,
        optimal_value=1.0,
        baseline_value=0.0,
        test_function=test_function,
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
    outcome_names: list[str] = field(default_factory=list)
    num_outcomes: int = 1
    dim: int = 6

    def __post_init__(self) -> None:
        self.outcome_names = [f"objective_{i}" for i in range(self.num_outcomes)]

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


def get_jenatton_arm(i: int) -> Arm:
    """
    Args:
        i Non-negative int.
    """
    jenatton_x_params = {f"x{j}": j % (i + 1) for j in range(1, 8)}
    jenatton_r_params = {"r8": 0.0, "r9": 0.0}
    return Arm(parameters={**jenatton_x_params, **jenatton_r_params}, name=f"0_{i}")


def get_jenatton_experiment() -> Experiment:
    experiment = Experiment(
        search_space=get_jenatton_search_space(),
        name="test_jenatton",
        is_test=True,
    )
    return experiment


def get_jenatton_trials(n_trials: int) -> dict[int, Trial]:
    experiment = get_jenatton_experiment()
    for i in range(n_trials):
        trial = experiment.new_trial()
        trial.add_arm(get_jenatton_arm(i=i))
    # pyre-fixme: Incompatible return type [7]: Expected `Dict[int, Trial]` but
    # got `Dict[int, BaseTrial]`.
    return experiment.trials


def get_jenatton_batch_trial() -> BatchTrial:
    experiment = get_jenatton_experiment()
    trial = experiment.new_batch_trial()
    trial.add_arm(get_jenatton_arm(0))
    trial.add_arm(get_jenatton_arm(1))
    return trial


class DeterministicGenerationNode(ExternalGenerationNode):
    """
    A GenerationNode that explores a discrete search space with one parameter
    deterministically.
    """

    def __init__(
        self,
        search_space: SearchSpace,
    ) -> None:
        if len(search_space.parameters) != 1:
            raise ValueError(
                "DeterministicGenerationNode only supports search spaces with one "
                "parameter."
            )
        param = list(search_space.parameters.values())[0]
        if not isinstance(param, ChoiceParameter):
            raise ValueError(
                "DeterministicGenerationNode only supports ChoiceParameters."
            )
        super().__init__(node_name="Deterministic")

        self.param_name: str = param.name
        self.iterator: Iterator[TParamValue] = iter(param.values)

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        return

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        return {self.param_name: next(self.iterator)}


def get_discrete_search_space(n_values: int = 20) -> SearchSpace:
    return SearchSpace(
        parameters=[
            ChoiceParameter(
                name="x0",
                parameter_type=ParameterType.INT,
                values=list(range(n_values)),
            )
        ]
    )


def get_async_benchmark_method(
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
) -> BenchmarkMethod:
    gs = GenerationStrategy(
        nodes=[DeterministicGenerationNode(search_space=get_discrete_search_space())]
    )
    return BenchmarkMethod(
        generation_strategy=gs,
        distribute_replications=False,
        max_pending_trials=2,
        batch_size=1,
        early_stopping_strategy=early_stopping_strategy,
    )


def get_async_benchmark_problem(
    map_data: bool,
    step_runtime_fn: TBenchmarkStepRuntimeFunction | None = None,
    n_steps: int = 1,
    lower_is_better: bool = False,
) -> BenchmarkProblem:
    search_space = get_discrete_search_space()
    test_function = IdentityTestFunction(n_steps=n_steps)
    optimization_config = get_soo_opt_config(
        outcome_names=["objective"],
        use_map_metric=map_data,
        observe_noise_sd=True,
        lower_is_better=lower_is_better,
    )

    return BenchmarkProblem(
        name="test",
        search_space=search_space,
        optimization_config=optimization_config,
        test_function=test_function,
        num_trials=4,
        baseline_value=19 if lower_is_better else 0,
        optimal_value=0 if lower_is_better else 19,
        step_runtime_function=step_runtime_fn,
    )


def get_benchmark_metric() -> BenchmarkMetric:
    return BenchmarkMetric(name="test", lower_is_better=True)


def get_benchmark_map_metric() -> BenchmarkMapMetric:
    return BenchmarkMapMetric(name="test", lower_is_better=True)


def get_benchmark_time_varying_metric() -> BenchmarkTimeVaryingMetric:
    return BenchmarkTimeVaryingMetric(name="test", lower_is_better=True)


def get_benchmark_map_unavailable_while_running_metric() -> (
    BenchmarkMapUnavailableWhileRunningMetric
):
    return BenchmarkMapUnavailableWhileRunningMetric(name="test", lower_is_better=True)
