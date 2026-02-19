# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, InitVar
from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ax.benchmark.benchmark_problem import BenchmarkProblem, get_soo_opt_config
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.problems.data import AbstractParquetDataLoader
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.exceptions.core import UserInputError

DEFAULT_NUM_TRIALS = 50
DEFAULT_METRIC_NAME: str = "valid_mse"
DATASET_NAMES: list[str] = [
    "naval_propulsion",
    "parkinsons_telemonitoring",
    "slice_localization",
    "protein_structure",
]


class FCNetDataLoader(AbstractParquetDataLoader):
    def __init__(
        self,
        dataset_name: str,
        stem: str,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(
            benchmark_name="FCNetLite",
            dataset_name=dataset_name,
            stem=stem,
            cache_dir=cache_dir,
        )

    @property
    def url(self) -> None:
        return


def get_fcnet_search_space() -> SearchSpace:
    """Construct the FCNet search space."""
    parameters = [
        ChoiceParameter(
            name="init_lr",
            parameter_type=ParameterType.FLOAT,
            values=[5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
            is_ordered=True,
            sort_values=True,
        ),
        ChoiceParameter(
            name="lr_schedule",
            parameter_type=ParameterType.STRING,
            values=["cosine", "const"],
            is_ordered=False,
            sort_values=False,
        ),
        ChoiceParameter(
            name="batch_size",
            parameter_type=ParameterType.INT,
            values=[2**k for k in range(3, 7)],
            is_ordered=True,
            sort_values=True,
        ),
    ]
    for i in range(2):
        parameters.extend(
            [
                ChoiceParameter(
                    name=f"activation_fn_{i + 1}",
                    parameter_type=ParameterType.STRING,
                    values=["tanh", "relu"],
                    is_ordered=False,
                    sort_values=False,
                ),
                ChoiceParameter(
                    name=f"n_units_{i + 1}",
                    parameter_type=ParameterType.INT,
                    values=[2**k for k in range(4, 10)],
                    is_ordered=True,
                    sort_values=True,
                ),
                ChoiceParameter(
                    name=f"dropout_{i + 1}",
                    parameter_type=ParameterType.FLOAT,
                    values=[0.3 * k for k in range(3)],
                    is_ordered=True,
                    sort_values=True,
                ),
            ]
        )

    search_space: SearchSpace = SearchSpace(parameters=parameters)
    return search_space


def get_fcnet_optimization_config(
    metric_name: str = DEFAULT_METRIC_NAME,
    observe_noise_sd: bool = False,
    use_map_metric: bool = False,
) -> OptimizationConfig:
    return get_soo_opt_config(
        outcome_names=[metric_name],
        lower_is_better=False,
        observe_noise_sd=observe_noise_sd,
        use_map_metric=use_map_metric,
    )


@dataclass(kw_only=True)
class FCNetBenchmarkTestFunction(BenchmarkTestFunction):
    metric_series: pd.Series = field(init=False)
    n_steps: int = field(init=False)
    outcome_names: Sequence[str] = field(default_factory=lambda: [DEFAULT_METRIC_NAME])
    dataset_name: str
    repetition: int

    def __post_init__(self) -> None:
        if len(self.outcome_names) != 1:
            raise ValueError("Exactly one outcome is supported currently")

        result_df = FCNetDataLoader(self.dataset_name, stem="results").read()
        metrics_df = FCNetDataLoader(self.dataset_name, stem="metrics").read()

        metric_name = self.outcome_names[0]
        self.metric_series = metrics_df[metric_name]
        self.n_steps = 100

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> torch.Tensor:
        key = json.dumps(params, sort_keys=True)
        trial_id = hashlib.md5(key.encode("utf-8")).hexdigest()
        y = torch.from_numpy(self.metric_series.loc[trial_id, self.repetition].values)
        return y.unsqueeze(0)

    def step_runtime(self, params: Mapping[str, TParamValue]) -> float:
        Y = np.random.randn(1)  # shape: (1,)
        return Y.item()


def get_fcnet_early_stopping_benchmark_problem(
    dataset_name: str,
    metric_name: str = DEFAULT_METRIC_NAME,
    num_trials: int = DEFAULT_NUM_TRIALS,
    constant_step_runtime: bool = False,
    noise_std: Mapping[str, float] | float = 0.0,
    observe_noise_sd: bool = False,
    repetition: int = 0,
) -> BenchmarkProblem:
    """Construct an LCBench early-stopping benchmark problem.

    Args:
        dataset_name: Must be one of the keys of `DEFAULT_AND_OPTIMAL_VALUES`, which
            correspond to the names of the datasets available in LCBench.
        metric_name: The name of the metric to use for the objective.
        num_trials: The number of optimization trials to run.
        constant_step_runtime: Determines if the step runtime is fixed or varies
            based on the hyperparameters.
        noise_std: The standard deviation of the observation noise.
        observe_noise_sd: Whether to report the standard deviation of the
            obervation noise.
        seed: The random seed used in training the surrogate model to ensure
            reproducibility and consistency of results.

    Returns:
        An LCBench surrogate benchmark problem.
    """

    if dataset_name not in DATASET_NAMES:
        raise UserInputError(f"`dataset_name` must be one of {sorted(DATASET_NAMES)}")

    name = f"FCNet_Tabular_{dataset_name}_{metric_name}:v1"

    # _, optimal_value = DEFAULT_AND_OPTIMAL_VALUES[dataset_name]
    optimal_value = 0.0

    # baseline_value = BASELINE_VALUES[dataset_name]
    baseline_value = 3.1415926535

    search_space: SearchSpace = get_fcnet_search_space()
    optimization_config: OptimizationConfig = get_fcnet_optimization_config(
        metric_name=metric_name,
        observe_noise_sd=observe_noise_sd,
        use_map_metric=True,
    )

    test_function = FCNetBenchmarkTestFunction(
        dataset_name=dataset_name, repetition=repetition
    )

    step_runtime_function = (
        None if constant_step_runtime else test_function.step_runtime
    )

    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=optimal_value,
        baseline_value=baseline_value,
        test_function=test_function,
        step_runtime_function=step_runtime_function,
        noise_std=noise_std,
    )
