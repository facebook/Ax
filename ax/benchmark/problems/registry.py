# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.problems.hd_embedding import embed_higher_dimension
from ax.benchmark.problems.hpo.torchvision import PyTorchCNNTorchvisionBenchmarkProblem
from ax.benchmark.problems.synthetic.hss.jenatton import get_jenatton_benchmark_problem
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Ackley, Branin, Hartmann, Powell


@dataclass
class BenchmarkProblemRegistryEntry:
    factory_fn: Callable[..., BenchmarkProblem]
    factory_kwargs: Dict[str, Any]


BENCHMARK_PROBLEM_REGISTRY = {
    "ackley": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Ackley(), "num_trials": 50},
    ),
    "branin": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Branin(), "num_trials": 30},
    ),
    "branin_currin": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={"test_problem": BraninCurrin(), "num_trials": 30},
    ),
    "branin_currin30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem=BraninCurrin(), num_trials=100
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "hartmann6": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Hartmann(dim=6), "num_trials": 50},
    ),
    "hartmann30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem=Hartmann(dim=6), num_trials=100
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "hpo_pytorch_cnn_MNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "MNIST", "num_trials": 50},
    ),
    "hpo_pytorch_cnn_FashionMNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "FashionMNIST", "num_trials": 50},
    ),
    "jenatton": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 50},
    ),
    "powell": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Powell(), "num_trials": 50},
    ),
}


def get_problem(
    problem_name: str,
) -> BenchmarkProblem:
    entry = BENCHMARK_PROBLEM_REGISTRY[problem_name]
    return entry.factory_fn(**entry.factory_kwargs)
