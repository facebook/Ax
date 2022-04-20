# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Callable, Tuple

from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    BenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.benchmark.problems.hd_embedding import embed_higher_dimension
from ax.benchmark.problems.hpo.torchvision import PyTorchCNNTorchvisionBenchmarkProblem
from ax.storage.json_store.decoder import object_from_json
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Hartmann, Branin, Ackley


@dataclass
class BenchmarkProblemRegistryEntry:
    factory_fn: Callable[..., BenchmarkProblem]
    factory_kwargs: Dict[str, Any]
    baseline_results_path: str


BENCHMARK_PROBLEM_REGISTRY = {
    "ackley": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Ackley()},
        baseline_results_path="baseline_results/synthetic/ackley.json",
    ),
    "branin": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={"test_problem": Branin()},
        baseline_results_path="baseline_results/synthetic/branin.json",
    ),
    "branin_currin": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={"test_problem": BraninCurrin()},
        baseline_results_path="baseline_results/synthetic/branin_currin.json",
    ),
    "branin_currin30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem=BraninCurrin()
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
        baseline_results_path="baseline_results/synthetic/hd/branin_currin_30d.json",
    ),
    "hartmann50": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem=Hartmann(dim=6)
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 50},
        baseline_results_path="baseline_results/synthetic/hd/hartmann_50d.json",
    ),
    "hpo_pytorch_cnn_MNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "MNIST"},
        baseline_results_path="baseline_results/hpo/torchvision/mnist.json",
    ),
    "hpo_pytorch_cnn_FashionMNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "FashionMNIST"},
        baseline_results_path="baseline_results/hpo/torchvision/fashion_mnist.json",
    ),
}


def get_problem_and_baseline(
    problem_name: str,
) -> Tuple[BenchmarkProblem, AggregatedBenchmarkResult]:
    entry = BENCHMARK_PROBLEM_REGISTRY[problem_name]

    problem = entry.factory_fn(**entry.factory_kwargs)

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, entry.baseline_results_path)

    with open(file=file_path) as file:
        loaded = json.loads(file.read())
        baseline_result = object_from_json(loaded)

        return (problem, baseline_result)
