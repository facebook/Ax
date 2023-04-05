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
from botorch.test_functions import synthetic
from botorch.test_functions.multi_objective import BraninCurrin


@dataclass
class BenchmarkProblemRegistryEntry:
    factory_fn: Callable[..., BenchmarkProblem]
    factory_kwargs: Dict[str, Any]


BENCHMARK_PROBLEM_REGISTRY = {
    "ackley4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Ackley,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "infer_noise": True,
        },
    ),
    "branin": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": True,
        },
    ),
    "branin_currin": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": True,
        },
    ),
    "branin_currin30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=BraninCurrin,
                test_problem_kwargs={},
                num_trials=100,
                infer_noise=True,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "griewank4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Griewank,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "infer_noise": True,
        },
    ),
    "hartmann3": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 3},
            "num_trials": 30,
            "infer_noise": True,
        },
    ),
    "hartmann6": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "num_trials": 50,
            "infer_noise": True,
        },
    ),
    "hartmann30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=synthetic.Hartmann,
                test_problem_kwargs={"dim": 6},
                num_trials=100,
                infer_noise=True,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "hpo_pytorch_cnn_MNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "MNIST", "num_trials": 50, "infer_noise": True},
    ),
    "hpo_pytorch_cnn_FashionMNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={"name": "FashionMNIST", "num_trials": 50, "infer_noise": True},
    ),
    "jenatton": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 50, "infer_noise": True},
    ),
    "levy4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Levy,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "infer_noise": True,
        },
    ),
    "powell4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Powell,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "infer_noise": True,
        },
    ),
    "rosenbrock4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Rosenbrock,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "infer_noise": True,
        },
    ),
    "six_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.SixHumpCamel,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": True,
        },
    ),
    "three_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.ThreeHumpCamel,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": True,
        },
    ),
    # Problems without inferred noise
    "branin_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": False,
        },
    ),
    "branin_currin_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "infer_noise": False,
        },
    ),
    "branin_currin30_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=BraninCurrin,
                test_problem_kwargs={},
                num_trials=100,
                infer_noise=False,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "hartmann6_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "num_trials": 50,
            "infer_noise": False,
        },
    ),
    "hartmann30_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=synthetic.Hartmann,
                test_problem_kwargs={"dim": 6},
                num_trials=100,
                infer_noise=False,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30},
    ),
    "jenatton_fixed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 50, "infer_noise": False},
    ),
}


def get_problem(
    problem_name: str,
) -> BenchmarkProblem:
    entry = BENCHMARK_PROBLEM_REGISTRY[problem_name]
    return entry.factory_fn(**entry.factory_kwargs)
