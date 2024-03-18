# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
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
            "lower_is_better": True,
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "branin": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "lower_is_better": True,
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "branin_currin": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "branin_currin30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n, num_trials: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=BraninCurrin,
                test_problem_kwargs={},
                num_trials=num_trials,
                observe_noise_sd=False,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30, "num_trials": 30},
    ),
    "griewank4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Griewank,
            "test_problem_kwargs": {"dim": 4},
            "lower_is_better": True,
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "hartmann3": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 3},
            "lower_is_better": True,
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "hartmann6": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "lower_is_better": True,
            "num_trials": 35,
            "observe_noise_sd": False,
        },
    ),
    "hartmann30": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n, num_trials: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=synthetic.Hartmann,
                test_problem_kwargs={"dim": 6},
                lower_is_better=True,
                num_trials=num_trials,
                observe_noise_sd=False,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30, "num_trials": 25},
    ),
    "hpo_pytorch_cnn_MNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={
            "name": "MNIST",
            "num_trials": 20,
        },
    ),
    "hpo_pytorch_cnn_FashionMNIST": BenchmarkProblemRegistryEntry(
        factory_fn=PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name,
        factory_kwargs={
            "name": "FashionMNIST",
            "num_trials": 50,
        },
    ),
    "jenatton": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 50, "observe_noise_sd": False},
    ),
    "levy4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Levy,
            "test_problem_kwargs": {"dim": 4},
            "lower_is_better": True,
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "powell4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Powell,
            "test_problem_kwargs": {"dim": 4},
            "lower_is_better": True,
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "rosenbrock4": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Rosenbrock,
            "test_problem_kwargs": {"dim": 4},
            "lower_is_better": True,
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "six_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.SixHumpCamel,
            "test_problem_kwargs": {},
            "lower_is_better": True,
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "three_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.ThreeHumpCamel,
            "test_problem_kwargs": {},
            "lower_is_better": True,
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    # Problems where we observe the noise level
    "branin_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "lower_is_better": True,
            "num_trials": 20,
            "observe_noise_sd": True,
        },
    ),
    "branin_currin_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": True,
        },
    ),
    "branin_currin30_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n, num_trials: embed_higher_dimension(
            problem=MultiObjectiveBenchmarkProblem.from_botorch_multi_objective(
                test_problem_class=BraninCurrin,
                test_problem_kwargs={},
                num_trials=num_trials,
                observe_noise_sd=True,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30, "num_trials": 30},
    ),
    "hartmann6_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "lower_is_better": True,
            "num_trials": 50,
            "observe_noise_sd": True,
        },
    ),
    "hartmann30_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=lambda n, num_trials: embed_higher_dimension(
            problem=SingleObjectiveBenchmarkProblem.from_botorch_synthetic(
                test_problem_class=synthetic.Hartmann,
                test_problem_kwargs={"dim": 6},
                lower_is_better=True,
                num_trials=num_trials,
                observe_noise_sd=True,
            ),
            total_dimensionality=n,
        ),
        factory_kwargs={"n": 30, "num_trials": 25},
    ),
    "jenatton_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 25, "observe_noise_sd": True},
    ),
    "constrained_gramacy_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        factory_kwargs={
            "test_problem_class": synthetic.ConstrainedGramacy,
            "test_problem_kwargs": {},
            "lower_is_better": True,
            "num_trials": 50,
            "observe_noise_sd": True,
        },
    ),
}


def get_problem(problem_name: str, **additional_kwargs: Any) -> BenchmarkProblem:
    entry = BENCHMARK_PROBLEM_REGISTRY[problem_name]
    kwargs = copy.copy(entry.factory_kwargs)
    kwargs.update(additional_kwargs)
    return entry.factory_fn(**kwargs)
