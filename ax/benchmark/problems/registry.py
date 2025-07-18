# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ax.benchmark.benchmark_problem import BenchmarkProblem, create_problem_from_botorch
from ax.benchmark.problems.hpo.torchvision import (
    get_pytorch_cnn_torchvision_benchmark_problem,
)
from ax.benchmark.problems.runtime_funcs import int_from_params
from ax.benchmark.problems.surrogate.lcbench.early_stopping import (
    get_lcbench_early_stopping_benchmark_problem,
)
from ax.benchmark.problems.surrogate.lcbench.transfer_learning import (
    get_lcbench_benchmark_problem,
)
from ax.benchmark.problems.synthetic.bandit import get_bandit_problem
from ax.benchmark.problems.synthetic.discretized.mixed_integer import (
    get_discrete_ackley,
    get_discrete_hartmann,
    get_discrete_rosenbrock,
)
from ax.benchmark.problems.synthetic.hss.jenatton import get_jenatton_benchmark_problem
from botorch.test_functions import synthetic
from botorch.test_functions.multi_objective import BraninCurrin


@dataclass
class BenchmarkProblemRegistryEntry:
    factory_fn: Callable[..., BenchmarkProblem]
    factory_kwargs: dict[str, Any]


# Baseline values were obtained with `compute_baseline_value_from_sobol`
BENCHMARK_PROBLEM_REGISTRY: dict[str, BenchmarkProblemRegistryEntry] = {
    "ackley4": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Ackley,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "ackley4_async_noisy": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Ackley,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "noise_std": 1.0,
            "observe_noise_sd": False,
            "step_runtime_function": int_from_params,
            "name": "ackley4_async_noisy",
        },
    ),
    "Bandit": BenchmarkProblemRegistryEntry(
        factory_fn=get_bandit_problem, factory_kwargs={}
    ),
    "branin": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "branin_currin": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "branin_currin30": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
            "baseline_value": 3.0187520516793587,
            "n_dummy_dimensions": 28,
        },
    ),
    "griewank4": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Griewank,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "hartmann3": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 3},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "hartmann6": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "num_trials": 35,
            "observe_noise_sd": False,
        },
    ),
    "hartmann30": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "num_trials": 25,
            "observe_noise_sd": False,
            "n_dummy_dimensions": 24,
        },
    ),
    "hpo_pytorch_cnn_MNIST": BenchmarkProblemRegistryEntry(
        factory_fn=get_pytorch_cnn_torchvision_benchmark_problem,
        factory_kwargs={
            "name": "MNIST",
            "num_trials": 20,
        },
    ),
    "hpo_pytorch_cnn_FashionMNIST": BenchmarkProblemRegistryEntry(
        factory_fn=get_pytorch_cnn_torchvision_benchmark_problem,
        factory_kwargs={
            "name": "FashionMNIST",
            "num_trials": 50,
        },
    ),
    "jenatton": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 50, "observe_noise_sd": False},
    ),
    "LCBench:v1 Fashion-MNIST": BenchmarkProblemRegistryEntry(
        get_lcbench_benchmark_problem, factory_kwargs={"dataset_name": "Fashion-MNIST"}
    ),
    "levy4": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Levy,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "powell4": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Powell,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "rosenbrock4": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Rosenbrock,
            "test_problem_kwargs": {"dim": 4},
            "num_trials": 40,
            "observe_noise_sd": False,
        },
    ),
    "six_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.SixHumpCamel,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    "three_hump_camel": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.ThreeHumpCamel,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": False,
        },
    ),
    # Problems where we observe the noise level
    "branin_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Branin,
            "test_problem_kwargs": {},
            "num_trials": 20,
            "observe_noise_sd": True,
        },
    ),
    "branin_currin_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": True,
        },
    ),
    "branin_currin30_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": BraninCurrin,
            "test_problem_kwargs": {},
            "num_trials": 30,
            "observe_noise_sd": True,
            "n_dummy_dimensions": 28,
        },
    ),
    "hartmann6_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "num_trials": 50,
            "observe_noise_sd": True,
        },
    ),
    "hartmann30_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.Hartmann,
            "test_problem_kwargs": {"dim": 6},
            "observe_noise_sd": True,
            "num_trials": 25,
            "n_dummy_dimensions": 24,
        },
    ),
    "jenatton_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=get_jenatton_benchmark_problem,
        factory_kwargs={"num_trials": 25, "observe_noise_sd": True},
    ),
    "constrained_gramacy_observed_noise": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.ConstrainedGramacy,
            "test_problem_kwargs": {},
            "num_trials": 50,
            "observe_noise_sd": True,
        },
    ),
    "Discrete Hartmann": BenchmarkProblemRegistryEntry(
        factory_fn=get_discrete_hartmann,
        factory_kwargs={},
    ),
    "Discrete Ackley": BenchmarkProblemRegistryEntry(
        factory_fn=get_discrete_ackley, factory_kwargs={}
    ),
    "Discrete Rosenbrock": BenchmarkProblemRegistryEntry(
        factory_fn=get_discrete_rosenbrock, factory_kwargs={}
    ),
    "PressureVessel": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.PressureVessel,
            "test_problem_kwargs": {},
            "num_trials": 50,
        },
    ),
    "TensionCompressionString": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.TensionCompressionString,
            "test_problem_kwargs": {},
            "num_trials": 50,
        },
    ),
    "WeldedBeamSO": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.WeldedBeamSO,
            "test_problem_kwargs": {},
            "num_trials": 50,
        },
    ),
    "SpeedReducer": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.SpeedReducer,
            "test_problem_kwargs": {},
            "num_trials": 50,
        },
    ),
    "KeaneBumpFunction2": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.KeaneBumpFunction,
            "test_problem_kwargs": {"dim": 2},
            "num_trials": 30,
        },
    ),
    "KeaneBumpFunction10": BenchmarkProblemRegistryEntry(
        factory_fn=create_problem_from_botorch,
        factory_kwargs={
            "test_problem_class": synthetic.KeaneBumpFunction,
            "test_problem_kwargs": {"dim": 10},
            "num_trials": 50,
        },
    ),
    **{
        f"LCBench:ES {dataset_name}": BenchmarkProblemRegistryEntry(
            get_lcbench_early_stopping_benchmark_problem,
            factory_kwargs={"dataset_name": dataset_name},
        )
        for dataset_name in (
            "airlines",
            "albert",
            "covertype",
            "christine",
            "Fashion-MNIST",
        )
    },
}


def get_benchmark_problem(
    problem_key: str,
    registry: Mapping[str, BenchmarkProblemRegistryEntry] | None = None,
    **additional_kwargs: Any,
) -> BenchmarkProblem:
    """
    Generate a benchmark problem from a key, registry, and additional arguments.

    Args:
        problem_key: The key by which a `BenchmarkProblemRegistryEntry` is
            looked up in the registry; a problem will then be generated from
            that entry and `additional_kwargs`. Note that this is not
            necessarily the same as the `name` attribute of the problem, and
            that one `problem_key` can generate several different
            `BenchmarkProblem`s by passing `additional_kwargs`. However, it is a
            good practice to maintain a 1:1 mapping between `problem_key` and
            the name.
        registry: If not provided, uses `BENCHMARK_PROBLEM_REGISTRY` to use
            problems defined within Ax.
        additional_kwargs: Additional kwargs to pass to the factory function of
            the `BenchmarkProblemRegistryEntry`.
    """
    registry = BENCHMARK_PROBLEM_REGISTRY if registry is None else registry
    entry = registry[problem_key]
    kwargs = copy.copy(entry.factory_kwargs)
    kwargs.update(additional_kwargs)
    return entry.factory_fn(**kwargs)
