# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Tuple

from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    BenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.storage.json_store.decoder import object_from_json
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.synthetic import Branin, Ackley


# Mapping from problem name to (BoTorch class, Ax factory method, path to baseline)
_REGISTRY = {
    "ackley": (
        Ackley,
        SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        "baseline_results/synthetic/ackley.json",
    ),
    "branin": (
        Branin,
        SingleObjectiveBenchmarkProblem.from_botorch_synthetic,
        "baseline_results/synthetic/branin.json",
    ),
    "branin_currin": (
        BraninCurrin,
        MultiObjectiveBenchmarkProblem.from_botorch_multi_objective,
        "baseline_results/synthetic/branin_currin.json",
    ),
}


def get_problem_and_baseline_from_botorch(
    problem_name: str,
) -> Tuple[BenchmarkProblem, AggregatedBenchmarkResult]:
    botorch_cls, factory, baseline_path = _REGISTRY[problem_name]

    problem = factory(test_problem=botorch_cls())

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, baseline_path)

    with open(file=file_path) as file:
        loaded = json.loads(file.read())
        baseline_result = object_from_json(loaded)

        return (problem, baseline_result)
