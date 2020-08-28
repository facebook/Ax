#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.utils.measurement.synthetic_functions import from_botorch
from ax.utils.testing.core_stubs import (
    get_augmented_branin_optimization_config,
    get_augmented_hartmann_optimization_config,
    get_branin_search_space,
    get_hartmann_search_space,
)
from botorch.test_functions.synthetic import Ackley, Branin


# Initialize the single-fidelity problems
ackley = SimpleBenchmarkProblem(f=from_botorch(Ackley()), noise_sd=0.0, minimize=True)
branin = SimpleBenchmarkProblem(f=from_botorch(Branin()), noise_sd=0.0, minimize=True)
single_fidelity_problem_group = [ackley, branin]

# Initialize the multi-fidelity problems
augmented_branin = BenchmarkProblem(
    search_space=get_branin_search_space(with_fidelity_parameter=True),
    optimization_config=get_augmented_branin_optimization_config(),
)
augmented_hartmann = BenchmarkProblem(
    search_space=get_hartmann_search_space(with_fidelity_parameter=True),
    optimization_config=get_augmented_hartmann_optimization_config(),
)
multi_fidelity_problem_group = [augmented_branin, augmented_hartmann]

# Gather all of the problems
MODULAR_BOTORCH_PROBLEM_GROUPS = {
    "single_fidelity_models": single_fidelity_problem_group,
    "multi_fidelity_models": multi_fidelity_problem_group,
}
