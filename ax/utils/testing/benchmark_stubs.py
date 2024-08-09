#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional

import numpy as np
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    create_multi_objective_problem_from_botorch,
    create_single_objective_problem_from_botorch,
    MultiObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.problems.surrogate import (
    MOOSurrogateBenchmarkProblem,
    SOOSurrogateBenchmarkProblem,
)
from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.experiment import Experiment
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
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.test_functions.multi_objective import BraninCurrin, ConstrainedBraninCurrin
from botorch.test_functions.synthetic import Branin


def get_single_objective_benchmark_problem(
    observe_noise_sd: bool = False,
    num_trials: int = 4,
    test_problem_kwargs: Optional[dict[str, Any]] = None,
) -> BenchmarkProblem:
    return create_single_objective_problem_from_botorch(
        test_problem_class=Branin,
        test_problem_kwargs=test_problem_kwargs or {},
        lower_is_better=True,
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
    )


def get_multi_objective_benchmark_problem(
    observe_noise_sd: bool = False, num_trials: int = 4
) -> MultiObjectiveBenchmarkProblem:
    return create_multi_objective_problem_from_botorch(
        test_problem_class=BraninCurrin,
        test_problem_kwargs={},
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
    )


def get_constrained_multi_objective_benchmark_problem(
    observe_noise_sd: bool = False, num_trials: int = 4
) -> MultiObjectiveBenchmarkProblem:
    return create_multi_objective_problem_from_botorch(
        test_problem_class=ConstrainedBraninCurrin,
        test_problem_kwargs={},
        num_trials=num_trials,
        observe_noise_sd=observe_noise_sd,
    )


def get_sobol_benchmark_method() -> BenchmarkMethod:
    return BenchmarkMethod(
        name="SOBOL",
        generation_strategy=GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)],
            name="SOBOL",
        ),
        scheduler_options=SchedulerOptions(
            total_trials=4, init_seconds_between_polls=0
        ),
    )


def get_soo_surrogate() -> SOOSurrogateBenchmarkProblem:
    experiment = get_branin_experiment(with_completed_trial=True)
    surrogate = TorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        model=BoTorchModel(surrogate=Surrogate(botorch_model_class=SingleTaskGP)),
        data=experiment.lookup_data(),
        transforms=[],
    )
    runner = SurrogateRunner(
        name="test",
        search_space=experiment.search_space,
        outcome_names=["branin"],
        get_surrogate_and_datasets=lambda: (surrogate, []),
    )
    return SOOSurrogateBenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=checked_cast(
            OptimizationConfig, experiment.optimization_config
        ),
        num_trials=6,
        observe_noise_stds=True,
        optimal_value=0.0,
        runner=runner,
        is_noiseless=runner.is_noiseless,
    )


def get_moo_surrogate() -> MOOSurrogateBenchmarkProblem:
    experiment = get_branin_experiment_with_multi_objective(with_completed_trial=True)
    surrogate = TorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        model=BoTorchModel(surrogate=Surrogate(botorch_model_class=SingleTaskGP)),
        data=experiment.lookup_data(),
        transforms=[],
    )

    runner = SurrogateRunner(
        name="test",
        search_space=experiment.search_space,
        outcome_names=["branin_a", "branin_b"],
        get_surrogate_and_datasets=lambda: (surrogate, []),
    )
    return MOOSurrogateBenchmarkProblem(
        name="test",
        search_space=experiment.search_space,
        optimization_config=checked_cast(
            MultiObjectiveOptimizationConfig, experiment.optimization_config
        ),
        num_trials=10,
        observe_noise_stds=True,
        optimal_value=1.0,
        runner=runner,
        is_noiseless=runner.is_noiseless,
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
            runner=problem.runner,
            is_test=True,
        ),
        optimization_trace=np.array([3, 2, 1, 0.1]),
        score_trace=np.array([3, 2, 1, 0.1]),
        fit_time=0.1,
        gen_time=0.2,
    )


def get_aggregated_benchmark_result() -> AggregatedBenchmarkResult:
    result = get_benchmark_result()
    return AggregatedBenchmarkResult.from_benchmark_results([result, result])
