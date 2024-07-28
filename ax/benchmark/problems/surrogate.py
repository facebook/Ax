# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Union

from ax.benchmark.metrics.base import BenchmarkMetricBase

from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.search_space import SearchSpace
from ax.utils.common.base import Base


class SurrogateBenchmarkProblemBase(Base):
    """
    Base class for SOOSurrogateBenchmarkProblem and MOOSurrogateBenchmarkProblem.

    Its `runner` is a `SurrogateRunner`, which allows for the surrogate to be
    constructed lazily and datasets to be downloaded lazily.
    """

    def __init__(
        self,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        num_trials: int,
        runner: SurrogateRunner,
        is_noiseless: bool,
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
    ) -> None:
        """Construct a `SurrogateBenchmarkProblemBase` instance.

        Args:
            name: The name of the benchmark problem.
            search_space: The search space to optimize over.
            optimization_config: THe optimization config for the problem.
            num_trials: The number of trials to run.
            runner: A `SurrogateRunner`, allowing for lazy construction of the
                surrogate and datasets.
            observe_noise_stds: Whether or not to observe the observation noise
                level for each metric. If True/False, observe the the noise standard
                deviation for all/no metrics. If a dictionary, specify this for
                individual metrics (metrics not appearing in the dictionary will
                be assumed to not provide observation noise levels).
            tracking_metrics: Additional tracking metrics to compute during the
                optimization (not used to inform the optimization).
        """

        self.name = name
        self.search_space = search_space
        self.optimization_config = optimization_config
        self.num_trials = num_trials
        self.observe_noise_stds = observe_noise_stds
        self.tracking_metrics: List[BenchmarkMetricBase] = tracking_metrics or []
        self.runner = runner
        self.is_noiseless = is_noiseless

    @property
    def has_ground_truth(self) -> bool:
        # All surrogate-based problems have a ground truth
        return True

    def __repr__(self) -> str:
        """
        Return a string representation that includes only the attributes that
        print nicely and contain information likely to be useful.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"optimization_config={self.optimization_config}, "
            f"num_trials={self.num_trials}, "
            f"is_noiseless={self.is_noiseless}, "
            f"observe_noise_stds={self.observe_noise_stds}, "
            f"noise_stds={self.runner.noise_stds}, "
            f"tracking_metrics={self.tracking_metrics})"
        )


class SOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `MultiObjectiveBenchmarkProblem`,
    but its runner is not constructed until needed, to allow for deferring
    constructing the surrogate and downloading data. The surrogate is only
    defined when `runner` is accessed or `set_runner` is called.
    """

    def __init__(
        self,
        optimal_value: float,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        num_trials: int,
        runner: SurrogateRunner,
        is_noiseless: bool,
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            observe_noise_stds=observe_noise_stds,
            runner=runner,
            is_noiseless=is_noiseless,
        )
        self.optimal_value = optimal_value


class MOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `MultiObjectiveBenchmarkProblem`,
    but its runner is not constructed until needed, to allow for deferring
    constructing the surrogate and downloading data. The surrogate is only
    defined when `runner` is accessed or `set_runner` is called.
    """

    optimization_config: MultiObjectiveOptimizationConfig

    def __init__(
        self,
        optimal_value: float,
        reference_point: List[float],
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: MultiObjectiveOptimizationConfig,
        num_trials: int,
        runner: SurrogateRunner,
        is_noiseless: bool,
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            observe_noise_stds=observe_noise_stds,
            tracking_metrics=tracking_metrics,
            runner=runner,
            is_noiseless=is_noiseless,
        )
        self.reference_point = reference_point
        self.optimal_value = optimal_value
