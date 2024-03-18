# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Dict, List, Optional, Tuple, Union

from ax.benchmark.metrics.base import BenchmarkMetricBase

from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.utils.datasets import SupervisedDataset


class SurrogateBenchmarkProblemBase(Base):
    """
    Base class for SOOSurrogateBenchmarkProblem and MOOSurrogateBenchmarkProblem.

    Allows for lazy creation of objects needed to construct a `runner`,
    including a surrogate and datasets.
    """

    def __init__(
        self,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        num_trials: int,
        outcome_names: List[str],
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
        noise_stds: Union[float, Dict[str, float]] = 0.0,
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        """Construct a `SurrogateBenchmarkProblemBase` instance.

        Args:
            name: The name of the benchmark problem.
            search_space: The search space to optimize over.
            optimization_config: THe optimization config for the problem.
            num_trials: The number of trials to run.
            outcome_names: The names of the metrics the benchmark problem
                produces outcome observations for.
            observe_noise_stds: Whether or not to observe the observation noise
                level for each metric. If True/False, observe the the noise standard
                deviation for all/no metrics. If a dictionary, specify this for
                individual metrics (metrics not appearing in the dictionary will
                be assumed to not provide observation noise levels).
            noise_stds: The standard deviation(s) of the observation noise(s).
                If a single value is provided, it is used for all metrics. Providing
                a dictionary allows specifying different noise levels for different
                metrics (metrics not appearing in the dictionary will be assumed to
                be noiseless - but not necessarily be known to the problem to be
                noiseless).
            get_surrogate_and_datasets: A factory function that retunrs the Surrogate
                and a list of datasets to be used by the surrogate.
            tracking_metrics: Additional tracking metrics to compute during the
                optimization (not used to inform the optimization).
        """

        if get_surrogate_and_datasets is None and _runner is None:
            raise ValueError(
                "Either `get_surrogate_and_datasets` or `_runner` required."
            )
        self.name = name
        self.search_space = search_space
        self.optimization_config = optimization_config
        self.num_trials = num_trials
        self.outcome_names = outcome_names
        self.observe_noise_stds = observe_noise_stds
        self.noise_stds = noise_stds
        self.get_surrogate_and_datasets = get_surrogate_and_datasets
        self.tracking_metrics: List[BenchmarkMetricBase] = tracking_metrics or []
        self._runner = _runner

    @property
    def is_noiseless(self) -> bool:
        if self.noise_stds is None:
            return True
        if isinstance(self.noise_stds, float):
            return self.noise_stds == 0.0
        return all(std == 0.0 for std in checked_cast(dict, self.noise_stds).values())

    @property
    def has_ground_truth(self) -> bool:
        # All surrogate-based problems have a ground truth
        return True

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if type(other) is not type(self):
            return False

        # Checking the whole datasets' equality here would be too expensive to be
        # worth it; just check names instead
        return self.name == other.name

    def set_runner(self) -> None:
        surrogate, datasets = not_none(self.get_surrogate_and_datasets)()

        self._runner = SurrogateRunner(
            name=self.name,
            surrogate=surrogate,
            datasets=datasets,
            search_space=self.search_space,
            outcome_names=self.outcome_names,
            noise_stds=self.noise_stds,
        )

    @property
    def runner(self) -> Runner:
        if self._runner is None:
            self.set_runner()
        return not_none(self._runner)

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
            f"noise_stds={self.noise_stds}, "
            f"tracking_metrics={self.tracking_metrics})"
        )


class SOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `SingleObjectiveBenchmarkProblem`,
    but allows for constructing from a surrogate.
    """

    def __init__(
        self,
        optimal_value: float,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        num_trials: int,
        outcome_names: List[str],
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
        noise_stds: Union[float, Dict[str, float]] = 0.0,
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            outcome_names=outcome_names,
            observe_noise_stds=observe_noise_stds,
            noise_stds=noise_stds,
            get_surrogate_and_datasets=get_surrogate_and_datasets,
            tracking_metrics=tracking_metrics,
            _runner=_runner,
        )
        self.optimal_value = optimal_value


class MOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `MultiObjectiveBenchmarkProblem`,
    but its runner is not constructed until needed, to allow for deferring
    constructing the surrogate.

    Simple aspects of the problem problem such as its search space
    are defined immediately, while the surrogate is only defined when [TODO]
    in order to avoid expensive operations like downloading files and fitting
    a model.
    """

    optimization_config: MultiObjectiveOptimizationConfig

    def __init__(
        self,
        maximum_hypervolume: float,
        reference_point: List[float],
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: MultiObjectiveOptimizationConfig,
        num_trials: int,
        outcome_names: List[str],
        observe_noise_stds: Union[bool, Dict[str, bool]] = False,
        noise_stds: Union[float, Dict[str, float]] = 0.0,
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[BenchmarkMetricBase]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            outcome_names=outcome_names,
            observe_noise_stds=observe_noise_stds,
            noise_stds=noise_stds,
            get_surrogate_and_datasets=get_surrogate_and_datasets,
            tracking_metrics=tracking_metrics,
            _runner=_runner,
        )
        self.reference_point = reference_point
        self.maximum_hypervolume = maximum_hypervolume

    @property
    def optimal_value(self) -> float:
        return self.maximum_hypervolume
