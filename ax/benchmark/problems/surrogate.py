# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import torch
from ax.benchmark.benchmark_problem import (
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch_modular.surrogate import Surrogate

from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.result import Err, Ok
from botorch.utils.datasets import SupervisedDataset


class SurrogateBenchmarkProblem(SingleObjectiveBenchmarkProblem):
    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, SurrogateBenchmarkProblem):
            return False

        # Checking the whole datasets' equality here would be too expensive to be
        # worth it; just check names instead
        return self.name == other.name

    @classmethod
    def from_surrogate(
        cls,
        name: str,
        search_space: SearchSpace,
        surrogate: Surrogate,
        datasets: List[SupervisedDataset],
        optimal_value: float,
        optimization_config: OptimizationConfig,
        num_trials: int,
        metric_names: List[str],
        infer_noise: bool = True,
    ) -> "SurrogateBenchmarkProblem":
        return SurrogateBenchmarkProblem(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=SurrogateRunner(
                name=name,
                surrogate=surrogate,
                datasets=datasets,
                search_space=search_space,
                metric_names=metric_names,
            ),
            optimal_value=optimal_value,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


class MOOSurrogateBenchmarkProblem(MultiObjectiveBenchmarkProblem):
    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, MOOSurrogateBenchmarkProblem):
            return False

        # Checking the whole datasets' equality here would be too expensive to be
        # worth it; just check names instead
        return self.name == other.name

    @classmethod
    def from_surrogate(
        cls,
        name: str,
        search_space: SearchSpace,
        surrogate: Surrogate,
        datasets: List[SupervisedDataset],
        optimization_config: MultiObjectiveOptimizationConfig,
        maximum_hypervolume: float,
        reference_point: List[float],
        num_trials: int,
        metric_names: List[str],
        infer_noise: bool = True,
    ) -> "MOOSurrogateBenchmarkProblem":
        if not all(
            isinstance(m, SurrogateMetric)
            for m in optimization_config.objective.metrics
        ):
            raise UnsupportedError(
                "MOOSurrogateBenchmarkProblem only supports SurrogateMetrics."
            )

        return MOOSurrogateBenchmarkProblem(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=SurrogateRunner(
                name=name,
                surrogate=surrogate,
                datasets=datasets,
                search_space=search_space,
                metric_names=metric_names,
            ),
            maximum_hypervolume=maximum_hypervolume,
            reference_point=reference_point,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


class SurrogateMetric(Metric):
    def __init__(
        self, name: str, lower_is_better: bool, infer_noise: bool = True
    ) -> None:
        super().__init__(name=name, lower_is_better=lower_is_better)
        self.infer_noise = infer_noise

    # pyre-fixme[2]: Parameter must be annotated.
    def fetch_trial_data(self, trial: BaseTrial, **kwargs) -> MetricFetchResult:
        try:
            prediction = [
                trial.run_metadata[self.name][name]
                for name, arm in trial.arms_by_name.items()
            ]
            df = pd.DataFrame(
                {
                    "arm_name": [name for name, _ in trial.arms_by_name.items()],
                    "metric_name": self.name,
                    "mean": prediction,
                    "sem": None if self.infer_noise else 0,
                    "trial_index": trial.index,
                }
            )

            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(
                    message=f"Failed to predict for trial {trial}", exception=e
                )
            )


class SurrogateRunner(Runner):
    def __init__(
        self,
        name: str,
        surrogate: Surrogate,
        datasets: List[SupervisedDataset],
        search_space: SearchSpace,
        metric_names: List[str],
    ) -> None:
        self.name = name
        self.surrogate = surrogate
        self.metric_names = metric_names
        self.datasets = datasets
        self.search_space = search_space

        self.results: Dict[int, float] = {}
        self.statuses: Dict[int, TrialStatus] = {}

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        self.statuses[trial.index] = TrialStatus.COMPLETED
        preds = {  # Cache predictions for each arm
            arm.name: self.surrogate.predict(
                X=torch.tensor([*arm.parameters.values()]).reshape(
                    [1, len(arm.parameters)]
                )
            )[0].squeeze(0)
            for arm in trial.arms
        }
        return {
            metric_name: {arm_name: pred[i] for arm_name, pred in preds.items()}
            for i, metric_name in enumerate(self.metric_names)
        }

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.

        WARNING: Because of issues with consistently saving and loading BoTorch and
        GPyTorch modules the SurrogateRunner cannot be serialized at this time. At load
        time the runner will be replaced with a SyntheticRunner.
        """
        return {}

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        return {}
