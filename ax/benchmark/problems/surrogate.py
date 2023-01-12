# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import torch
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
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
        minimize: bool,
        optimal_value: float,
        num_trials: int,
        infer_noise: bool = True,
    ) -> "SurrogateBenchmarkProblem":
        return SurrogateBenchmarkProblem(
            name=name,
            search_space=search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=SurrogateMetric(infer_noise=infer_noise),
                    minimize=minimize,
                )
            ),
            runner=SurrogateRunner(
                name=name,
                surrogate=surrogate,
                datasets=datasets,
                search_space=search_space,
            ),
            optimal_value=optimal_value,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


class SurrogateMetric(Metric):
    def __init__(self, infer_noise: bool = True) -> None:
        super().__init__(name="prediction")
        self.infer_noise = infer_noise

    # pyre-fixme[2]: Parameter must be annotated.
    def fetch_trial_data(self, trial: BaseTrial, **kwargs) -> MetricFetchResult:
        try:
            prediction = [
                trial.run_metadata["prediction"][name]
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
    ) -> None:
        self.name = name
        self.surrogate = surrogate
        self.datasets = datasets
        self.search_space = search_space

        self.results: Dict[int, float] = {}
        self.statuses: Dict[int, TrialStatus] = {}

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        self.statuses[trial.index] = TrialStatus.COMPLETED
        return {
            "prediction": {
                arm.name: self.surrogate.predict(
                    X=torch.tensor([*arm.parameters.values()]).reshape(
                        [1, len(arm.parameters)]
                    )
                )[0].item()
                for arm in trial.arms
            }
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
