# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Set

import numpy as np

import pandas as pd
import torch
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from ax.models.torch.botorch_modular.surrogate import Surrogate

from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.serialization import extract_init_args
from ax.utils.common.typeutils import checked_cast
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


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
    ) -> "SurrogateBenchmarkProblem":
        return SurrogateBenchmarkProblem(
            name=name,
            search_space=search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=SurrogateMetric(),
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
        )


class SurrogateMetric(Metric):
    def __init__(self) -> None:
        super().__init__(name="prediction")

    def fetch_trial_data(self, trial: BaseTrial, **kwargs) -> Data:
        prediction = [
            trial.run_metadata["prediction"][name]
            for name, arm in trial.arms_by_name.items()
        ]
        df = pd.DataFrame(
            {
                "arm_name": [name for name, _ in trial.arms_by_name.items()],
                "metric_name": self.name,
                "mean": prediction,
                "sem": np.nan,
                "trial_index": trial.index,
            }
        )

        return Data(df=df)


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

        surrogate.fit(
            datasets=datasets,
            metric_names=["objective"],
            search_space_digest=extract_search_space_digest(
                search_space=search_space, param_names=[*search_space.parameters.keys()]
            ),
        )

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
    def serialize_init_args(cls, runner: Runner) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.
        """
        runner = checked_cast(SurrogateRunner, runner)

        init_args = super().serialize_init_args(runner=runner)

        init_args["datasets"] = [
            (dataset.X().tolist(), dataset.Y().tolist()) for dataset in runner.datasets
        ]

        return init_args

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        init_args = extract_init_args(args=args, class_=cls)

        init_args["datasets"] = [
            SupervisedDataset(X=Tensor(X), Y=Tensor(Y))
            for X, Y in init_args["datasets"]
        ]

        from ax.storage.json_store.decoder import object_from_json

        init_args["surrogate"] = object_from_json(init_args["surrogate"])
        init_args["search_space"] = object_from_json(init_args["search_space"])

        return init_args
