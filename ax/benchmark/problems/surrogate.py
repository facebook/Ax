# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import torch
from ax.benchmark.benchmark_problem import BenchmarkProblemBase
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.parameter import RangeParameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.log import Log
from ax.models.torch.botorch_modular.surrogate import Surrogate

from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.result import Err, Ok
from ax.utils.common.typeutils import not_none
from botorch.utils.datasets import SupervisedDataset


class SurrogateBenchmarkProblemBase(Base, BenchmarkProblemBase):
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
        infer_noise: bool,
        metric_names: List[str],
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[Metric]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        if get_surrogate_and_datasets is None and _runner is None:
            raise ValueError(
                "Either `get_surrogate_and_datasets` or `_runner` required."
            )
        self.name = name
        self.search_space = search_space
        self.optimization_config = optimization_config
        self.num_trials = num_trials
        self.infer_noise = infer_noise
        self.metric_names = metric_names
        self.get_surrogate_and_datasets = get_surrogate_and_datasets
        self.tracking_metrics: List[Metric] = (
            [] if tracking_metrics is None else tracking_metrics
        )
        self._runner = _runner

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
            metric_names=self.metric_names,
        )

    @property
    def runner(self) -> Runner:
        if self._runner is None:
            self.set_runner()
        return not_none(self._runner)


class SOOSurrogateBenchmarkProblem(SurrogateBenchmarkProblemBase):
    """
    Has the same attributes/properties as a `SingleObjectiveBenchmarkProblem`,
    but allows for constructing from a surrogate.
    """

    def __init__(
        self,
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        num_trials: int,
        infer_noise: bool,
        optimal_value: float,
        metric_names: List[str],
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[Metric]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            infer_noise=infer_noise,
            metric_names=metric_names,
            get_surrogate_and_datasets=get_surrogate_and_datasets,
            tracking_metrics=tracking_metrics,
            _runner=_runner,
        )
        self.optimization_config = optimization_config
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
        *,
        name: str,
        search_space: SearchSpace,
        optimization_config: MultiObjectiveOptimizationConfig,
        num_trials: int,
        infer_noise: bool,
        maximum_hypervolume: float,
        reference_point: List[float],
        metric_names: List[str],
        get_surrogate_and_datasets: Optional[
            Callable[[], Tuple[Surrogate, List[SupervisedDataset]]]
        ] = None,
        tracking_metrics: Optional[List[Metric]] = None,
        _runner: Optional[Runner] = None,
    ) -> None:
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            num_trials=num_trials,
            infer_noise=infer_noise,
            metric_names=metric_names,
            get_surrogate_and_datasets=get_surrogate_and_datasets,
            tracking_metrics=tracking_metrics,
            _runner=_runner,
        )
        self.reference_point = reference_point
        self.maximum_hypervolume = maximum_hypervolume


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

        # If there are log scale parameters, these need to be transformed.
        if any(
            isinstance(p, RangeParameter) and p.log_scale
            for p in search_space.parameters.values()
        ):
            int_to_float_tf = IntToFloat(search_space=search_space)
            log_tf = Log(
                search_space=int_to_float_tf.transform_search_space(
                    search_space.clone()
                )
            )
            self.transforms: Optional[Tuple[IntToFloat, Log]] = (
                int_to_float_tf,
                log_tf,
            )
        else:
            self.transforms = None

    def _get_transformed_parameters(
        self, parameters: TParameterization
    ) -> TParameterization:
        if self.transforms is None:
            return parameters

        obs_ft = ObservationFeatures(parameters=parameters)
        for t in not_none(self.transforms):
            obs_ft = t.transform_observation_features([obs_ft])[0]
        return obs_ft.parameters

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        self.statuses[trial.index] = TrialStatus.COMPLETED
        preds = {  # Cache predictions for each arm
            arm.name: self.surrogate.predict(
                X=torch.tensor(
                    [*self._get_transformed_parameters(arm.parameters).values()]
                ).reshape([1, len(arm.parameters)])
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
