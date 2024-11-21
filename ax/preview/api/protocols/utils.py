# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable, Mapping

import pandas as pd

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.exceptions.storage import JSONEncodeError
from ax.preview.api.types import TParameterization
from ax.utils.common.result import Err, Ok
from pyre_extensions import assert_is_instance, none_throws, override


class _APIMetric(MapMetric, ABC):
    """
    This Metric provides implmementations for the essential MapMetric methods expected
    by the Ax Scheduler, shimming between our IMetric and MapMetric interfaces.

    Users should never instantiate or subclass this class directly.

    Ideally we will be able to remove this class in the future, once we have stablized
    structure of ax.core.Metric to be more in line with our long term vision for Ax.
    """

    map_key_info: MapKeyInfo[float] = MapKeyInfo(key="progression", default_value=0.0)

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @abstractmethod
    def fetch(
        self,
        trial_index: int,
        trial_metadata: Mapping[str, Any],
    ) -> tuple[int, float | tuple[float, float]]: ...

    @override
    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        """
        Fetch data for a single trial for this metric by calling the user-implemented
        fetch method.
        """
        trial = assert_is_instance(trial, Trial)

        try:
            progression, outcome = self.fetch(
                trial_index=trial.index, trial_metadata=trial.run_metadata
            )

            if isinstance(outcome, float):
                mean = outcome
                sem = float("nan")
            else:
                mean, sem = outcome

            record = {
                "trial_index": trial.index,
                "arm_name": none_throws(trial.arm).name,
                "metric_name": self.name,
                self.map_key_info.key: progression,
                "mean": mean,
                "sem": sem,
            }
            return Ok(
                value=MapData(
                    df=pd.DataFrame.from_records([record]),
                    map_key_infos=[self.map_key_info],
                )
            )
        except Exception as e:
            return Err(
                value=MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


class _APIRunner(Runner, ABC):
    """
    This Runner provides implementations for the essential Runner methods expected
    by the Ax Scheduler, shimming between our IRunner and Runner interfaces.

    Users should never instantiate or subclass this class directly.

    Ideally we will be able to remove this class in the future, once we have stablized
    structure of ax.core.Runner to be more in line with our long term vision for Ax.
    """

    @abstractmethod
    def run_trial(
        self, trial_index: int, parameterization: TParameterization
    ) -> dict[str, Any]: ...

    @abstractmethod
    def poll_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> TrialStatus: ...

    @abstractmethod
    def stop_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    @override
    def run(self, trial: BaseTrial) -> dict[str, Any]:
        """
        Runs a trial by calling the user-implemented run_trial method, returning
        appropriate metadata.
        """
        metadata = self.run_trial(
            trial_index=trial.index,
            # pyre-ignore[6] Arms in core Ax may have None in their parameters
            parameterization=none_throws(
                assert_is_instance(trial, Trial).arm
            ).parameters,
        )

        # Runtime validate metadata is JSON serializable to avoid issues when
        # serializing the Trial.
        try:
            json.dumps(metadata)
            return metadata
        except TypeError:
            raise JSONEncodeError(
                f"Metadata must be JSON serializable, received {metadata}"
            )

    @override
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        """
        Polls the status of trials by calling the user-implemented poll_trial method.
        """
        res = defaultdict(set)
        for trial in trials:
            status = self.poll_trial(
                trial_index=trial.index, trial_metadata=trial.run_metadata
            )

            res[status].add(trial.index)

        return res

    @override
    def stop(self, trial: BaseTrial, reason: str | None = None) -> dict[str, Any]:
        """Stops a trial by calling the user-implemented stop_trial method"""
        return self.stop_trial(
            trial_index=trial.index, trial_metadata=trial.run_metadata
        )
