#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from time import time
from typing import Optional, TYPE_CHECKING

import pandas as pd

from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.timeutils import unixtime_to_pandas_ts
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


logger: Logger = get_logger(__name__)


class TimeAsFeature(Transform):
    """Convert start time and duration into features that can be used for modeling.

    If no end_time is available, the current time is used.

    Duration is normalized to the unit cube.

    Transform is done in-place.

    TODO: revise this when better support for non-tunable features is added.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert observations is not None, "TimeAsFeature requires observations"
        if isinstance(search_space, RobustSearchSpace):
            raise UnsupportedError(
                "TimeAsFeature transform is not supported for RobustSearchSpace."
            )
        self.min_start_time: float = float("inf")
        self.max_start_time: float = float("-inf")
        self.min_duration: float = float("inf")
        self.max_duration: float = float("-inf")
        self.current_time: float = time()
        for obs in observations:
            obsf = obs.features
            if obsf.start_time is None:
                raise ValueError(
                    "Unable to use TimeAsFeature since not all observations have "
                    "start time specified."
                )
            start_time = none_throws(obsf.start_time).timestamp()
            self.min_start_time = min(self.min_start_time, start_time)
            self.max_start_time = max(self.max_start_time, start_time)
            duration = self._get_duration(start_time=start_time, end_time=obsf.end_time)
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            self.duration_range: float = self.max_duration - self.min_duration
            if self.duration_range == 0:
                # no need to case-distinguish during normalization
                self.duration_range = 1.0

    def _get_duration(self, start_time: float, end_time: pd.Timestamp | None) -> float:
        return (
            self.current_time if end_time is None else end_time.timestamp()
        ) - start_time

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            if obsf.start_time is not None:
                start_time = obsf.start_time.timestamp()
                obsf.parameters["start_time"] = start_time
                duration = self._get_duration(
                    start_time=start_time, end_time=obsf.end_time
                )
                # normalize duration to the unit cube
                obsf.parameters["duration"] = (
                    duration - self.min_duration
                ) / self.duration_range
            else:
                # start time can be None for pending arms that generated
                # with a model that did not use the TimeAsFeature transform.
                # In that case, assume the arm is going to be evaluated at the
                # current time, and that the duration is the midpoint of the
                # range.
                obsf.parameters["start_time"] = self.current_time
                obsf.parameters["duration"] = 0.5
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name in ("start_time", "duration"):
            if p_name in search_space.parameters:
                raise ValueError(
                    f"Parameter name {p_name} is reserved when using "
                    "TimeAsFeature transform, but is part of the provided "
                    "search space. Please choose a different name for "
                    "this parameter."
                )
        param = RangeParameter(
            name="start_time",
            parameter_type=ParameterType.FLOAT,
            lower=self.min_start_time,
            upper=self.max_start_time,
        )
        search_space.add_parameter(param)
        param = RangeParameter(
            name="duration",
            parameter_type=ParameterType.FLOAT,
            # duration is normalized to [0,1]
            lower=0.0,
            upper=1.0,
        )
        search_space.add_parameter(param)
        return search_space

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            start_time = obsf.parameters.pop("start_time", None)
            duration = obsf.parameters.pop("duration", None)
            if start_time is not None:
                start_time = assert_is_instance(start_time, float)
                obsf.start_time = unixtime_to_pandas_ts(start_time)
                if duration is not None:
                    duration = assert_is_instance(duration, float)
                    obsf.end_time = unixtime_to_pandas_ts(
                        duration * self.duration_range + self.min_duration + start_time
                    )
        return observation_features
