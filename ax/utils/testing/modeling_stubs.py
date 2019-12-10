#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Type

import numpy as np
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_search_space,
    get_search_space_for_value,
)


logger = get_logger("ae_experiment")


# Observations


def get_observation_features() -> ObservationFeatures:
    return ObservationFeatures(
        parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
    )


def get_observation() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def get_observation1() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def get_observation_status_quo0() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"w": 0.85, "x": 1, "y": "baz", "z": False},
            trial_index=np.int64(0),
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="0_0",
    )


def get_observation_status_quo1() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"w": 0.85, "x": 1, "y": "baz", "z": False},
            trial_index=np.int64(1),
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="0_0",
    )


def get_observation1trans() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 9.0, "y": 10.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([9.0, 25.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def get_observation2() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 3.0, "y": 2.0}, trial_index=np.int64(1)
        ),
        data=ObservationData(
            means=np.array([2.0, 1.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def get_observation2trans() -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 16.0, "y": 2.0}, trial_index=np.int64(1)
        ),
        data=ObservationData(
            means=np.array([9.0, 4.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


# Modeling layer


def get_generation_strategy(with_experiment: bool = False) -> GenerationStrategy:
    gs = choose_generation_strategy(search_space=get_search_space())
    if with_experiment:
        gs._experiment = get_experiment()
    return gs


def get_transform_type() -> Type[Transform]:
    return IntToFloat


def get_experiment_for_value() -> Experiment:
    return Experiment(get_search_space_for_value(), "test")


class transform_1(Transform):
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        new_ss = search_space.clone()
        new_ss.parameters["x"]._value += 1.0  # pyre-ignore[16]: testing hack.
        return new_ss

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[ModelBridge],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        return (  # pyre-ignore[7]: pyre is right, this is a hack for testing.
            optimization_config + 1
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            if "x" in obsf.parameters:
                obsf.parameters["x"] = (
                    not_none(obsf.parameters["x"]) + 1  # pyre-ignore[6]
                )
        return observation_features

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means += 1
        return observation_data

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            obsf.parameters["x"] = obsf.parameters["x"] - 1  # pyre-ignore
        return observation_features

    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means -= 1
        return observation_data


class transform_2(Transform):
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        new_ss = search_space.clone()
        new_ss.parameters["x"]._value *= 2.0  # pyre-ignore[16]: testing hack.
        return new_ss

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[ModelBridge],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        return (
            optimization_config ** 2
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            if "x" in obsf.parameters:
                obsf.parameters["x"] = obsf.parameters["x"] ** 2  # pyre-ignore
        return observation_features

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means = obsd.means ** 2
        return observation_data

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            obsf.parameters["x"] = np.sqrt(obsf.parameters["x"])
        return observation_features

    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means = np.sqrt(obsd.means)
        return observation_data
