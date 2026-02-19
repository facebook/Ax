#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterable
from logging import Logger
from typing import Any, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.core.observation import ObservationFeatures
from ax.core.parameter import Parameter
from ax.core.search_space import SearchSpace
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


logger: Logger = get_logger(__name__)


class MetadataToParameterMixin:
    """
    This Mixin has utilities for converting metadata from observation features
    into a parameter.

    Transform is done in-place.
    """

    _parameter_list: list[Parameter] = []
    parameters: dict[str, dict[str, Any]] = {}

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for parameter in self._parameter_list:
            search_space.add_parameter(parameter.clone())

        return search_space

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            self._transform_observation_feature(obsf)
        return observation_features

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            obsf.metadata = obsf.metadata or {}
            _transfer(
                src=obsf.parameters,
                dst=obsf.metadata,
                keys=[p.name for p in self._parameter_list],
                target_values=[p.target_value for p in self._parameter_list],
            )
        return observation_features

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        _transfer(
            src=obsf.metadata or {},
            dst=obsf.parameters,
            keys=[p.name for p in self._parameter_list],
            target_values=[p.target_value for p in self._parameter_list],
        )

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        for name in self.parameters:
            arm_data[name] = [md.get(name) for md in arm_data["metadata"]]
        return ExperimentData(
            arm_data=arm_data,
            observation_data=experiment_data.observation_data,
        )


def _transfer(
    src: dict[str, Any],
    dst: dict[str, Any],
    keys: Iterable[str],
    target_values: list[Any],
) -> None:
    """Transfer items in-place from one dictionary to another.

    If a key is not present in the source dictionary, the target value is used.
    """
    for key, target_value in zip(keys, target_values):
        dst[key] = src.pop(key, target_value)
