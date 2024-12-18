#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Any, Iterable, Optional, SupportsFloat, TYPE_CHECKING

from ax.core import ParameterType

from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


logger: Logger = get_logger(__name__)


class MetadataToFloat(Transform):
    """
    This transform converts metadata from observation features into range (float)
    parameters for a search space.

    It allows the user to specify the `config` with `parameters` as the key, where
    each entry maps a metadata key to a dictionary of keyword arguments for the
    corresponding RangeParameter constructor.

    Transform is done in-place.
    """

    DEFAULT_LOG_SCALE: bool = False
    DEFAULT_LOGIT_SCALE: bool = False
    DEFAULT_IS_FIDELITY: bool = False
    ENFORCE_BOUNDS: bool = False

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        if observations is None or not observations:
            raise DataRequiredError(
                "`MetadataToRange` transform requires non-empty data."
            )
        config = config or {}
        self.parameters: dict[str, dict[str, Any]] = assert_is_instance(
            config.get("parameters", {}), dict
        )

        self._parameter_list: list[RangeParameter] = []
        for name in self.parameters:
            values: list[float] = []
            for obs in observations:
                obsf_metadata = none_throws(obs.features.metadata)
                value = float(assert_is_instance(obsf_metadata[name], SupportsFloat))
                values.append(value)

            lower: float = self.parameters[name].get("lower", min(values))
            upper: float = self.parameters[name].get("upper", max(values))

            log_scale = self.parameters[name].get("log_scale", self.DEFAULT_LOG_SCALE)
            logit_scale = self.parameters[name].get(
                "logit_scale", self.DEFAULT_LOGIT_SCALE
            )
            digits = self.parameters[name].get("digits")
            is_fidelity = self.parameters[name].get(
                "is_fidelity", self.DEFAULT_IS_FIDELITY
            )

            target_value = self.parameters[name].get("target_value")

            parameter = RangeParameter(
                name=name,
                parameter_type=ParameterType.FLOAT,
                lower=lower,
                upper=upper,
                log_scale=log_scale,
                logit_scale=logit_scale,
                digits=digits,
                is_fidelity=is_fidelity,
                target_value=target_value,
            )
            self._parameter_list.append(parameter)

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
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
                keys=self.parameters.keys(),
            )
        return observation_features

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        _transfer(
            src=none_throws(obsf.metadata),
            dst=obsf.parameters,
            keys=self.parameters.keys(),
        )


def _transfer(
    src: dict[str, Any],
    dst: dict[str, Any],
    keys: Iterable[str],
) -> None:
    """Transfer items in-place from one dictionary to another."""
    for key in keys:
        dst[key] = src.pop(key)
