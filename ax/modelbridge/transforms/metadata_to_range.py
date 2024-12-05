#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Any, Iterable, Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import (
    PARAMETER_PYTHON_TYPE_MAP,
    ParameterType,
    RangeParameter,
    TParameterType,
)
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

INVERSE_PARAMETER_PYTHON_TYPE_MAP: dict[TParameterType, ParameterType] = {
    v: k for k, v in PARAMETER_PYTHON_TYPE_MAP.items()
}


class MetadataToRange(Transform):
    """
    A transform that converts metadata from observation features into range parameters
    for a search space.

    This transform takes a list of observations and extracts specified metadata keys
    to be used as parameter in the search space. It also updates the search space with
    new Range parameters based on the metadata values.

    TODO[tiao]: update following
    Accepts the following `config` parameters:

    - "keys": A list of strings representing the metadata keys to be extracted and
        used as features.
    - "log_scale": A boolean indicating whether the parameters should be on a
        log scale. Defaults to False.
    - "is_fidelity": A boolean indicating whether the parameters are fidelity
        parameters. Defaults to False.

    Transform is done in-place.
    """

    DEFAULT_LOG_SCALE: bool = False
    DEFAULT_LOGIT_SCALE: bool = False
    DEFAULT_IS_FIDELITY: bool = False

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

        # TODO[tiao]: make config option
        strict = True

        self._parameter_list: list[RangeParameter] = []
        for name in self.parameters:
            parameter_type = None
            lb = ub = None  # de facto bounds
            for obs in observations:
                obsf_metadata = none_throws(obs.features.metadata)
                val = obsf_metadata[name]

                # TODO[tiao]: give user option to explicitly specify parameter type(?)
                # TODO[tiao]: check the inferred type is consistent across all
                # observations; such inconsistencies may actually be impossible
                # by virtue of the validations carried out upstream(?)
                parameter_type = parameter_type or _infer_parameter_type(val)

                lb = min(val, lb) if lb is not None else val
                ub = max(val, ub) if ub is not None else val

            lower = self.parameters[name].get("lower", lb)
            upper = self.parameters[name].get("upper", ub)

            if ub != upper:
                if ub > upper:
                    # TODO[tiao]: necessary to raise the exception here if it will
                    # be caught and raised down the line anyway?
                    raise ValueError(
                        f"Upper bound of `{name}` must be greater than or equal to the "
                        f"highest observed value ({ub})."
                    )
                if strict:
                    raise DataRequiredError(
                        f"No values observed at upper bound {upper}"
                        f" (highest observed: {ub})"
                    )

            if lb != lower:
                if lb < lower:
                    # TODO[tiao]: necessary to raise the exception here if it will
                    # be caught and raised down the line anyway?
                    raise ValueError(
                        f"Lower bound of {name} must be less than or equal to the "
                        f"lowest observed value ({lb})."
                    )
                if strict:
                    raise DataRequiredError(
                        f"No values observed at lower bound {lower}"
                        f" (lowest observed: {lb})"
                    )
            # (additional validation logic such as upper - lower > 0
            # is left to RangeParameter._validate_range_param)

            log_scale = self.parameters[name].get(
                "log_scale", MetadataToRange.DEFAULT_LOG_SCALE
            )
            logit_scale = self.parameters[name].get(
                "logit_scale", MetadataToRange.DEFAULT_LOGIT_SCALE
            )
            digits = self.parameters[name].get("digits")
            is_fidelity = self.parameters[name].get(
                "is_fidelity", MetadataToRange.DEFAULT_IS_FIDELITY
            )

            # TODO[tiao]: necessary to check within bounds?
            target_value = self.parameters[name].get("target_value")

            parameter = RangeParameter(
                name=name,
                parameter_type=none_throws(parameter_type),
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
            search_space.add_parameter(parameter)
        return search_space

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            if not obsf.parameters:
                for p in self._parameter_list:
                    # TODO[tiao]: can we use be p.target_value?
                    # not its original intended use but could be advantageous
                    obsf.parameters[p.name] = p.upper
                continue
            _transfer(
                src=none_throws(obsf.metadata),
                dst=obsf.parameters,
                keys=self.parameters.keys(),
            )
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


def _infer_parameter_type(x: TParameterType) -> ParameterType:
    # search in order of class hierarchy (e.g. bool is a subclass of int)
    # therefore cannot directly use ax.core.parameter.SUPPORTED_PARAMETER_TYPES
    # (unless it is sorted correctly)
    return next(
        INVERSE_PARAMETER_PYTHON_TYPE_MAP[typ]
        for typ in (bool, int, float, str)
        if isinstance(x, typ)
    )


def _transfer(
    src: dict[str, Any],
    dst: dict[str, Any],
    keys: Iterable[str],
) -> None:
    """Transfer items in-place from one dictionary to another."""
    for key in keys:
        dst[key] = src.pop(key)
