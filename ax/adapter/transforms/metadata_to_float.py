#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Any, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.metadata_to_parameter import MetadataToParameterMixin
from ax.core import ParameterType
from ax.core.parameter import RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


logger: Logger = get_logger(__name__)


class MetadataToFloat(MetadataToParameterMixin, Transform):
    """
    This transform converts metadata from observation features into range (float)
    parameters for a search space.

    It allows the user to specify the `config` with `parameters` as the key, where
    each entry maps a metadata key to a dictionary of keyword arguments for the
    corresponding RangeParameter constructor. NOTE: log and logit-scale options
    are not supported.

    Transform is done in-place.
    """

    requires_data_for_initialization: bool = True

    DEFAULT_IS_FIDELITY: bool = False

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        Transform.__init__(
            self,
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        config = config or {}
        self.parameters: dict[str, dict[str, Any]] = assert_is_instance(
            config.get("parameters", {}), dict
        )

        self._parameter_list: list[RangeParameter] = []
        for name in self.parameters:
            values: set[float] = self._get_values_for_parameter(
                name=name, experiment_data=none_throws(experiment_data)
            )

            if len(values) == 0:
                logger.debug(
                    f"Did not encounter any non-NaN values for "
                    f"metadata key '{name}'. Not adding to parameters."
                )
                continue
            if len(values) == 1:
                (value,) = values
                logger.debug(
                    f"Encountered only a single unique value {value:.1f} in "
                    f"metadata key '{name}'. Not adding to parameters."
                )
                continue

            lower: float = self.parameters[name].get("lower", min(values))
            upper: float = self.parameters[name].get("upper", max(values))

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
                digits=digits,
                is_fidelity=is_fidelity,
                target_value=target_value,
            )
            self._parameter_list.append(parameter)

    def _get_values_for_parameter(
        self,
        name: str,
        experiment_data: ExperimentData,
    ) -> set[float]:
        all_metadata = experiment_data.arm_data["metadata"]
        return all_metadata.str.get(name).dropna().astype(float).tolist()
