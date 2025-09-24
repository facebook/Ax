#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.generators.types import TConfig
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class FillMissingParameters(Transform):
    """If a parameter is missing from an arm, fill it with the value from
    the dict given in the config.

    Config supports two options.
        fill_values: a dict of {parameter_name: value} to fill in for missing
            parameters. Required.
        fill_None: a boolean indicating whether to fill in None values. Default
            is True. If False, parameters specified as None will remain None,
            and only parameters absent altogether will be filled.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        config = config or {}
        self.fill_values: TParameterization | None = config.get(  # pyre-ignore[8]
            "fill_values", None
        )
        self.fill_None: bool = assert_is_instance(config.get("fill_None", True), bool)

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        if self.fill_values is None:
            return observation_features
        for obsf in observation_features:
            fill_params = {
                k: v
                for k, v in none_throws(self.fill_values).items()
                if k not in obsf.parameters
                or (obsf.parameters[k] is None and self.fill_None)
            }
            obsf.parameters.update(fill_params)
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        if self.fill_values is None:
            return experiment_data
        if self.fill_None is False:
            # This shouldn't be relevant in regular usage. We add both
            # FillMissingParameters and Cast as default transfroms in
            # Adapter. Cast will drop parameterizations with missing / None
            # values, so not filling None will just lead to it being dropped.
            # The exception is added here for completeness.
            raise UnsupportedError(
                "Transforming `ExperimentData` is not supported for "
                "FillMissingParameters with fill_None=False. "
                "We cannot distinguish between parameters that are missing "
                "and those that are None in `ExperimentData`. "
            )
        arm_data = experiment_data.arm_data.fillna(value=self.fill_values)
        # If any of the fill columns are missing in arm_data, add it.
        missing_columns = set(none_throws(self.fill_values)) - set(arm_data.columns)
        for col in missing_columns:
            arm_data[col] = none_throws(self.fill_values)[col]
        return ExperimentData(
            arm_data=arm_data,
            observation_data=experiment_data.observation_data,
        )
