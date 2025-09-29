#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import cast, Optional, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401

logger: Logger = get_logger(__name__)


class FillMissingParameters(Transform):
    """If a parameter is missing from an arm, fill it with the value from
    the dict given in the config.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,  # Deprecated
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        self._fill_values: TParameterization = {}
        # Read fill_values from deprecated config if provided to maintain backwards
        # compatibility
        if config is not None:
            logger.error(
                "Use of config for FillMissingParameters has been deprecated. "
                "Use search_space.add_parameters instead."
            )
            self._fill_values.update(
                cast(TParameterization, config.get("fill_values", {}))
            )

        # Add backfill values from search space. These will override any values
        # provided in the deprecated config.
        if search_space is not None:
            self._fill_values.update(search_space.backfill_values())

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            fill_params = {
                k: v
                for k, v in self._fill_values.items()
                if k not in obsf.parameters or (obsf.parameters[k] is None)
            }
            obsf.parameters.update(fill_params)
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data.fillna(value=self._fill_values)
        # If any of the fill columns are missing in arm_data, add it.
        missing_columns = set(self._fill_values) - set(arm_data.columns)
        for col in missing_columns:
            arm_data[col] = self._fill_values[col]
        return ExperimentData(
            arm_data=arm_data,
            observation_data=experiment_data.observation_data,
        )
