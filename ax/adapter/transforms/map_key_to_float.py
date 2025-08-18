#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from math import isnan
from typing import Any, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.metadata_to_float import MetadataToFloat
from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.utils import extract_map_keys_from_opt_config
from ax.generators.types import TConfig
from pandas import Index, MultiIndex
from pyre_extensions import none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class MapKeyToFloat(MetadataToFloat):
    """
    This transform extracts the entry from the metadata field of the observation
    features corresponding to the `parameters` specified in the transform config
    and inserts it into the parameter field. If no parameters are specified in the
    config, the transform will extract all map key names from the optimization config.

    Inheriting from the `MetadataToFloat` transform, this transform also adds a range
    (float) parameter to the search space. Similarly, users can override the default
    behavior by specifying the `config` with `parameters` as the key, where each entry
    maps a metadata key to a dictionary of keyword arguments for the corresponding
    RangeParameter constructor. NOTE: log and logit-scale options are not supported.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        if "parameters" not in config:
            # Extract map keys from the optimization config, if no parameters are
            # specified in the config.
            if adapter is not None and adapter._optimization_config is not None:
                config["parameters"] = {
                    key: {}
                    for key in extract_map_keys_from_opt_config(
                        optimization_config=adapter._optimization_config
                    )
                }
            else:
                warnings.warn(
                    (
                        f"{self.__class__.__name__} is unable to identify `parameters` "
                        "in the transform config or an adapter with an "
                        "optimization config from which the map keys can be inferred; "
                        "this transform will not perform any operations and behave as "
                        "a no-op."
                    ),
                    stacklevel=2,
                )
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

    def _get_values_for_parameter(
        self,
        name: str,
        observations: list[Observation] | None,
        experiment_data: ExperimentData | None,
    ) -> set[float]:
        if experiment_data is not None:
            obs_data = experiment_data.observation_data
            if name not in obs_data.index.names:
                raise ValueError(
                    f"Parameter {name} is not in the index of the observation data."
                )
            return set(
                obs_data.index.unique(level=name).dropna().astype(float).tolist()
            )
        # For Observations, the logic is identical to the parent class.
        return super()._get_values_for_parameter(
            name=name,
            observations=observations,
            experiment_data=experiment_data,
        )

    def _transform_observation_feature(self, obsf: ObservationFeatures) -> None:
        if len(obsf.parameters) == 0:
            obsf.parameters = {p.name: p.upper for p in self._parameter_list}
            return
        if obsf.metadata is None:
            obsf.metadata = {}
        metadata: dict[str, Any] = none_throws(obsf.metadata)
        for p in self._parameter_list:
            if isnan(metadata.get(p.name, float("nan"))):
                metadata[p.name] = p.upper
        super()._transform_observation_feature(obsf)

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Transform experiment data by replacing NaN/null values
        in the corresponding index of observation_data with the upper bound
        of the parameter.

        This reproduces the behavior of `_transform_observation_feature` when
        the parameter is missing in the observation features.

        In contrast with other transforms that extract features from various
        metadata, this transform does not add a new column to arm_data. Instead,
        the parameter values are directly extracted from the index of the
        observation data, as long as the parameter is added to the search space.
        """
        if not self._parameter_list:
            return experiment_data

        observation_data = experiment_data.observation_data

        # Create a mapping of parameter names to their upper bounds for quick lookup
        param_upper_bounds = {p.name: p.upper for p in self._parameter_list}

        # Build new index arrays, filling NaN values where needed.
        new_levels: list[Index] = []
        index_needs_update = False

        for level_name in observation_data.index.names:
            level_values = observation_data.index.get_level_values(level_name)

            if level_name in param_upper_bounds and level_values.isna().any():
                # Replace NaN/null values with the parameter's upper bound.
                filled_values = level_values.fillna(param_upper_bounds[level_name])
                new_levels.append(filled_values)
                index_needs_update = True
            else:
                # Keep original level values.
                new_levels.append(level_values)

        # Update the index only if any levels needed updating.
        if index_needs_update:
            observation_data.index = MultiIndex.from_arrays(
                new_levels, names=observation_data.index.names
            )

        return ExperimentData(
            arm_data=experiment_data.arm_data,
            observation_data=observation_data,
        )
