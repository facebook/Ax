#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from math import isnan
from typing import Any, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.metadata_to_parameter import MetadataToParameterMixin
from ax.core.map_data import MAP_KEY
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from pandas import Index, MultiIndex
from pyre_extensions import none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


logger: Logger = get_logger(__name__)


class MapKeyToFloat(MetadataToParameterMixin, Transform):
    """
    This transform is used to extract the progression and convert it into a feature for
    for modeling. For observation features, it is extracted from the metadata field and
    inserted into the parameter field. For ``ExperimentData``, not much is done since
    the corresponding feature is extracted directly from the index of
    ``observation_data`` when extracting the datasets in ``TorchAdapter``.
    A parameter corresponding to the map key is added to the search space, to ensure
    that all features have a corresponding parameter in the search space.

    The user can optionally specify ``config`` with ``parameters`` as the key, where
    the entry maps the MAP_KEY (step) to a dictionary of keyword arguments for the
    corresponding ``RangeParameter`` constructor.
    NOTE: log and logit-scale options are not supported for the ``RangeParameter``.

    Transform is done in-place.
    """

    requires_data_for_initialization: bool = True

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
        # pyre-fixme[9]: Incompatible variable type [9]: parameters is declared
        # to have type `Dict[str, Dict[str, typing.Any]]` but is used as type
        # `Union[None, Dict[int, typing.Any], Dict[str, typing.Any], List[int],
        # List[str], OptimizationConfig, WinsorizationConfig,
        # AcquisitionFunction, float, int, str]`.
        parameters: dict[str, dict[str, Any]] = self.config.get("parameters", {})
        is_map_data = (
            # Note: experiment_data can't be None because
            # `requires_data_for_initialization` is True; if it is None, there
            # will be an error in super().__init__
            experiment_data is not None
            and MAP_KEY in experiment_data.observation_data.index.names
        )

        # Check if config contains unsupported config options.
        if set(self.config.keys()).difference({"parameters"}):
            raise UserInputError(
                "MapKeyToFloat only supports the `parameters` key in the config. "
                f"Got {self.config=}."
            )
        # Check if any disallowed parameters were provided.
        # The only parameter allowed is "step" (MAP_KEY)
        received_parameters = set(parameters)
        if is_map_data:
            needed_parameters = {MAP_KEY} if is_map_data else set()
            disallowed_parameters = received_parameters - needed_parameters
            if len(disallowed_parameters) > 0:
                raise ValueError(
                    "The only allowed key in `config['parameters']` is "
                    f"{MAP_KEY}. Got {disallowed_parameters}."
                )
        elif len(parameters) > 0:
            raise ValueError(
                "No parameters may be provided to MapKeyToFloat with non-map "
                f"data. Got {received_parameters}."
            )

        # Add MAP_KEY to the parameters if it is needed and not provided.
        if is_map_data and MAP_KEY not in parameters:
            parameters = {MAP_KEY: {}}

        self.parameters: dict[str, dict[str, Any]] = parameters
        self._parameter_list: list[RangeParameter] = []
        # Construct the parameter if needed.
        if is_map_data:
            values: set[float] = self._get_map_key_values(
                experiment_data=none_throws(experiment_data)
            )

            if len(values) == 0:
                logger.debug(
                    f"Did not encounter any non-NaN values for "
                    f"'{MAP_KEY=}'. Not adding to parameters."
                )
                return
            if len(values) == 1:
                logger.debug(
                    f"Encountered only a single unique value {values.pop()} for "
                    f"'{MAP_KEY=}'. Not adding to parameters."
                )
                return

            p_config = self.parameters[MAP_KEY]
            self._parameter_list.append(
                RangeParameter(
                    name=MAP_KEY,
                    parameter_type=ParameterType.FLOAT,
                    lower=p_config.get("lower", min(values)),
                    upper=p_config.get("upper", max(values)),
                    digits=p_config.get("digits", None),
                    is_fidelity=p_config.get("is_fidelity", False),
                    target_value=p_config.get("target_value", None),
                )
            )

    def _get_map_key_values(
        self,
        experiment_data: ExperimentData,
    ) -> set[float]:
        obs_data = experiment_data.observation_data
        if MAP_KEY not in obs_data.index.names:
            raise ValueError(f"{MAP_KEY=} is not in the index of the observation data.")
        return set(obs_data.index.unique(level=MAP_KEY).dropna().astype(float).tolist())

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
