#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.core.arm import Arm
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.observation_utils import separate_observations
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.generators.types import TConfig


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class Transform:
    """Defines the API for a transform that is applied to search_space,
    observation_features, observation_data, and optimization_config.

    Transforms are used to adapt the search space and data into the types
    and structures expected by the model. When Transforms are used (for
    instance, in Adapter), it is always assumed that they may potentially
    mutate the transformed object in-place.

    Forward transforms are defined for all four of those quantities. Reverse
    transforms are defined for observation_data and observation.

    The forward transform for observation features must accept a partial
    observation with not all features recorded.

    Forward and reverse transforms for observation data accept a list of
    observation features as an input, but they will not be mutated.

    The forward transform for optimization config accepts the adapter and
    fixed features as inputs, but they will not be mutated.

    This class provides an identify transform.
    """

    config: TConfig
    adapter: adapter_module.base.Adapter | None
    # Set this to True if the transform does not need to transform ExperimentData.
    # If True, base method will return the input unmodified. Otherwise, it'll error out.
    no_op_for_experiment_data: bool = False
    # Set this to True if the transform requires non-empty data for initialization.
    requires_data_for_initialization: bool = False

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Do any initial computations for preparing the transform.

        Args:
            search_space: The search space of the experiment.
            experiment_data: A container for the parameterizations, metadata and
                observations for the trials in the experiment.
                Constructed using ``extract_experiment_data``.
            adapter: Adapter for referencing experiment, status quo, etc.
            config: A dictionary of options specific to each transform.
        """
        if self.requires_data_for_initialization and (
            experiment_data is None or experiment_data.observation_data.empty
        ):
            raise DataRequiredError(
                f"`{self.__class__.__name__}` transform requires non-empty data."
            )
        if config is None:
            config = {}
        self.config = deepcopy(config)
        self.adapter = adapter

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Transform search space.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            search_space: The search space

        Returns: transformed search space.
        """
        return search_space

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        """Transform optimization config.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            optimization_config: The optimization config

        Returns: transformed optimization config.
        """
        if optimization_config.pruning_target_parameterization is not None:
            pruning_target_params = (
                optimization_config.pruning_target_parameterization.parameters
            )
            optimization_config.pruning_target_parameterization = Arm(
                parameters=self.transform_observation_features(
                    observation_features=[
                        ObservationFeatures(parameters=pruning_target_params)
                    ]
                )[0].parameters
            )
        return optimization_config

    def transform_observations(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """Transform observations.

        Typically done in place. By default, the effort is split into separate
        transformations of the features and the data.

        Args:
            observations: Observations.

        Returns: transformed observations.
        """
        obs_feats, obs_data = separate_observations(observations=observations)
        obs_feats = self.transform_observation_features(observation_features=obs_feats)
        obs_data = self._transform_observation_data(observation_data=obs_data)
        trans_obs = [
            Observation(features=obs_feats[i], data=obs_data[i], arm_name=obs.arm_name)
            for i, obs in enumerate(observations)
        ]
        return trans_obs

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Transform observation features.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            observation_features: Observation features

        Returns: transformed observation features
        """
        return observation_features

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Transform observation features.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        This method does not need to be implemented if transform_observations
        is overridden.

        Args:
            observation_data: Observation data

        Returns: transformed observation data
        """
        return observation_data

    def untransform_observations(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """Untransform observations.

        Typically done in place. By default, the effort is split into separate
        backwards transformations of the features and the data.

        Args:
            observations: Observations.

        Returns: untransformed observations.
        """
        obs_feats, obs_data = separate_observations(observations=observations)
        obs_feats = self.untransform_observation_features(
            observation_features=obs_feats
        )
        obs_data = self._untransform_observation_data(observation_data=obs_data)
        untrans_obs = [
            Observation(features=obs_feats[i], data=obs_data[i], arm_name=obs.arm_name)
            for i, obs in enumerate(observations)
        ]
        return untrans_obs

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Untransform observation features.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            observation_features: Observation features in the transformed space

        Returns: observation features in the original space
        """
        return observation_features

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Untransform observation data.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        This method does not need to be implemented if untransform_observations
        is overridden.

        Args:
            observation_data: Observation data, in transformed space

        Returns: observation data in original space.
        """
        return observation_data

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        """Untransform outcome constraints.

        If outcome constraints are modified in transform_optimization_config,
        this method should reverse the portion of that transformation that was
        applied to the outcome constraints.
        """
        return outcome_constraints

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Transform ``ExperimentData``.

        This is typically done in-place. This class implements the identity
        transform, if ``self.no_op_for_experiment_data is True``, otherwise
        it errors out to protect against accidental use of transforms that
        do not support ``ExperimentData``.

        Args:
            experiment_data: The ``ExperimentData`` to transform.

        Returns: The transformed experiment data.
        """
        if self.no_op_for_experiment_data or self.__class__ == Transform:
            return experiment_data
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement "
                "`transform_experiment_data`."
            )
