#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    separate_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.models.types import TConfig


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class Transform:
    """Defines the API for a transform that is applied to search_space,
    observation_features, observation_data, and optimization_config.

    Transforms are used to adapt the search space and data into the types
    and structures expected by the model. When Transforms are used (for
    instance, in ModelBridge), it is always assumed that they may potentially
    mutate the transformed object in-place.

    Forward transforms are defined for all four of those quantities. Reverse
    transforms are defined for observation_data and observation.

    The forward transform for observation features must accept a partial
    observation with not all features recorded.

    Forward and reverse transforms for observation data accept a list of
    observation features as an input, but they will not be mutated.

    The forward transform for optimization config accepts the modelbridge and
    fixed features as inputs, but they will not be mutated.

    This class provides an identify transform.
    """

    config: TConfig
    modelbridge: Optional[modelbridge_module.base.ModelBridge]

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        """Do any initial computations for preparing the transform.

        This takes in search space and observations, but they are not modified.

        Args:
            search_space: The search space
            observations: Observations
            modelbridge: ModelBridge for referencing experiment, status quo, etc...
            config: A dictionary of options specific to each transform
        """
        if config is None:
            config = {}
        self.config = config
        self.modelbridge = modelbridge

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Transform search space.

        The transforms are typically done in-place. This calls two private methods,
        `_transform_search_space`, which transforms the core search space attributes,
        and `_transform_parameter_distributions`, which transforms the distributions
        when using a `RobustSearchSpace`.

        Args:
            search_space: The search space

        Returns: transformed search space.
        """
        self._transform_parameter_distributions(search_space=search_space)
        return self._transform_search_space(search_space=search_space)

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
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
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        """Transform optimization config.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            optimization_config: The optimization config

        Returns: transformed optimization config.
        """
        return optimization_config

    def transform_observations(
        self, observations: List[Observation]
    ) -> List[Observation]:
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
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
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
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
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
        self, observations: List[Observation]
    ) -> List[Observation]:
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
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
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
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
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
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        """Untransform outcome constraints.

        If outcome constraints are modified in transform_optimization_config,
        this method should reverse the portion of that transformation that was
        applied to the outcome constraints.
        """
        return outcome_constraints

    def _transform_parameter_distributions(self, search_space: SearchSpace) -> None:
        """Transform the parameter distributions of the given search space, in-place.

        This method should be called in transform_search_space before parameters
        are transformed.

        The base implementation is a no-op for most transforms, except for those
        that have a `transform_parameters` attribute, in which case this will
        raise an `UnsupportedError` if a parameter with an associated distribution
        is being transformed.
        """
        if isinstance(search_space, RobustSearchSpace) and hasattr(
            self, "transform_parameters"
        ):
            # pyre-ignore Undefined attribute [16]
            for p_name in self.transform_parameters:
                if p_name in search_space._distributional_parameters:
                    raise UnsupportedError(
                        f"{self.__class__.__name__} transform is not supported for "
                        "parameters with an associated distribution. Consider updating "
                        "the transform config."
                    )
