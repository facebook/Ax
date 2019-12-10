#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig


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

    def __init__(
        self,
        search_space: Optional[SearchSpace],
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        """Do any initial computations for preparing the transform.

        This takes in search space and observations, but they are not modified.

        Args:
            search_space: The search space
            observation_features: Observation features
            observation_data: Observation data
            config: A dictionary of options specific to each transform
        """
        if config is None:
            config = {}
        self.config = config

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
        modelbridge: Optional["modelbridge_module.base.ModelBridge"],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        """Transform optimization config.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            optimization_config: The optimization config

        Returns: transformed optimization config.
        """
        return optimization_config

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

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Transform observation features.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        This takes in observation_features, so that data transforms can be
        conditional on features, but observation_features are notmutated.

        Args:
            observation_data: Observation data
            observation_features: Corresponding observation features

        Returns: transformed observation data
        """
        return observation_data

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

    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Untransform observation data.

        This is typically done in-place. This class implements the identity
        transform (does nothing).

        Args:
            observation_data: Observation data, in transformed space
            observation_features: Corresponding observation features, in same
                space.

        Returns: observation data in original space.
        """
        return observation_data
