#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
        """Do any initial computations for preparing the transform"""
        if config is None:
            config = {}
        self.config = config

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Transform search_space as needed to do modeling."""
        return search_space

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        """Transform optimization_config as needed to do modeling."""
        return optimization_config

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        """Transform observation_features as needed to do modeling."""
        return observation_features

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Transform observation_data as needed to do modeling."""
        return observation_data

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        """Transform observation_features used for modeling back to the original."""
        return observation_features

    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Transform observation_data used for modeling back to the original."""
        return observation_data
