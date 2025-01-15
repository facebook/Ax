#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.exceptions.core import UserInputError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class Cast(Transform):
    """Cast each param value to the respective parameter's type/format and
    to a flattened version of the hierarchical search space, if applicable.

    This is a default transform that should run across all models.

    NOTE: In case where searh space is hierarchical and this transform is
    configured to flatten it:
      * All calls to `Cast.transform_...` transform Ax objects defined in
        terms of hierarchical search space, to their definitions in terms of
        flattened search space.
      * All calls to `Cast.untransform_...` cast Ax objects back to a
        hierarchical search space.
      * The hierarchical search space is seen as the "original" search space,
        and the flattened search space –– as "transformed".

    Transform is done in-place for casting types, but objects are copied
    during flattening of- and casting to the hierarchical search space.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        self.search_space: SearchSpace = none_throws(search_space).clone()
        config = (config or {}).copy()
        self.flatten_hss: bool = assert_is_instance(
            config.pop(
                "flatten_hss", isinstance(search_space, HierarchicalSearchSpace)
            ),
            bool,
        )
        self.inject_dummy_values_to_complete_flat_parameterization: bool = (
            assert_is_instance(
                config.pop(
                    "inject_dummy_values_to_complete_flat_parameterization", True
                ),
                bool,
            )
        )
        self.use_random_dummy_values: bool = assert_is_instance(
            config.pop("use_random_dummy_values", False), bool
        )
        if config:
            raise UserInputError(
                f"Unexpected config parameters for `Cast` transform: {config}."
            )

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """Flattens the hierarchical search space and returns the flat
        ``SearchSpace`` if this transform is configured to flatten hierarchical
        search spaces. Does nothing if the search space is not hierarchical.

        NOTE: All calls to `Cast.transform_...` transform Ax objects defined in
        terms of hierarchical search space, to their definitions in terms of
        flattened search space. All calls to `Cast.untransform_...` cast Ax
        objects back to a hierarchical search space.

        Args:
            search_space: The search space to flatten.

        Returns: transformed search space.
        """
        if not self.flatten_hss:
            return search_space
        return assert_is_instance(search_space, HierarchicalSearchSpace).flatten()

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Transform observation features by adding parameter values that
        were removed during casting of observation features to hierarchical
        search space.

        Args:
            observation_features: Observation features

        Returns: transformed observation features
        """
        if not self.flatten_hss:
            return observation_features
        # Inject the parameters model suggested in the flat search space, which then
        # got removed during casting to HSS as they were not applicable under the
        # hierarchical structure of the search space.
        return [
            assert_is_instance(
                self.search_space, HierarchicalSearchSpace
            ).flatten_observation_features(
                observation_features=obs_feats,
                inject_dummy_values_to_complete_flat_parameterization=(
                    self.inject_dummy_values_to_complete_flat_parameterization
                ),
                use_random_dummy_values=self.use_random_dummy_values,
            )
            for obs_feats in observation_features
        ]

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Untransform observation features by casting parameter values to their
        expected types and removing parameter values that are not applicable given
        the values of other parameters and the hierarchical structure of the search
        space.

        Args:
            observation_features: Observation features in the transformed space

        Returns: observation features in the original space
        """
        for obsf in observation_features:
            for p_name, p_value in obsf.parameters.items():
                if p_name in self.search_space.parameters:
                    obsf.parameters[p_name] = self.search_space[p_name].cast(p_value)

        if not self.flatten_hss:
            return observation_features

        return [
            assert_is_instance(
                self.search_space, HierarchicalSearchSpace
            ).cast_observation_features(observation_features=obs_feats)
            for obs_feats in observation_features
        ]
