#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.typeutils import checked_cast, not_none

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


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
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        self.search_space: SearchSpace = not_none(search_space).clone()
        self.flatten_hss: bool = (
            config is None or checked_cast(bool, config.get("flatten_hss", True))
        ) and isinstance(search_space, HierarchicalSearchSpace)

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
        return checked_cast(HierarchicalSearchSpace, search_space).flatten()

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
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
            checked_cast(
                HierarchicalSearchSpace, self.search_space
            ).flatten_observation_features(observation_features=obs_feats)
            for obs_feats in observation_features
        ]

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
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
            checked_cast(
                HierarchicalSearchSpace, self.search_space
            ).cast_observation_features(observation_features=obs_feats)
            for obs_feats in observation_features
        ]
