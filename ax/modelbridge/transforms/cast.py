#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.core.observation import Observation, ObservationFeatures, separate_observations
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
        modelbridge: Optional["modelbridge_module.base.Adapter"] = None,
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

    def transform_observations(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """Transform observations.

        Typically done in place. By default, the effort is split into separate
        transformations of the features and the data.

        NOTE: We overwrite it here, since ``transform_observation_features`` will drop
        features with ``None`` in them, leading to errors in the base implementation.

        Args:
            observations: Observations.

        Returns: transformed observations.
        """
        obs_feats, obs_data = separate_observations(observations=observations)
        # NOTE: looping here is ok, since the underlying methods for Cast also process
        # the features one by one in a loop.
        trans_obs = []
        for obs_ft, obs_d, obs in zip(obs_feats, obs_data, observations, strict=True):
            tf_obs_feats = self.transform_observation_features(
                observation_features=[obs_ft]
            )
            if len(tf_obs_feats) == 1:
                # Only re-package if the observation features haven't been dropped.
                trans_obs.append(
                    Observation(
                        features=tf_obs_feats[0], data=obs_d, arm_name=obs.arm_name
                    )
                )

        return trans_obs

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Transform observation features by
        - adding parameter values that were removed during casting of observation
          features to hierarchical search space;
        - casting parameter values to the corresponding parameter type;
        - dropping any observations with ``None`` parameter values.

        Args:
            observation_features: Observation features

        Returns: transformed observation features
        """
        observation_features = self._cast_parameter_values(
            observation_features=observation_features
        )

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
        observation_features = self._cast_parameter_values(
            observation_features=observation_features
        )

        if not self.flatten_hss:
            return observation_features

        return [
            assert_is_instance(
                self.search_space, HierarchicalSearchSpace
            ).cast_observation_features(observation_features=obs_feats)
            for obs_feats in observation_features
        ]

    def _cast_parameter_values(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """Cast parameter values of the given ``ObseravationFeatures`` to the
        ``ParameterType`` of the corresponding parameters in the search space.

        NOTE: This is done in-place. ``ObservationFeatures`` with ``None``
        values are dropped.

        Args:
            observation_features: A list of ``ObservationFeatures`` to cast.

        Returns: observation features with casted parameter values.
        """
        new_obsf = []
        for obsf in observation_features:
            for p_name, p_value in obsf.parameters.items():
                if p_value is None:
                    # Skip obsf if there are `None`s.
                    # The else block below will not be executed.
                    break
                if p_name in self.search_space.parameters:
                    obsf.parameters[p_name] = self.search_space[p_name].cast(p_value)
            else:
                # No `None`s in the parameterization.
                new_obsf.append(obsf)
        return new_obsf
