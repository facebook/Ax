#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationFeatures
from ax.core.observation_utils import separate_observations
from ax.core.parameter import (
    DerivedParameter,
    get_dummy_value_for_parameter,
    PARAMETER_PYTHON_TYPE_MAP,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from pandas import DataFrame
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class Cast(Transform):
    """Cast each param value to the respective parameter's type/format and
    to a flattened version of the hierarchical search space, if applicable.

    This is a default transform that should run across all models.

    NOTE: In case where the search space is hierarchical and this transform is
    configured to flatten it:
      * All calls to `Cast.transform_...` transform Ax objects defined in
        terms of hierarchical search space, to their definitions in terms of
        flattened search space.
      * All calls to `Cast.untransform_...` cast Ax objects back to a
        hierarchical search space.
      * The hierarchical search space is seen as the "original" search space,
        and the flattened search space---as "transformed".

    Transform is done in-place for casting types, but objects are copied
    during flattening of- and casting to the hierarchical search space.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        self.search_space: SearchSpace = none_throws(search_space).clone()
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

        # 1. The default behavior for non-hierarchical search spaces: The flag
        # `flatten_hss` is irrelevant. It's simply ignored.
        #
        # 2. The default behavior for hierarchical search spaces: `flatten_hss=True`.
        # The search space is flattened, and the inactive parameters are filled with
        # dummy values.
        #
        # 3. To exploit the hierarchical structure, one need set `flatten_hss=False`.
        # The search space will be kept as is. The observation features are still
        # flattened so that the resulting tensors received by BoTorch have the same
        # dimensionality. If the underlying BoTorch model is designed to exploit the
        # hierarchical structure, it infers which parameter is active based
        # on `search_space_digest.hierarchical_dependencies`.
        self.flatten_hss: bool = assert_is_instance(
            self.config.pop("flatten_hss", none_throws(search_space).is_hierarchical),
            bool,
        )
        self.inject_dummy_values_to_complete_flat_parameterization: bool = (
            assert_is_instance(
                self.config.pop(
                    "inject_dummy_values_to_complete_flat_parameterization", True
                ),
                bool,
            )
        )
        if self.config:
            raise UserInputError(
                f"Unexpected config parameters for `Cast` transform: {self.config}."
            )
        self.derived_parameters: dict[str, DerivedParameter] = {
            name: p.clone()
            for name, p in self.search_space.nontunable_parameters.items()
            if isinstance(p, DerivedParameter)
        }

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        """If the transform is configured to flatten the search space
        (``self.flatten_hss``), flattens the hierarchical search space by removing
        all dependent  information, and returns the flat ``SearchSpace``.
        Does nothing if the search space is not hierarchical.

        NOTE: All calls to `Cast.transform_...` transform Ax objects defined in
        terms of hierarchical search space, to their definitions in terms of
        flattened search space. All calls to `Cast.untransform_...` cast Ax
        objects back to a hierarchical search space.

        Args:
            search_space: The search space to flatten.

        Returns: transformed search space.
        """
        if search_space.is_hierarchical:
            return search_space.flatten() if self.flatten_hss else search_space
        else:
            return search_space

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

        if self.search_space.is_hierarchical:
            # Inject the parameters model suggested in the flat search space, which then
            # got removed during casting to HSS as they were not applicable under the
            # hierarchical structure of the search space.
            return [
                self.search_space.flatten_observation_features(
                    observation_features=obs_feats,
                    inject_dummy_values_to_complete_flat_parameterization=(
                        self.inject_dummy_values_to_complete_flat_parameterization
                    ),
                )
                for obs_feats in observation_features
            ]
        else:
            return observation_features

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

        if self.search_space.is_hierarchical:
            # The inactive parameters in the HSS have been filled with dummy values,
            # which should be removed from the observations.
            return [
                self.search_space.cast_observation_features(
                    observation_features=obs_feats
                )
                for obs_feats in observation_features
            ]
        else:
            return observation_features

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
            has_none = False
            for p_name, p_value in obsf.parameters.items():
                if p_value is None:
                    has_none = True
                    # Skip obsf if there are `None`s.
                    break
                if p_name in self.derived_parameters:
                    continue
                elif p_name in self.search_space.parameters:
                    obsf.parameters[p_name] = self.search_space[p_name].cast(p_value)

            if not has_none:
                # Re-compute derived parameter values, since casting
                # may change the consitutent parameter values (e.g.
                # via rounding)
                for p_name, p in self.derived_parameters.items():
                    if p_name in obsf.parameters:
                        obsf.parameters[p_name] = p.compute(parameters=obsf.parameters)
                # No `None`s in the parameterization.
                new_obsf.append(obsf)
        return new_obsf

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Cast the column values of the given experiment data to the
        data type corresponding to the ``ParameterType``. If any rows
        include None / NaN values for the parameterization, they are dropped.

        For hierarchical search spaces, the parameterizations are flattened
        before applying the rest of the transform.
        """
        arm_data = experiment_data.arm_data
        # If applicable, flatten first to fill otherwise NaN values.
        # See `CastTransformTest.test_transform_experiment_data_flatten` for an
        # example of the arm_data before and after the flattening operation.
        # Any parameter values that are missing in the corresponding columns
        # are filled with the values from `metadata["Keys.FULL_PARAMETERIZATION"]`.
        # If the metadata does not include the full parameterization, a dummy
        # value is constructed using the middle of the parameter domain.
        if self.search_space.is_hierarchical:
            arm_data = self._flatten_arm_data(arm_data=arm_data)

        parameter_names = list(arm_data.columns)
        parameter_names.remove("metadata")

        # Drop rows with None / NaN values. These are dropped in `parameter.cast`
        # call in `Cast.transform_observation_features`.
        arm_data = arm_data.dropna(axis="index", how="any", subset=parameter_names)
        # Update observation_data to include the same indices.
        observation_data = experiment_data.observation_data
        index_names = list(observation_data.index.names)
        if len(index_names) == 2:
            observation_data = observation_data.loc[arm_data.index]
        else:
            observation_data = observation_data.loc[
                observation_data.index.droplevel(level=index_names[2:]).isin(
                    arm_data.index
                )
            ]
        # Add any missing columns to the arm_data to complete it.
        arm_data = arm_data.reindex(
            columns=list(self.search_space.parameters) + ["metadata"], fill_value=None
        )
        # Cast columns to the correct datatype & round RangeParameters, if applicable.
        type_map = PARAMETER_PYTHON_TYPE_MAP.copy()
        # pyre-ignore [6]: Writing str to type map that is typed with Types.
        # Basic int errors out with NaNs, which are added for missing columns above.
        # This happens with heterogeneous SS BOTL, where transforms work on joint space.
        type_map[ParameterType.INT] = "Int64"
        column_to_type = {
            p: type_map[param.parameter_type]
            for p, param in self.search_space.parameters.items()
        }
        arm_data = arm_data.astype(dtype=column_to_type)
        # Round to digits if any parameter specifies it.
        for p_name in parameter_names:
            parameter = self.search_space.parameters[p_name]
            if isinstance(parameter, RangeParameter) and parameter.digits is not None:
                arm_data[p_name] = arm_data[p_name].round(parameter.digits)

        return ExperimentData(arm_data=arm_data, observation_data=observation_data)

    def _flatten_arm_data(self, arm_data: DataFrame) -> DataFrame:
        """Flatten hierarchical search space parameterizations in arm_data using
        vectorized pandas operations.

        This method:
        1. Extracts full parameterizations from metadata
        2. Fills missing parameter values from metadata using vectorized operations
        3. Injects dummy values for remaining missing parameters (if configured)

        Args:
            arm_data: DataFrame with arm parameterizations and metadata.
                Modified in-place.

        Returns:
            DataFrame updated with flattened parameterizations, in-place.
        """
        # Extract full parameterizations from metadata column.
        full_params_df = DataFrame(
            arm_data["metadata"]
            .apply(lambda md: md.get(Keys.FULL_PARAMETERIZATION, {}))
            .tolist(),
            index=arm_data.index,
        )

        # Fill missing values from full parameterizations.
        for param_name in full_params_df.columns:
            if param_name not in arm_data.columns:
                arm_data[param_name] = full_params_df[param_name]
        arm_data.fillna(full_params_df, inplace=True)

        # Inject dummy values for remaining missing parameters if configured.
        if self.inject_dummy_values_to_complete_flat_parameterization:
            for param_name, param in self.search_space.parameters.items():
                dummy_value = get_dummy_value_for_parameter(param=param)
                if param_name not in arm_data.columns:
                    arm_data[param_name] = dummy_value
                else:
                    # Fill missing values with dummy value
                    missing_mask = arm_data[param_name].isna()
                    if missing_mask.any():
                        arm_data.loc[missing_mask, param_name] = dummy_value

        return arm_data
