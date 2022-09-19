#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.types import TCandidateMetadata, TParameterization
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none


TIME_COLS = {"start_time", "end_time"}

OBS_COLS: Set[str] = {
    "arm_name",
    "trial_index",
    "random_split",
    "fidelities",
    *TIME_COLS,
}

OBS_KWARGS: Set[str] = {"trial_index", "random_split", *TIME_COLS}


class ObservationFeatures(Base):
    """The features of an observation.

    These include both the arm parameters and the features of the
    observation found in the Data object: trial index, times,
    and random split. This object is meant to contain everything needed to
    represent this observation in a model feature space. It is essentially a
    row of Data joined with the arm parameters.

    An ObservationFeatures object would typically have a corresponding
    ObservationData object that provides the observed outcomes.

    Attributes:
        parameters: arm parameters
        trial_index: trial index
        start_time: batch start time
        end_time: batch end time
        random_split: random split

    """

    def __init__(
        self,
        parameters: TParameterization,
        trial_index: Optional[np.int64] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        random_split: Optional[np.int64] = None,
        metadata: TCandidateMetadata = None,
    ) -> None:
        self.parameters = parameters
        self.trial_index = trial_index
        self.start_time = start_time
        self.end_time = end_time
        self.random_split = random_split
        self.metadata = metadata

    @staticmethod
    def from_arm(
        arm: Arm,
        trial_index: Optional[np.int64] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        random_split: Optional[np.int64] = None,
        metadata: TCandidateMetadata = None,
    ) -> ObservationFeatures:
        """Convert a Arm to an ObservationFeatures, including additional
        data as specified.
        """
        return ObservationFeatures(
            parameters=arm.parameters,
            trial_index=trial_index,
            start_time=start_time,
            end_time=end_time,
            random_split=random_split,
            metadata=metadata,
        )

    def update_features(self, new_features: ObservationFeatures) -> ObservationFeatures:
        """Updates the existing ObservationFeatures with the fields of the the input.

        Adds all of the new parameters to the existing parameters and overwrites
        any other fields that are not None on the new input features."""
        self.parameters.update(new_features.parameters)
        if new_features.trial_index is not None:
            self.trial_index = new_features.trial_index
        if new_features.start_time is not None:
            self.start_time = new_features.start_time
        if new_features.end_time is not None:
            self.end_time = new_features.end_time
        if new_features.random_split is not None:
            self.random_split = new_features.random_split
        return self

    def clone(
        self, replace_parameters: Optional[TParameterization] = None
    ) -> ObservationFeatures:
        """Make a copy of these ``ObservationFeatures``.

        Args:
            replace_parameters: An optimal parameterization, to which to set the
                parameters of the cloned ``ObservationFeatures``. Useful when
                transforming observation features in a way that requires a
                change to parameterization –– for example, while casting it to
                a hierarchical search space.
        """
        parameters = (
            self.parameters if replace_parameters is None else replace_parameters
        )
        return ObservationFeatures(
            parameters=parameters.copy(),
            trial_index=self.trial_index,
            start_time=self.start_time,
            end_time=self.end_time,
            random_split=self.random_split,
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        strs = []
        for attr in ["trial_index", "start_time", "end_time", "random_split"]:
            if getattr(self, attr) is not None:
                strs.append(", {attr}={val}".format(attr=attr, val=getattr(self, attr)))
        repr_str = "ObservationFeatures(parameters={parameters}".format(
            parameters=self.parameters
        )
        repr_str += "".join(strs) + ")"
        return repr_str

    def __hash__(self) -> int:
        parameters = self.parameters.copy()
        for k, v in parameters.items():
            if type(v) is np.int64:
                parameters[k] = int(v)  # pragma: no cover
            elif type(v) is np.float32:
                parameters[k] = float(v)  # pragma: no cover
        return hash(
            (
                json.dumps(parameters, sort_keys=True),
                self.trial_index,
                self.start_time,
                self.end_time,
                self.random_split,
            )
        )


class ObservationData(Base):
    """Outcomes observed at a point.

    The "point" corresponding to this ObservationData would be an
    ObservationFeatures object.

    Attributes:
        metric_names: A list of k metric names that were observed
        means: a k-array of observed means
        covariance: a (k x k) array of observed covariances
    """

    def __init__(
        self, metric_names: List[str], means: np.ndarray, covariance: np.ndarray
    ) -> None:
        k = len(metric_names)
        if means.shape != (k,):
            raise ValueError(
                "Shape of means should be {}, is {}.".format((k,), (means.shape))
            )
        if covariance.shape != (k, k):
            raise ValueError(
                "Shape of covariance should be {}, is {}.".format(
                    (k, k), (covariance.shape)
                )
            )
        self.metric_names = metric_names
        self.means = means
        self.covariance = covariance

    @property
    def means_dict(self) -> Dict[str, float]:
        """Extract means from this observation data as mapping from metric name to
        mean.
        """
        return dict(zip(self.metric_names, self.means))

    @property
    def covariance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Extract covariance matric from this observation data as mapping from
        metric name (m1) to mapping of another metric name (m2) to the covariance
        of the two metrics (m1 and m2).
        """
        return {
            m1: {
                m2: float(self.covariance[idx1][idx2])
                for idx2, m2 in enumerate(self.metric_names)
            }
            for idx1, m1 in enumerate(self.metric_names)
        }

    def __repr__(self) -> str:
        return "ObservationData(metric_names={mn}, means={m}, covariance={c})".format(
            mn=self.metric_names, m=self.means, c=self.covariance
        )


class Observation(Base):
    """Represents an observation.

    A set of features (ObservationFeatures) and corresponding measurements
    (ObservationData). Optionally, an arm name associated with the features.

    Attributes:
        features (ObservationFeatures)
        data (ObservationData)
        arm_name (Optional[str])
    """

    def __init__(
        self,
        features: ObservationFeatures,
        data: ObservationData,
        arm_name: Optional[str] = None,
    ) -> None:
        self.features = features
        self.data = data
        self.arm_name = arm_name


def _observations_from_dataframe(
    experiment: Experiment,
    df: pd.DataFrame,
    cols: List[str],
    arm_name_only: bool,
    map_keys: Iterable[str],
    include_abandoned: bool,
    map_keys_as_parameters: bool = False,
) -> List[Observation]:
    """Helper method for extracting observations grouped by `cols` from `df`.

    Args:
        experiment: Experiment with arm parameters.
        df: DataFrame derived from experiment Data.
        cols: columns used to group data into different observations.
        map_keys: columns that map dict-like Data
            e.g. `timestamp` in timeseries data, `epoch` in ML training traces.
        include_abandoned: Whether data for abandoned trials and arms should
            be included in the observations, returned from this function.
        map_keys_as_parameters: Whether map_keys should be returned as part of
            the parameters of the Observation objects.

    Returns:
        List of Observation objects.
    """
    observations = []
    abandoned_arms_dict = {}
    for g, d in df.groupby(by=cols):
        obs_kwargs = {}
        if arm_name_only:
            features = {"arm_name": g}
            arm_name = g
            trial_index = None
        else:
            features = dict(zip(cols, g))
            arm_name = features["arm_name"]
            trial_index = features.get("trial_index", None)

        if trial_index is not None:
            trial = experiment.trials[trial_index]
            metadata = trial._get_candidate_metadata(arm_name) or {}
            if Keys.TRIAL_COMPLETION_TIMESTAMP not in metadata:
                if trial._time_completed is not None:
                    metadata[Keys.TRIAL_COMPLETION_TIMESTAMP] = not_none(
                        trial._time_completed
                    ).timestamp()
            obs_kwargs[Keys.METADATA] = metadata

            if not include_abandoned and trial.status.is_abandoned:
                # Exclude abandoned trials.
                continue

            if not include_abandoned and isinstance(trial, BatchTrial):
                # Exclude abandoned arms from batch trial's observations.
                if trial.index not in abandoned_arms_dict:
                    # Same abandoned arm names to dict to avoid recomputing them
                    # on creation of every observation.
                    abandoned_arms_dict[trial.index] = trial.abandoned_arm_names
                if arm_name in abandoned_arms_dict[trial.index]:
                    continue

        obs_parameters = experiment.arms_by_name[arm_name].parameters.copy()
        if obs_parameters:
            obs_kwargs["parameters"] = obs_parameters
        for f, val in features.items():
            if f in OBS_KWARGS:
                obs_kwargs[f] = val
        fidelities = features.get("fidelities")
        if fidelities is not None:
            obs_parameters.update(json.loads(fidelities))

        for map_key in map_keys:
            if map_key in obs_parameters or map_keys_as_parameters:
                obs_parameters[map_key] = features[map_key]
            else:
                obs_kwargs[Keys.METADATA][map_key] = features[map_key]
        observations.append(
            Observation(
                features=ObservationFeatures(**obs_kwargs),
                data=ObservationData(
                    metric_names=d["metric_name"].tolist(),
                    means=d["mean"].values,
                    covariance=np.diag(d["sem"].values ** 2),
                ),
                arm_name=arm_name,
            )
        )
    return observations


def get_feature_cols(data: Data) -> List[str]:
    feature_cols = OBS_COLS.intersection(data.df.columns)

    for column in TIME_COLS:
        if column in feature_cols and len(data.df[column].unique()) > 1:
            warnings.warn(
                f"`{column} is not consistent and being discarded from observation data"
            )
            feature_cols.discard(column)

    return list(feature_cols)


def get_feature_cols_from_map_data(map_data: MapData) -> List[str]:
    feature_cols = (OBS_COLS.intersection(map_data.df.columns)).union(map_data.map_keys)

    for column in TIME_COLS:
        if column in feature_cols and len(map_data.df[column].unique()) > 1:
            warnings.warn(
                f"`{column} is not consistent and being discarded from observation data"
            )
            feature_cols.discard(column)

    return list(feature_cols)


def observations_from_data(
    experiment: Experiment, data: Data, include_abandoned: bool = False
) -> List[Observation]:
    """Convert Data to observations.

    Converts a Data object to a list of Observation objects. Pulls arm parameters from
    from experiment. Overrides fidelity parameters in the arm with those found in the
    Data object.

    Uses a diagonal covariance matrix across metric_names.

    Args:
        experiment: Experiment with arm parameters.
        data: Data of observations.
        include_abandoned: Whether data for abandoned trials and arms should
            be included in the observations, returned from this function.

    Returns:
        List of Observation objects.
    """

    feature_cols = get_feature_cols(data)
    observations = []
    arm_name_only = len(feature_cols) == 1  # there will always be an arm name
    # One DataFrame where all rows have all features.
    isnull = data.df[feature_cols].isnull()
    isnull_any = isnull.any(axis=1)
    incomplete_df_cols = isnull[isnull_any].any()

    # Get the incomplete_df columns that are complete, and usable as groupby keys.
    complete_feature_cols = list(
        OBS_COLS.intersection(incomplete_df_cols.index[~incomplete_df_cols])
    )

    if set(feature_cols) == set(complete_feature_cols):
        complete_df = data.df
        incomplete_df = None
    else:
        # The groupby and filter is expensive, so do it only if we have to.
        grouped = data.df.groupby(by=complete_feature_cols)
        complete_df = grouped.filter(lambda r: ~r[feature_cols].isnull().any().any())
        incomplete_df = grouped.filter(lambda r: r[feature_cols].isnull().any().any())

    # Get Observations from complete_df
    observations.extend(
        _observations_from_dataframe(
            experiment=experiment,
            df=complete_df,
            cols=feature_cols,
            arm_name_only=arm_name_only,
            map_keys=[],
            include_abandoned=include_abandoned,
        )
    )
    if incomplete_df is not None:
        # Get Observations from incomplete_df
        observations.extend(
            _observations_from_dataframe(
                experiment=experiment,
                df=incomplete_df,
                cols=complete_feature_cols,
                arm_name_only=arm_name_only,
                map_keys=[],
                include_abandoned=include_abandoned,
            )
        )
    return observations


def observations_from_map_data(
    experiment: Experiment,
    map_data: MapData,
    include_abandoned: bool = False,
    map_keys_as_parameters: bool = False,
    limit_total_rows: Optional[int] = None,
    limit_rows_per_group: Optional[int] = None,
) -> List[Observation]:
    """Convert MapData to observations.

    Converts a MapData object to a list of Observation objects. Pulls arm parameters
    from experiment. Overrides fidelity parameters in the arm with those found in the
    Data object.

    Uses a diagonal covariance matrix across metric_names.

    Args:
        experiment: Experiment with arm parameters.
        map_data: MapData of observations.
        include_abandoned: Whether data for abandoned trials and arms should
            be included in the observations, returned from this function.
        map_keys_as_parameters: Whether map_keys should be returned as part of
            the parameters of the Observation objects.
        limit_total_rows: If specified, uses MapData.subsample() with
            `limit_total_rows` equal to the specified value on the first
            map_key (map_data.map_keys[0]) to subsample the MapData. This is
            useful in, e.g., cases where learning curves are frequently
            updated, leading to an intractable number of Observation objects
            created.
        limit_rows_per_group: If specified, uses MapData.subsample() with
            `limit_rows_per_group` equal to the specified value on the first
            map_key (map_data.map_keys[0]) to subsample the MapData.

    Returns:
        List of Observation objects.
    """
    if limit_total_rows is not None or limit_rows_per_group is not None:
        map_data = map_data.subsample(
            map_key=map_data.map_keys[0],
            limit_total_rows=limit_total_rows,
            limit_rows_per_group=limit_rows_per_group,
            include_first_last=True,
        )
    feature_cols = get_feature_cols_from_map_data(map_data)
    observations = []
    arm_name_only = len(feature_cols) == 1  # there will always be an arm name
    # One DataFrame where all rows have all features.
    isnull = map_data.map_df[feature_cols].isnull()
    isnull_any = isnull.any(axis=1)
    incomplete_df_cols = isnull[isnull_any].any()

    # Get the incomplete_df columns that are complete, and usable as groupby keys.
    obs_cols_and_map = OBS_COLS.union(map_data.map_keys)
    complete_feature_cols = list(
        obs_cols_and_map.intersection(incomplete_df_cols.index[~incomplete_df_cols])
    )

    if set(feature_cols) == set(complete_feature_cols):
        complete_df = map_data.map_df
        incomplete_df = None
    else:
        # The groupby and filter is expensive, so do it only if we have to.
        grouped = map_data.map_df.groupby(by=complete_feature_cols)
        complete_df = grouped.filter(lambda r: ~r[feature_cols].isnull().any().any())
        incomplete_df = grouped.filter(lambda r: r[feature_cols].isnull().any().any())

    # Get Observations from complete_df
    observations.extend(
        _observations_from_dataframe(
            experiment=experiment,
            df=complete_df,
            cols=feature_cols,
            arm_name_only=arm_name_only,
            map_keys=map_data.map_keys,
            include_abandoned=include_abandoned,
            map_keys_as_parameters=map_keys_as_parameters,
        )
    )
    if incomplete_df is not None:
        # Get Observations from incomplete_df
        observations.extend(
            _observations_from_dataframe(
                experiment=experiment,
                df=incomplete_df,
                cols=complete_feature_cols,
                arm_name_only=arm_name_only,
                map_keys=map_data.map_keys,
                include_abandoned=include_abandoned,
                map_keys_as_parameters=map_keys_as_parameters,
            )
        )
    return observations


def separate_observations(
    observations: List[Observation], copy: bool = False
) -> Tuple[List[ObservationFeatures], List[ObservationData]]:
    """Split out observations into features+data.

    Args:
        observations: input observations

    Returns:
        observation_features: ObservationFeatures
        observation_data: ObservationData
    """
    if copy:
        observation_features = [deepcopy(obs.features) for obs in observations]
        observation_data = [deepcopy(obs.data) for obs in observations]
    else:
        observation_features = [obs.features for obs in observations]
        observation_data = [obs.data for obs in observations]
    return observation_features, observation_data


def recombine_observations(
    observation_features: List[ObservationFeatures],
    observation_data: List[ObservationData],
) -> List[Observation]:
    if len(observation_features) != len(observation_data):
        raise ValueError("Got features and data of different lengths")
    return [
        Observation(features=observation_features[i], data=obsd)
        for i, obsd in enumerate(observation_data)
    ]
