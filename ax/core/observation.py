#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from copy import deepcopy
from logging import Logger

import ax.core.experiment as experiment
import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.trial_status import NON_ABANDONED_STATUSES, TrialStatus
from ax.core.types import TCandidateMetadata, TParameterization
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)

TIME_COLS = {"start_time", "end_time"}

OBS_COLS: set[str] = {
    "arm_name",
    "trial_index",
    "random_split",
    "fidelities",
    *TIME_COLS,
}

OBS_KWARGS: set[str] = {"trial_index", "random_split", *TIME_COLS}


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
        trial_index: int | None = None,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        random_split: int | None = None,
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
        trial_index: int | None = None,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        random_split: int | None = None,
        metadata: TCandidateMetadata = None,
    ) -> ObservationFeatures:
        """Convert a Arm to an ObservationFeatures, including additional
        data as specified.
        """
        return ObservationFeatures(
            # NOTE: Arm.parameters makes a copy of the original dict, so any
            # modifications to the parameters dict will not be reflected in
            # the original Arm parameters.
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
        self, replace_parameters: TParameterization | None = None
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
                strs.append(f", {attr}={getattr(self, attr)}")
        repr_str = "ObservationFeatures(parameters={parameters}".format(
            parameters=self.parameters
        )
        repr_str += "".join(strs) + ")"
        return repr_str

    def __hash__(self) -> int:
        parameters = self.parameters.copy()
        for k, v in parameters.items():
            if type(v) is np.int64:
                parameters[k] = int(v)
            elif type(v) is np.float32:
                parameters[k] = float(v)
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
        self,
        metric_names: list[str],
        means: npt.NDArray,
        covariance: npt.NDArray,
    ) -> None:
        k = len(metric_names)
        if means.shape != (k,):
            raise ValueError(f"Shape of means should be {(k,)}, is {(means.shape)}.")
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
    def means_dict(self) -> dict[str, float]:
        """Extract means from this observation data as mapping from metric name to
        mean.
        """
        return dict(zip(self.metric_names, self.means))

    @property
    def covariance_matrix(self) -> dict[str, dict[str, float]]:
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
        arm_name: str | None = None,
    ) -> None:
        self.features = features
        self.data = data
        self.arm_name = arm_name

    def __repr__(self) -> str:
        return (
            "Observation(\n"
            f"    features={self.features},\n"
            f"    data={self.data},\n"
            f"    arm_name='{self.arm_name}',\n"
            ")"
        )


def _observations_from_dataframe(
    experiment: experiment.Experiment,
    df: pd.DataFrame,
    cols: list[str],
    map_keys: Iterable[str],
    statuses_to_include: set[TrialStatus],
    statuses_to_include_map_metric: set[TrialStatus],
) -> list[Observation]:
    """Helper method for extracting observations grouped by `cols` from `df`.

    Args:
        experiment: Experiment with arm parameters.
        df: DataFrame derived from experiment Data.
        cols: columns used to group data into different observations.
            `cols` must always include `arm_name`.
        map_keys: columns that map dict-like Data
            e.g. `timestamp` in timeseries data, `epoch` in ML training traces.
        statuses_to_include: data from non-MapMetrics will only be included for trials
            with statuses in this set.
        statuses_to_include_map_metric: data from MapMetrics will only be included for
            trials with statuses in this set.

    Returns:
        List of Observation objects.
    """
    if len(cols) == 0:
        return []
    observations = []
    abandoned_arms_dict = {}
    # NOTE: dropna is important to avoid dropping the whole or part of the df if
    # a feature column is filled with NaN / NaT values.
    for g, d in df.groupby(by=cols, dropna=False):
        obs_kwargs = {}
        features = dict(zip(cols, g, strict=True))
        arm_name = features["arm_name"]
        trial_index = features.get("trial_index", None)

        is_arm_abandoned = False
        trial_status = None
        if trial_index is not None:
            trial = experiment.trials[trial_index]
            trial_status = trial.status
            metadata = trial._get_candidate_metadata(arm_name) or {}
            if Keys.TRIAL_COMPLETION_TIMESTAMP not in metadata:
                if trial._time_completed is not None:
                    metadata[Keys.TRIAL_COMPLETION_TIMESTAMP] = none_throws(
                        trial._time_completed
                    ).timestamp()
            obs_kwargs[Keys.METADATA] = metadata

            # Determine if this arm is abandoned.
            is_arm_abandoned = trial.is_abandoned
            if isinstance(trial, BatchTrial):
                if trial.index not in abandoned_arms_dict:
                    # Same abandoned arm names to dict to avoid recomputing them
                    # on creation of every observation.
                    abandoned_arms_dict[trial.index] = trial.abandoned_arm_names
                if arm_name in abandoned_arms_dict[trial.index]:
                    is_arm_abandoned = True

        obs_parameters = experiment.arms_by_name[arm_name].parameters.copy()
        if obs_parameters:
            obs_kwargs["parameters"] = obs_parameters
        for f, val in features.items():
            if f in OBS_KWARGS and not pd.isna(val):
                obs_kwargs[f] = val
        # add start and end time of trial if the start and end time
        # is the same for all metrics and arms
        for col in TIME_COLS:
            if col in d.columns:
                times = d[col]
                if times.nunique() == 1 and not times.isnull().any():
                    obs_kwargs[col] = times.iloc[0]
        fidelities = features.get("fidelities")
        if fidelities is not None:
            obs_parameters.update(json.loads(fidelities))

        for map_key in map_keys:
            obs_kwargs[Keys.METADATA][map_key] = features[map_key]
        d = _filter_data_on_status(
            df=d,
            experiment=experiment,
            trial_status=trial_status,
            is_arm_abandoned=is_arm_abandoned,
            statuses_to_include=statuses_to_include,
            statuses_to_include_map_metric=statuses_to_include_map_metric,
        )
        if len(d) == 0:
            continue
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


def _filter_data_on_status(
    df: pd.DataFrame,
    experiment: experiment.Experiment,
    trial_status: TrialStatus | None,
    is_arm_abandoned: bool,
    statuses_to_include: set[TrialStatus],
    statuses_to_include_map_metric: set[TrialStatus],
) -> pd.DataFrame:
    """Filters the dataframe to only include observations for the metrics attached
    to the experiment, and only the observations from the trials with statuses in
    corresponding `statuses_to_include` for each metric attached to the experiment.

    Args:
        df: Dataframe of observations to filter.
        experiment: The experiment to which the data belongs. Used to check whether
            the metric is attached to the experiment and the type of the metric.
        trial_status: The status of the trial to which the data belongs.
        is_arm_abandoned: Whether the arm to which the data belongs is abandoned.
            This is used to handle abandoned arms on a ``BatchTrial``. ``BatchTrial``
            may include abandoned arms even when the trial itself is not abandoned.
            Abandoned arms are treated as if they belong to an abandoned trial.
        statuses_to_include: Data from non-``MapMetric`` sub-classes will be filtered to
            only include observations from trials with statuses in this set.
        statuses_to_include_map_metric: Data from ``MapMetric`` sub-classes will be
            filtered to only include observations from trials with statuses in this set.

    Returns:
        A dataframe with filtered observations.
    """
    if "metric_name" not in df.columns:
        raise ValueError(f"`metric_name` column is missing from {df!r}.")
    dfs = []
    for g, d in df.groupby(by="metric_name"):
        metric_name = g
        if metric_name not in experiment.metrics:
            # Observations can only be made for metrics attached to the experiment.
            logger.exception(
                f"Data contains metric {metric_name} that has not been added to the "
                "experiment. You can either update the `optimization_config` or attach "
                "it as a tracking metric using `Experiment.add_tracking_metrics` "
                "or `AxClient.add_tracking_metrics`. Ignoring all data for "
                f"metric {metric_name}."
            )
            continue
        metric = experiment.metrics[metric_name]
        statuses_to_include_metric = (
            statuses_to_include_map_metric
            if isinstance(metric, MapMetric)
            else statuses_to_include
        )
        if trial_status is not None and trial_status not in statuses_to_include_metric:
            continue
        if is_arm_abandoned and TrialStatus.ABANDONED not in statuses_to_include_metric:
            continue
        dfs.append(d)
    if len(dfs) == 0:
        return pd.DataFrame()
    df = pd.concat(dfs)
    return df


def get_feature_cols(data: Data) -> list[str]:
    """Get the columns used to identify and group observations from a Data object.

    Args:
        data: the Data object from which to extract the feature columns.
            If the Data object is an instance of MapData, the map_keys will be
            included in the feature columns.

    Returns:
        A list of column names to be used to group observations.
    """
    feature_cols = OBS_COLS.intersection(data.true_df.columns)
    if isinstance(data, MapData):
        feature_cols = feature_cols.union(data.map_keys)

    for column in TIME_COLS:
        if column in feature_cols and len(data.df[column].unique()) > 1:
            warnings.warn(
                f"`{column} is not consistent and being discarded from "
                "observation data",
                stacklevel=5,
            )
            feature_cols.discard(column)
    # NOTE: This ensures the order of feature_cols is deterministic so that the order
    # of lists of observations are deterministic, to avoid nondeterministic tests.
    # Necessary for test_TorchAdapter.
    return sorted(feature_cols)


def observations_from_data(
    experiment: experiment.Experiment,
    data: Data,
    statuses_to_include: set[TrialStatus] | None = None,
    statuses_to_include_map_metric: set[TrialStatus] | None = None,
    latest_rows_per_group: int | None = None,
    limit_rows_per_metric: int | None = None,
    limit_rows_per_group: int | None = None,
) -> list[Observation]:
    """Convert Data (or MapData) to observations.

    Converts a Data (or MapData) object to a list of Observation objects.
    Pulls arm parameters from from experiment. Overrides fidelity parameters
    in the arm with those found in the Data object.

    Uses a diagonal covariance matrix across metric_names.

    Args:
        experiment: Experiment with arm parameters.
        data: Data (or MapData) of observations.
        statuses_to_include: data from non-MapMetrics will only be included for trials
            with statuses in this set. Defaults to all statuses except abandoned.
        statuses_to_include_map_metric: data from MapMetrics will only be included for
            trials with statuses in this set. Defaults to all statuses except abandoned.
        latest_rows_per_group: If specified and data is an instance of MapData,
            uses MapData.latest() with `rows_per_group=latest_rows_per_group` to
            retrieve the most recent rows for each group. Useful in cases where
            learning curves are frequently updated, preventing an excessive
            number of Observation objects. Overrides `limit_rows_per_metric`
            and `limit_rows_per_group`.
        limit_rows_per_metric: If specified and data is an instance of MapData,
            uses MapData.subsample() with `limit_rows_per_metric` on the first
            map_key (map_data.map_keys[0]) to subsample the MapData. Useful for
            managing the number of Observation objects when learning curves are
            frequently updated. Ignored if `latest_rows_per_group` is specified.
        limit_rows_per_group: If specified and data is an instance of MapData,
            uses MapData.subsample() with `limit_rows_per_group` on the first
            map_key (map_data.map_keys[0]) to subsample the MapData. Ignored if
            `latest_rows_per_group` is specified.

    Returns:
        List of Observation objects.
    """
    if statuses_to_include is None:
        statuses_to_include = NON_ABANDONED_STATUSES
    if statuses_to_include_map_metric is None:
        statuses_to_include_map_metric = NON_ABANDONED_STATUSES
    if isinstance(data, MapData):
        map_keys = data.map_keys
        if latest_rows_per_group is not None:
            data = data.latest(map_keys=map_keys, rows_per_group=latest_rows_per_group)
        elif limit_rows_per_metric is not None or limit_rows_per_group is not None:
            data = data.subsample(
                map_key=data.map_keys[0],
                limit_rows_per_metric=limit_rows_per_metric,
                limit_rows_per_group=limit_rows_per_group,
                include_first_last=True,
            )
        df = data.map_df
    else:
        map_keys = []
        df = data.df
    return _observations_from_dataframe(
        experiment=experiment,
        df=df,
        cols=get_feature_cols(data=data),
        map_keys=map_keys,
        statuses_to_include=statuses_to_include,
        statuses_to_include_map_metric=statuses_to_include_map_metric,
    )


def separate_observations(
    observations: list[Observation], copy: bool = False
) -> tuple[list[ObservationFeatures], list[ObservationData]]:
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
    observation_features: list[ObservationFeatures],
    observation_data: list[ObservationData],
    arm_names: list[str] | None = None,
) -> list[Observation]:
    """
    Construct a list of `Observation`s from the given arguments.

    In the returned list of `Observation`s, element `i` has `features` from
    `observation_features[i]`, `data` from `observation_data[i]`, and, if
    applicable, `arm_name` from `arm_names[i]`.
    """
    if len(observation_features) != len(observation_data):
        raise ValueError("Got features and data of different lengths")
    if arm_names is not None and len(observation_features) != len(arm_names):
        raise ValueError("Got features and arm_names of different lengths")
    return [
        Observation(
            features=observation_features[i],
            data=obsd,
            arm_name=None if arm_names is None else arm_names[i],
        )
        for i, obsd in enumerate(observation_data)
    ]
