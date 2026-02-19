#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

import warnings
from copy import deepcopy
from logging import Logger

import ax.core.experiment as experiment
import numpy as np
import pandas as pd
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data, MAP_KEY
from ax.core.map_metric import MapMetric
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.trial_status import NON_ABANDONED_STATUSES, TrialStatus
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)
TIME_COLS = {"start_time", "end_time"}

OBS_COLS: set[str] = {"arm_name", "trial_index", *TIME_COLS}

OBS_KWARGS: set[str] = {"trial_index", *TIME_COLS}


def _observations_from_dataframe(
    experiment: experiment.Experiment,
    df: pd.DataFrame,
    cols: list[str],
    is_map_data: bool,
    statuses_to_include: set[TrialStatus],
    statuses_to_include_map_metric: set[TrialStatus],
) -> list[Observation]:
    """Helper method for extracting observations grouped by `cols` from `df`.

    Args:
        experiment: Experiment with arm parameters.
        df: DataFrame derived from experiment Data.
        cols: columns used to group data into different observations.
            `cols` must always include `arm_name`.
        is_map_data: Whether the data is from a MapMetric, with ``MAP_KEY`` in
            the metadata.
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

        if is_map_data:
            obs_kwargs[Keys.METADATA][MAP_KEY] = features[MAP_KEY]
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
                    metric_signatures=d["metric_signature"].tolist(),
                    means=d["mean"].to_numpy(copy=True),
                    covariance=np.diag(d["sem"].to_numpy(copy=True) ** 2),
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
    if "metric_signature" not in df.columns:
        raise ValueError(f"`metric_signature` column is missing from {df!r}.")
    dfs = []
    metric_signature_to_name = {
        m.signature: m.name for m in experiment.metrics.values()
    }
    for g, d in df.groupby(by="metric_signature"):
        metric_signature = g
        if metric_signature not in metric_signature_to_name.keys():
            # Observations can only be made for metrics attached to the experiment.
            logger.exception(
                f"Data contains metric {metric_signature} that has not been added to "
                "the experiment. You can either update the `optimization_config` "
                "or attach it as a tracking metric using "
                "`Experiment.add_tracking_metrics` or `AxClient.add_tracking_metrics`. "
                "Ignoring all data for "
                f"metric {metric_signature}."
            )
            continue
        metric = experiment.metrics[metric_signature_to_name[metric_signature]]
        statuses_to_include_metric = (
            statuses_to_include_map_metric
            if isinstance(metric, MapMetric) and metric.has_map_data
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
        data: the Data object from which to extract the feature columns. If Data
            has a "step" (MAP_KEY) column, it will be included in the feature
            columns.

    Returns:
        A list of column names to be used to group observations.
    """
    feature_cols = OBS_COLS.intersection(data.full_df.columns)
    if data.has_step_column:
        feature_cols.add(MAP_KEY)

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
    experiment: experiment.Experiment, data: Data
) -> list[Observation]:
    """Convert Data to observations.

    Converts a Data object to a list of Observation objects.
    Pulls arm parameters from from experiment. Overrides fidelity parameters
    in the arm with those found in the Data object.

    Uses a diagonal covariance matrix across metric_signatures.

    Args:
        experiment: Experiment with arm parameters.
        data: Data.

    Returns:
        List of Observation objects.
    """
    return _observations_from_dataframe(
        experiment=experiment,
        df=data.full_df,
        cols=get_feature_cols(data=data),
        is_map_data=data.has_step_column,
        statuses_to_include=NON_ABANDONED_STATUSES,
        statuses_to_include_map_metric=NON_ABANDONED_STATUSES,
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
