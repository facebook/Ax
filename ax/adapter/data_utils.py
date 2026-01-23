#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


"""
Utilities for working with data for an Ax experiment.
Unlike the Ax Data object, "data" here refers to both the arm
parameterizations and the metric observations (from a Data object).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, InitVar
from typing import Any

import numpy as np
from ax.core.data import Data, MAP_KEY
from ax.core.experiment import Experiment
from ax.core.map_metric import MapMetric
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.trial_status import STATUSES_EXPECTING_DATA, TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.utils.common.constants import Keys
from pandas import DataFrame, MultiIndex, Series
from pyre_extensions import none_throws


@dataclass(frozen=True)
class DataLoaderConfig:
    """This dataclass contains parameters that control the behavior
    of `Adapter._set_and_filter_training_data`.

    Args:
        fit_abandoned: Whether data for abandoned arms or trials should be included in
            model training data. If `False`, only non-abandoned points are returned.
        fit_only_completed_map_metrics: Whether to fit a model to map metrics only when
            the trial is completed. This is useful for applications like modeling
            partially completed learning curves in AutoML.
        latest_rows_per_group: If specified and data has `has_step_column=True`, uses
            Data.latest() with `latest_rows_per_group` to retrieve the most recent
            rows for each group. Useful in cases where learning curves are frequently
            updated, preventing an excessive number of Observation objects.
        limit_rows_per_metric: Subsample the map data so that the total number of
            rows per metric is limited by this value.
        limit_rows_per_group: Subsample the map data so that the number of rows
            in the `MAP_KEY` column for each (arm, metric) is limited by this value.
    """

    # pyre-ignore [16]: Pyre doesn't understand InitVar.
    fit_out_of_design: InitVar[bool | None] = None
    fit_abandoned: bool = False
    fit_only_completed_map_metrics: bool = False
    latest_rows_per_group: int | None = 1
    limit_rows_per_metric: int | None = None
    limit_rows_per_group: int | None = None

    def __post_init__(self, fit_out_of_design: bool | None) -> None:
        if self.latest_rows_per_group is not None and (
            self.limit_rows_per_metric is not None
            or self.limit_rows_per_group is not None
        ):
            raise UnsupportedError(
                "`latest_rows_per_group` must be None if either of "
                "`limit_rows_per_metric` or `limit_rows_per_group` is specified."
            )
        if fit_out_of_design is not None:
            warnings.warn(
                # Deprecated after Ax 1.1.2. Can be removed with Ax 1.3.0.
                "`fit_out_of_design` is deprecated and will be removed in the future. "
                "Please remove it from your inputs to avoid future errors.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def statuses_to_fit(self) -> set[TrialStatus]:
        """The data from trials in these statuses will be used to fit the model
        for non map metrics. Defaults to all trial statuses if
        `fit_abandoned is True` and all statuses except ABANDONED, otherwise.
        """
        if self.fit_abandoned:
            return set(TrialStatus)
        return set(STATUSES_EXPECTING_DATA)

    @property
    def statuses_to_fit_map_metric(self) -> set[TrialStatus]:
        """The data from trials in these statuses will be used to fit the model
        for map metrics. Defaults to only COMPLETED trials if
        `fit_only_completed_map_metrics is True` and to `statuses_to_fit`, otherwise.
        """
        if self.fit_only_completed_map_metrics:
            return {TrialStatus.COMPLETED}
        return self.statuses_to_fit


@dataclass
class ExperimentData:
    """A container for the data from arms and observations of an experiment.

    This is intended as a lightweight container for the data that gets processed
    within the ``Adapter`` / ``Transform`` layer. This will be passed through
    transforms, allowing for column-wise, vectorized transforming of the values.
    This can then be used to construct any necessary data structures for the
    ``Generator``, such as the ``SupervisedDataset`` objects constructed
    in ``TorchAdapter``.

    NOTE: This should only be constructed using the ``extract_experiment_data`` helper.
    Construction of the underlying dataframes manually from scratch can be quite
    complicated and error prone due to the multi-index involved in both the
    index and columns of the dataframes.

    Attributes:
        arm_data: A dataframe, indexed by (trial_index, arm_name), containing the
            the parameterization of each arm, with one column per parameter, and
            a column for the metadata. Each row corresponds to the parameterization
            and metadata for the given (trial_index, arm_name) pair.
        observation_data: A dataframe, indexed by (trial_index, arm_name[, *map_keys])
            map_keys being optional, containing the mean and sem observations for each
            metric. The columns of the dataframe are multi-indexed, with the first level
            being "mean" or "sem" and the second level being the metric signature.
            This is typically constructed by pivoting `(Map)Data.full_df`.
            If the `Data` object contains additional metadata columns like `start_time`
            and `end_time`, these will be carried onto `observation_data`. The metadata
            columns will be indexed with the first level being "metadata" and the second
            level keeping the original column name.

    Example with non-map data:
        >>> experiment = Experiment(...)  # An experiment with non-map Data.
        >>> experiment_data = extract_experiment_data(
        ...     experiment=experiment,
        ...     data_loader_config=DataLoaderConfig(),
        ... )
        >>> print(experiment_data.arm_data)
                                     x         y
        trial_index arm_name
        0           0_0       0.539644  0.878898
        1           1_0       0.341846  0.213774
        3           3_0       0.837545  0.732969
        >>> print(experiment_data.observation_data)
                             mean      sem
        metric_signature            m1   m2  m1  m2
        trial_index arm_name
        0           0_0       0.1  1.0 NaN NaN
        1           1_0       0.2  2.0 NaN NaN

    Example with Data that has a "step" column:
        >>> experiment = Experiment(...)
        >>> experiment_data = extract_experiment_data(
        ...     experiment=experiment,
        ...     data_loader_config=DataLoaderConfig(
        ...         fit_only_completed_map_metrics=False,
        ...         latest_rows_per_group=None,
        ...     ),
        ... )
        >>> print(experiment_data.arm_data)
                               x1   x2
        trial_index arm_name
        0           0_0       0.0  0.0
        1           1_0       1.0  1.0
        >>> print(experiment_data.observation_data)
                                            mean               sem
        metric_signature                        branin branin_map branin branin_map
        trial_index arm_name step
        0           0_0      0.0        55.602113  55.602113    0.0        0.0
                             1.0              NaN  55.602113    NaN        0.0
                             2.0              NaN  55.602113    NaN        0.0
                             3.0              NaN  55.602113    NaN        0.0
        1           1_0      0.0        27.702906  27.702906    0.0        0.0
                             1.0              NaN  27.702906    NaN        0.0
    """

    arm_data: DataFrame
    observation_data: DataFrame

    def filter_by_arm_names(self, arm_names: Iterable[str]) -> ExperimentData:
        """
        Returns a new ``ExperimentData`` object that is filtered to only include
        the rows corresponding to arms in ``arm_names``.
        """
        return ExperimentData(
            arm_data=self.arm_data[
                self.arm_data.index.get_level_values("arm_name").isin(arm_names)
            ],
            observation_data=self.observation_data[
                self.observation_data.index.get_level_values("arm_name").isin(arm_names)
            ],
        )

    def filter_by_trial_index(self, trial_indices: Iterable[int]) -> ExperimentData:
        """
        Returns a new ``ExperimentData`` object that is filtered to only include
        the rows corresponding to trials in ``trial_indices``.
        """
        return ExperimentData(
            arm_data=self.arm_data[
                self.arm_data.index.get_level_values("trial_index").isin(trial_indices)
            ],
            observation_data=self.observation_data[
                self.observation_data.index.get_level_values("trial_index").isin(
                    trial_indices
                )
            ],
        )

    def filter_latest_observations(self) -> ExperimentData:
        """
        Returns a new ``ExperimentData`` object, where the ``observation_data`` is
        filtered to only include the latest (according to map key index) value
        for each ``(trial_index, arm_name)`` pair. The resulting ``observation_data``
        will not have the map keys in its index.

        NOTE: This is an expensive operation. Use sparingly!
        """
        if len(self.observation_data.index.names) == 2:
            # No map key in the observation data. Nothing to filter.
            return deepcopy(self)
        if len(self.observation_data.index.names) != 3:
            raise UnsupportedError(
                "Filtering latest observations is not supported when the index "
                f"includes multiple map keys. Got {self.observation_data.index=}"
            )
        # This sorts each (trial_index, arm_name) group by the map key value,
        # fills NaNs (missing observation for a given progression) in each column
        # with the latest non-null value, and gets the last
        # row, which is now filled with all the latest observations.
        # The resulting dataframe has the highest progression observation for each
        # (trial_index, arm_name) pair for each metric.
        # TODO: See if we can do this more efficiently, maybe by avoiding groupby.apply.
        observation_data = (
            self.observation_data.groupby(
                level=["trial_index", "arm_name"], group_keys=False
            )
            # With map keys, we expect this to be pre-sorted but we can't guarantee.
            .apply(lambda df: df.sort_index(level=2).ffill().tail(1))
            .droplevel(2)  # Remove map key from the index.
        )
        return ExperimentData(
            arm_data=self.arm_data.copy(),
            observation_data=observation_data,
        )

    def convert_to_list_of_observations(self) -> list[Observation]:
        """Converts the ``ExperimentData`` to a list of ``Observation`` objects.

        This is useful for compatibility with some older methods that expect
        the `Adapter` training data to be a list of ``Observation`` objects.
        """
        has_map_keys = self.observation_data.index.nlevels > 2
        observations = []
        # pyre-ignore [23]: Pyre doesn't know the structure of the index.
        for (trial_index, arm_name), row in self.arm_data.iterrows():
            obs_ft_base = ObservationFeatures(
                # NOTE: It is crucial to pop metadata first here.
                # Otherwise, it'd end up in parameters.
                # Copy ensures any changes to the metadata don't affect the original.
                metadata=row.pop("metadata").copy(),
                parameters=row.dropna().to_dict(),
                trial_index=trial_index,
            )
            # Different indexing in the two cases to ensure we get a dataframe.
            ind = (trial_index, arm_name)
            ind = ind if has_map_keys else [ind]
            data_rows = self.observation_data.loc[ind]
            has_multiple_rows = len(data_rows) > 1
            # Looping here to ensure we can capture different progression values
            # as different `Observation` objects.
            for idx in data_rows.index:
                # Keeping it as a df for consistent indexing.
                row_df = data_rows.loc[[idx]]
                # Only include metrics that have data.
                metric_signatures = list(row_df["mean"].dropna(axis="columns").columns)
                if len(metric_signatures) == 0:
                    continue
                if has_multiple_rows:
                    obs_ft = obs_ft_base.clone()
                else:
                    # No need to clone if there is only one row.
                    obs_ft = obs_ft_base
                if has_map_keys:
                    # Add map key to metadata as expected in ObservationFeatures.
                    none_throws(obs_ft.metadata)[data_rows.index.name] = idx
                obs_data = ObservationData(
                    metric_signatures=metric_signatures,
                    means=row_df["mean"][metric_signatures]
                    .to_numpy(copy=True)
                    .reshape(-1),
                    covariance=np.diag(
                        np.square(
                            row_df["sem"][metric_signatures]
                            .to_numpy(copy=True)
                            .reshape(-1)
                        )
                    ),
                )
                observations.append(
                    Observation(features=obs_ft, data=obs_data, arm_name=arm_name)
                )
        return observations

    @property
    def metric_signatures(self) -> list[str]:
        """The list of metric signatures that are available on ``observation_data``."""
        try:
            return list(self.observation_data["mean"].columns)
        except KeyError:
            # No data is available, return empty list.
            return []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentData):
            return False
        return self.arm_data.equals(other.arm_data) and self.observation_data.equals(
            other.observation_data
        )


def extract_experiment_data(
    experiment: Experiment,
    data_loader_config: DataLoaderConfig,
    data: Data | None = None,
) -> ExperimentData:
    """Extract ``ExperimentData`` from the trials & ``Data`` of an experiment.

    Args:
        experiment: The experiment to extract data from.
        data_loader_config: A DataLoaderConfig of options for loading data. See the
            docstring of DataLoaderConfig for more details.
        data: The observation data for the metrics of the experiment.
            Optional, defaults to `experiment.lookup_data()`.
    """
    data = data or experiment.lookup_data()
    arm_data = _extract_arm_data(experiment=experiment)
    observation_data = _extract_observation_data(
        experiment=experiment, data_loader_config=data_loader_config, data=data
    )
    # Filter arm_data to only include the rows in observation_data.
    index = observation_data.index
    if (num_levels := len(index.names)) > 2:
        # Keep only the first two levels: trial_index, arm_name.
        index = index.droplevel(level=list(range(2, num_levels)))
    arm_data = arm_data.loc[arm_data.index.isin(index)]
    return ExperimentData(arm_data=arm_data, observation_data=observation_data)


def _extract_arm_data(experiment: Experiment) -> DataFrame:
    """Extract a dataframe containing the trial index, arm name,
    parameterizations, and metadata from the given experiment.

    The dataframe will include a row for each (trial_index, arm_name) pair,
    as long as the corresponding trial has some data attached to the experiment.

    Args:
        experiment: The experiment to extract arms from.
    """
    records: dict[tuple[int, str], TParameterization] = {}
    for trial_index, trial in experiment.trials.items():
        for arm in trial.arms:
            column_values: dict[str, Any] = arm.parameters
            metadata = trial._get_candidate_metadata(arm.name) or {}
            if Keys.TRIAL_COMPLETION_TIMESTAMP not in metadata:
                if trial._time_completed is not None:
                    metadata[Keys.TRIAL_COMPLETION_TIMESTAMP] = (
                        trial._time_completed
                    ).timestamp()
            column_values["metadata"] = metadata
            records[(trial_index, arm.name)] = column_values
    if records:
        df = DataFrame.from_dict(records, orient="index")
        df.index.names = ["trial_index", "arm_name"]
    else:
        # No data, return an empty dataframe with the correct index & columns.
        index = MultiIndex.from_tuples([], names=["trial_index", "arm_name"])
        df = DataFrame(index=index, columns=list(experiment.parameters) + ["metadata"])
    return df


def _extract_observation_data(
    experiment: Experiment,
    data_loader_config: DataLoaderConfig,
    data: Data | None = None,
) -> DataFrame:
    """Extracts a dataframe containing filtered observations for the metrics
    on the experiment.

    See `extract_experiment_data` for a description of the arguments.

    Returns:
        A dataframe filtered to only include observations from the given statuses
        to include, and pivoted to be indexed by (trial_index, arm_name, *map_keys)
        and to have columns "mean" & "sem" for each metric.
        If `data` contains additional metadata columns like `start_time` and `end_time`
        they will be added to the pivoted dataframe as additional columns. The columns
        are labeled hierarchically (with a ``pd.MultiIndex`` structure) where the top
        level is "metadata" and the lower level keeps the original column name.
        For example, if the original DataFrame from ``Data`` includes a column
        ``start_time``, then ``observation_data`` will have a column that can be
        retrived with ``("metadata", "start_time")``.
    """
    data = data if data is not None else experiment.lookup_data()
    if data.has_step_column:
        if data_loader_config.latest_rows_per_group is not None:
            data = data.latest(
                rows_per_group=data_loader_config.latest_rows_per_group,
            )
        elif (
            data_loader_config.limit_rows_per_metric is not None
            or data_loader_config.limit_rows_per_group is not None
        ):
            data = data.subsample(
                limit_rows_per_metric=data_loader_config.limit_rows_per_metric,
                limit_rows_per_group=data_loader_config.limit_rows_per_group,
                include_first_last=True,
            )

    df = data.full_df
    df = _maybe_normalize_map_key(df)
    # Filter out rows for invalid statuses.
    to_keep = Series(index=df.index, data=False)
    trial_statuses = df["trial_index"].map(
        {trial_index: trial.status for trial_index, trial in experiment.trials.items()}
    )
    # If there are abandoned arms, mark the corresponding rows as abandoned.
    abandoned_arms = {
        i: {arm.name for arm in trial.abandoned_arms}
        for i, trial in experiment.trials.items()
        if trial.abandoned_arms
    }
    if abandoned_arms:
        is_abandoned = df[["trial_index", "arm_name"]].apply(
            lambda row: row["trial_index"] in abandoned_arms
            and row["arm_name"] in abandoned_arms[row["trial_index"]],
            axis=1,
        )
        trial_statuses[is_abandoned] = TrialStatus.ABANDONED
    # Check against valid statuses and filter the rows.
    for metric in experiment.metrics.values():
        valid_statuses = (
            data_loader_config.statuses_to_fit_map_metric
            if isinstance(metric, MapMetric) and metric.has_map_data
            else data_loader_config.statuses_to_fit
        )
        to_keep |= (df["metric_signature"] == metric.signature) & trial_statuses.isin(
            valid_statuses
        )
    df = df.loc[to_keep]
    # If df is empty, add mean & sem columns to facilitate pivoting.
    if df.empty:
        df = df.assign(mean=None, sem=None)

    # Identify potential metadata columns.
    index_cols = ["trial_index", "arm_name"]
    if data.has_step_column:
        index_cols.append(MAP_KEY)

    standard_columns = set(index_cols).union(
        {"metric_name", "metric_signature", "mean", "sem"}
    )
    metadata_columns = [col for col in df.columns if col not in standard_columns]

    # Pivot the df to be indexed by (trial_index, arm_name, *map_key)
    # and to have columns "mean" & "sem" for each metric.
    observation_data = df.pivot(
        columns="metric_signature", index=index_cols, values=["mean", "sem"]
    )

    # If metadata columns exist, add them to the pivoted dataframe.
    if metadata_columns:
        # Create a dataframe with just the index columns and metadata columns.
        metadata_df = df[index_cols + metadata_columns]
        # Set the index to match the observation_data index.
        # All null rows are dropped to still capture the metadata in the next step
        # if it exists for only a subset of metrics.
        metadata_df = metadata_df.set_index(index_cols).dropna(how="all")
        # Drop duplicates to ensure we have only one row per unique index.
        # This is necessary when there are multiple metrics.
        metadata_df = metadata_df.loc[~metadata_df.index.duplicated(keep="first")]
        # Create a multi-index for the columns to facilitate the merge.
        metadata_df.columns = MultiIndex.from_product([["metadata"], metadata_columns])
        # Join to the main dataframe.
        observation_data = observation_data.join(metadata_df)

    return observation_data


def _maybe_normalize_map_key(df: DataFrame) -> DataFrame:
    """Normalizes the MAP_KEY column in the given dataframe.
    For each metric, the values will be normalized to [0, 1] using the
    largest & smallest values for that metric. If a metric has constant
    values for MAP_KEY, it will be normalized to 1.0. Any NaN MAP_KEY
    values will not be modified.

    Args:
        df: The dataframe to normalize the MAP_KEY column in.

    Returns:
        The normalized dataframe.
    """
    if len(df) == 0 or MAP_KEY not in df or df[MAP_KEY].isna().all():
        return df
    df = df.copy()  # Don't modify the original dataframe.
    not_na = ~df[MAP_KEY].isna()
    for metric in df["metric_signature"].unique():
        # Only non-NaN rows corresponding to the particular metric.
        mask = (df["metric_signature"] == metric) & not_na
        map_values = df.loc[mask, MAP_KEY]
        if len(map_values) == 0:
            continue
        unique_values = map_values.unique()
        if len(unique_values) == 1:
            df.loc[mask, MAP_KEY] = 1.0
            continue
        # Get the min and max values and use them to normalize to [0, 1].
        map_key_min = unique_values.min()
        map_key_max = unique_values.max()
        df.loc[mask, MAP_KEY] = (map_values - map_key_min) / (map_key_max - map_key_min)
    return df
