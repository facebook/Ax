#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from hashlib import md5
from typing import Dict, Iterable, Optional, Set, Type

import numpy as np
import pandas as pd
from ax.core.base import Base
from ax.core.types import TFidelityTrialEvaluation, TTrialEvaluation


TPdTimestamp = pd.Timestamp

COLUMN_DATA_TYPES = {
    "arm_name": str,
    "metric_name": str,
    "mean": np.float64,
    "sem": np.float64,
    "trial_index": np.int64,
    "start_time": TPdTimestamp,
    "end_time": TPdTimestamp,
    "n": np.int64,
    "frac_nonnull": np.float64,
    "random_split": np.int64,
    "fidelities": str,  # Dictionary stored as json
}
REQUIRED_COLUMNS = {"arm_name", "metric_name", "mean"}


class Data(Base):
    """Class storing data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        df: DataFrame with underlying data, and required columns.
        description: Human-readable description of data.

    """

    def __init__(
        self, df: Optional[pd.DataFrame] = None, description: Optional[str] = None
    ) -> None:
        """Init Data.

        Args:
            df: DataFrame with underlying data, and required columns.
            description: Human-readable description of data.

        """
        # Initialize with barebones DF.
        if df is None:
            self._df = pd.DataFrame(columns=self.required_columns())
        else:
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            extra_columns = columns - set(self.column_data_types())
            if extra_columns:
                raise ValueError(f"Columns {list(extra_columns)} are not supported.")
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            df = self._safecast_df(df=df)

            # Reorder the columns for easier viewing
            col_order = [c for c in self.column_data_types() if c in df.columns]
            self._df = df[col_order]
        self.description = description

    @classmethod
    def _safecast_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Function for safely casting df to standard data types.

        Needed because numpy does not support NaNs in integer arrays.

        Args:
            df: DataFrame to safe-cast.

        Returns:
            safe_df: DataFrame cast to standard dtypes.

        """
        dtype = {
            # Pandas timestamp handlng is weird
            col: "datetime64[ns]" if coltype is TPdTimestamp else coltype
            for col, coltype in cls.column_data_types().items()
            if col in df.columns.values
            and not (
                cls.column_data_types()[col] is np.int64
                and df.loc[:, col].isnull().any()
            )
        }
        return df.astype(dtype=dtype)

    @staticmethod
    def required_columns() -> Set[str]:
        """Names of required columns."""
        return REQUIRED_COLUMNS

    @staticmethod
    def column_data_types() -> Dict[str, Type]:
        """Type specification for all supported columns."""
        return COLUMN_DATA_TYPES

    @staticmethod
    def from_multiple_data(data: Iterable["Data"]) -> "Data":
        dfs = [datum.df for datum in data]
        if len(dfs) == 0:
            return Data()
        return Data(df=pd.concat(dfs, axis=0, sort=True))

    @staticmethod
    def from_evaluations(
        evaluations: Dict[str, TTrialEvaluation],
        trial_index: int,
        sample_sizes: Optional[Dict[str, int]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> "Data":
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to metric outcomes (itself a mapping
                of metric names to tuples of mean and optionally a SEM).
            trial_index: Trial index to which this data belongs.
            sample_sizes: Number of samples collected for each arm.
            start_time: Optional start time of run of the trial that produced this
                data, in milliseconds.
            end_time: Optional end time of run of the trial that produced this
                data, in milliseconds.

        Returns:
            Ax Data object.
        """
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": evaluation[metric_name][0],
                "sem": evaluation[metric_name][1],
                "trial_index": trial_index,
            }
            for name, evaluation in evaluations.items()
            # pyre-fixme[10]: Name `evaluation` is used but not defined.
            for metric_name in evaluation.keys()
        ]
        if start_time is not None or end_time is not None:
            for record in records:
                record.update({"start_time": start_time, "end_time": end_time})
        if sample_sizes:
            for record in records:
                record["n"] = sample_sizes[str(record["arm_name"])]
        return Data(df=pd.DataFrame(records))

    @staticmethod
    def from_fidelity_evaluations(
        evaluations: Dict[str, TFidelityTrialEvaluation],
        trial_index: int,
        sample_sizes: Optional[Dict[str, int]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> "Data":
        """
        Convert dict of fidelity evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to list of (fidelity, metric outcomes)
                (where metric outcomes is itself a mapping of metric names to
                tuples of mean and SEM).
            trial_index: Trial index to which this data belongs.
            sample_sizes: Number of samples collected for each arm.
            start_time: Optional start time of run of the trial that produced this
                data, in milliseconds.
            end_time: Optional end time of run of the trial that produced this
                data, in milliseconds.

        Returns:
            Ax Data object.
        """
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": evaluation[metric_name][0],
                "sem": evaluation[metric_name][1],
                "trial_index": trial_index,
                "fidelities": json.dumps(fidelity),
            }
            for name, fidelity_and_metrics_list in evaluations.items()
            # pyre-fixme[10]: Name `fidelity_and_metrics_list` is used but not defined.
            for fidelity, evaluation in fidelity_and_metrics_list
            for metric_name in evaluation.keys()
        ]
        if start_time is not None or end_time is not None:
            for record in records:
                record.update({"start_time": start_time, "end_time": end_time})
        if sample_sizes:
            for record in records:
                record["n"] = sample_sizes[str(record["arm_name"])]
        return Data(df=pd.DataFrame(records))

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def df_hash(self) -> str:
        """Compute hash of pandas DataFrame.

        This first serializes the DataFrame and computes the md5 hash on the
        resulting string. Note that this may cause performance issue for very large
        DataFrames.

        Args:
            df: The DataFrame for which to compute the hash.

        Returns
            str: The hash of the DataFrame.

        """
        return md5(self.df.to_json().encode("utf-8")).hexdigest()  # pyre-ignore


def set_single_trial(data: Data) -> Data:
    """Returns a new Data object where we set all rows to have the same
    trial index (i.e. 0). This is meant to be used with our IVW transform,
    which will combine multiple observations of the same metric.
    """
    df = data._df.copy()
    if "trial_index" in df:
        df["trial_index"] = 0
    return Data(df=df)


def clone_without_metrics(data: Data, excluded_metric_names: Iterable[str]) -> Data:
    """Returns a new data object where rows containing the metrics specified by
    `metric_names` are filtered out. Used to sanitize data before using it as
    training data for a model that requires data rectangularity.

    Args:
        data: Original data to clone.
        excluded_metric_names: Metrics to avoid copying

    Returns:
        new version of the original data without specified metrics.
    """
    return Data(
        df=data.df[
            data.df["metric_name"].apply(lambda n: n not in excluded_metric_names)
        ].copy()
    )


def custom_data_class(
    column_data_types: Optional[Dict[str, Type]] = None,
    required_columns: Optional[Set[str]] = None,
    time_columns: Optional[Set[str]] = None,
) -> Type[Data]:
    """Creates a custom data class with additional columns.

    All columns and their designations on the base data class are preserved,
    the inputs here are appended to the definitions on the base class.

    Args:
        column_data_types: Dict from column name to column type.
        required_columns: Set of additional columns required for this data object.
        time_columns: Set of additional columns to cast to timestamp.

    Returns:
        New data subclass with amended column definitions.
    """

    class CustomData(Data):
        @staticmethod
        def required_columns() -> Set[str]:
            return (required_columns or set()).union(REQUIRED_COLUMNS)

        @staticmethod
        def column_data_types() -> Dict[str, Type]:
            return {**(column_data_types or {}), **COLUMN_DATA_TYPES}

    return CustomData
