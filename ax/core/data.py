#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from functools import reduce
from hashlib import md5
from typing import Any, Dict, Iterable, Optional, Set, Type, Union

import numpy as np
import pandas as pd
from ax.core.types import TFidelityTrialEvaluation, TTrialEvaluation
from ax.utils.common.base import Base
from ax.utils.common.serialization import (
    extract_init_args,
    SerializationMixin,
    serialize_init_args,
)
from ax.utils.common.typeutils import checked_cast, not_none


class Data(Base, SerializationMixin):
    """Class storing data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        df: DataFrame with underlying data, and required columns.
        description: Human-readable description of data.

    """

    # Note: Although the SEM (standard error of the mean) is a required column in data,
    # downstream models can infer missing SEMs. Simply specify NaN as the SEM value,
    # either in your Metric class or in Data explicitly.
    REQUIRED_COLUMNS = {"arm_name", "metric_name", "mean", "sem"}

    COLUMN_DATA_TYPES = {
        "arm_name": str,
        "metric_name": str,
        "mean": np.float64,
        "sem": np.float64,
        "trial_index": np.int64,
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "n": np.int64,
        "frac_nonnull": np.float64,
        "random_split": np.int64,
        "fidelities": str,  # Dictionary stored as json
    }

    _df: pd.DataFrame

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
    ) -> None:
        """Init Data.

        Args:
            df: DataFrame with underlying data, and required columns.
            description: Human-readable description of data.

        """
        # Initialize with barebones DF.
        if df is None:
            self._df = pd.DataFrame(columns=list(self.required_columns()))
        else:
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            extra_columns = columns - self.supported_columns()
            if extra_columns:
                raise ValueError(f"Columns {list(extra_columns)} are not supported.")
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            df = self._safecast_df(df=df)

            # Reorder the columns for easier viewing
            col_order = [c for c in self.column_data_types() if c in df.columns]
            self._df = df[col_order]

        self.description = description

    @classmethod
    def _safecast_df(
        cls,
        df: pd.DataFrame,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        extra_column_types: Optional[Dict[str, Type]] = None,
    ) -> pd.DataFrame:
        """Function for safely casting df to standard data types.

        Needed because numpy does not support NaNs in integer arrays.

        Allows `Any` to be specified as a type, and will skip casting for that column.

        Args:
            df: DataFrame to safe-cast.
            extra_column_types: types of columns only specified at instantiation-time.

        Returns:
            safe_df: DataFrame cast to standard dtypes.

        """
        extra_column_types = extra_column_types or {}
        dtype = {
            # Pandas timestamp handlng is weird
            col: "datetime64[ns]" if coltype is pd.Timestamp else coltype
            for col, coltype in cls.column_data_types(
                extra_column_types=extra_column_types
            ).items()
            if col in df.columns.values
            and not (
                cls.column_data_types(extra_column_types)[col] is np.int64
                and df.loc[:, col].isnull().any()
            )
            and not (coltype is Any)
        }

        return checked_cast(pd.DataFrame, df.astype(dtype=dtype))

    @classmethod
    def required_columns(cls) -> Set[str]:
        """Names of columns that must be present in the underlying ``DataFrame``."""
        return cls.REQUIRED_COLUMNS

    @classmethod
    def supported_columns(
        cls, extra_column_names: Optional[Iterable[str]] = None
    ) -> Set[str]:
        """Names of columns supported (but not necessarily required) by this class."""
        extra_column_names = set(extra_column_names or [])
        extra_column_types: Dict[str, Any] = {name: Any for name in extra_column_names}
        return cls.REQUIRED_COLUMNS.union(
            cls.column_data_types(extra_column_types=extra_column_types)
        )

    @classmethod
    def column_data_types(
        cls,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        extra_column_types: Optional[Dict[str, Type]] = None,
        excluded_columns: Optional[Iterable[str]] = None,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
    ) -> Dict[str, Type]:
        """Type specification for all supported columns."""
        extra_column_types = extra_column_types or {}
        excluded_columns = excluded_columns or []

        columns = {**cls.COLUMN_DATA_TYPES, **extra_column_types}

        for column in excluded_columns:
            if column in columns:
                del columns[column]

        return columns

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        """Serialize the class-dependent properties needed to initialize this Data.
        Used for storage and to help construct new similar Data.
        """
        data = checked_cast(Data, obj)
        return serialize_init_args(object=data)

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.
        """
        # Extract `df` only if present, since certain inputs to this fn, e.g.
        # SQAData.structure_metadata_json, don't have a `df` attribute.
        if "df" in args and not isinstance(args["df"], pd.DataFrame):
            # NOTE: Need dtype=False, otherwise infers arm_names like
            # "4_1" should be int 41.
            args["df"] = pd.read_json(args["df"]["value"], dtype=False)
        return extract_init_args(args=args, class_=cls)

    @property
    def true_df(self) -> pd.DataFrame:
        """Return the `DataFrame` being used as the source of truth (avoid using
        except for caching).
        """

        return self._df

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
        return md5(not_none(self.df.to_json()).encode("utf-8")).hexdigest()

    @property
    def metric_names(self) -> Set[str]:
        """Set of metric names that appear in the underlying dataframe of
        this object.
        """
        return set() if self.df.empty else set(self.df["metric_name"].values)

    def filter(
        self,
        trial_indices: Optional[Iterable[int]] = None,
        metric_names: Optional[Iterable[str]] = None,
    ) -> Data:
        """Construct a new Data object with the subset of rows corresponding to the
        provided trial indices AND metric names. If either trial_indices or
        metric_names are not provided, that dimension will not be filtered.
        """

        return Data(
            df=self._filter_df(
                df=self.df, trial_indices=trial_indices, metric_names=metric_names
            )
        )

    @staticmethod
    def _filter_df(
        df: pd.DataFrame,
        trial_indices: Optional[Iterable[int]] = None,
        metric_names: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        trial_indices_mask = (
            reduce(
                lambda left, right: left | right,
                [df["trial_index"] == trial_index for trial_index in trial_indices],
            )
            if trial_indices is not None
            else pd.Series([True] * len(df))
        )

        metric_names_mask = (
            reduce(
                lambda left, right: left | right,
                [df["metric_name"] == metric_name for metric_name in metric_names],
            )
            if metric_names is not None
            else pd.Series([True] * len(df))
        )

        return df.loc[trial_indices_mask & metric_names_mask]

    def get_filtered_results(self, **filters: Dict[str, Any]) -> pd.DataFrame:
        """Return filtered subset of data.

        Args:
            filter: Column names and values they must match.

        Returns
            df: The filtered DataFrame.
        """
        df = self.df.copy()
        if df.empty:
            return df

        columns = df.columns
        for colname, value in filters.items():
            if colname not in columns:
                raise ValueError(
                    f"{colname} not in the set of columns: {columns}"
                    f"in this data object of type: {str(type(self))}."
                )
            df = df[df[colname] == value]
        return df

    @staticmethod
    def from_multiple_data(
        data: Iterable[Data], subset_metrics: Optional[Iterable[str]] = None
    ) -> Data:
        """Combines multiple data objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax `Data` objects to combine.
            subset_metrics: If specified, combined `Data` will only contain
                metrics, names of which appear in this iterable,
                in the underlying dataframe.
        """
        dfs = [datum.df for datum in data]

        if len(dfs) == 0:
            return Data()

        if subset_metrics:
            dfs = [df.loc[df["metric_name"].isin(subset_metrics)] for df in dfs]

        return Data(df=pd.concat(dfs, axis=0, sort=True))

    @staticmethod
    def from_evaluations(
        evaluations: Dict[str, TTrialEvaluation],
        trial_index: int,
        sample_sizes: Optional[Dict[str, int]] = None,
        start_time: Optional[Union[int, str]] = None,
        end_time: Optional[Union[int, str]] = None,
    ) -> Data:
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to metric outcomes, which itself is  a
                mapping of metric names to means or tuples of mean and an SEM). If
                SEM is not specified, it will be set to None and inferred from data.
            trial_index: Trial index to which this data belongs.
            sample_sizes: Number of samples collected for each arm.
            start_time: Optional start time of run of the trial that produced this
                data, in milliseconds or iso format.  Milliseconds will be automatically
                converted to iso format because iso format automatically works with the
                pandas column type `Timestamp`.
            end_time: Optional end time of run of the trial that produced this
                data, in milliseconds or iso format.  Milliseconds will be automatically
                converted to iso format because iso format automatically works with the
                pandas column type `Timestamp`.

        Returns:
            Ax Data object.
        """
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else None,
                "trial_index": trial_index,
            }
            for name, evaluation in evaluations.items()
            for metric_name, value in evaluation.items()
        ]
        if start_time is not None or end_time is not None:
            if isinstance(start_time, int):
                start_time = _ms_epoch_to_isoformat(start_time)
            if isinstance(end_time, int):
                end_time = _ms_epoch_to_isoformat(end_time)

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
    ) -> Data:
        """
        Convert dict of fidelity evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to list of (fidelity, metric outcomes)
                where metric outcomes is itself a mapping of metric names to means
                or tuples of mean and SEM. If SEM is not specified, it will be set
                to None and inferred from data.
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
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else None,
                "trial_index": trial_index,
                "fidelities": json.dumps(fidelity),
            }
            for name, fidelity_and_metrics_list in evaluations.items()
            for fidelity, evaluation in fidelity_and_metrics_list
            for metric_name, value in evaluation.items()
        ]
        if start_time is not None or end_time is not None:
            for record in records:
                record.update({"start_time": start_time, "end_time": end_time})
        if sample_sizes:
            for record in records:
                record["n"] = sample_sizes[str(record["arm_name"])]
        return Data(df=pd.DataFrame(records))

    def copy_structure_with_df(self, df: pd.DataFrame) -> Data:
        """Serialize the structural properties needed to initialize this Data.
        Used for storage and to help construct new similar Data. All kwargs
        other than ``df`` and ``description`` are considered structural.
        """
        cls = type(self)
        return cls(df=df, **cls.serialize_init_args(self))


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


def _ms_epoch_to_isoformat(epoch: int) -> str:
    return pd.Timestamp(epoch, unit="ms").isoformat()


def custom_data_class(
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
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
        @classmethod
        def required_columns(cls) -> Set[str]:
            return (required_columns or set()).union(Data.REQUIRED_COLUMNS)

        @classmethod
        def column_data_types(
            cls, extra_column_types: Optional[Dict[str, Type]] = None
        ) -> Dict[str, Type]:
            return super().column_data_types(
                {**(extra_column_types or {}), **(column_data_types or {})}
            )

    return CustomData
