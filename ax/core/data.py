#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from io import StringIO
from typing import Any, cast, TypeVar

import numpy as np
import pandas as pd
from ax.core.types import TTrialEvaluation
from ax.utils.common.base import Base
from ax.utils.common.serialization import (
    extract_init_args,
    SerializationMixin,
    serialize_init_args,
    TClassDecoderRegistry,
    TDecoderRegistry,
)
from pyre_extensions import assert_is_instance

TData = TypeVar("TData", bound="Data")
DF_REPR_MAX_LENGTH = 1000


class Data(Base, SerializationMixin):
    """Class storing numerical data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        df: DataFrame with underlying data, and required columns. For Data, the
            required columns are "arm_name", "metric_name", "mean", and "sem", the
            latter two of which must be numeric.
        description: Human-readable description of data.

    """

    # Note: Although the SEM (standard error of the mean) is a required column,
    # downstream models can infer missing SEMs. Simply specify NaN as the SEM value,
    # either in your Metric class or in Data explicitly.
    REQUIRED_COLUMNS = {"trial_index", "arm_name", "metric_name", "mean", "sem"}

    # Note on text data: https://pandas.pydata.org/docs/user_guide/text.html
    # Its type can either be `numpy.dtypes.ObjectDType` or StringDtype extension
    # type; the later is still experimental. So we are using object.
    COLUMN_DATA_TYPES: dict[str, Any] = {
        # Ubiquitous columns.
        "trial_index": int,
        "arm_name": np.dtype("O"),
        # Metric data-related columns.
        "metric_name": np.dtype("O"),
        "mean": np.float64,
        "sem": np.float64,
        # Metadata columns available for all subclasses.
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "n": int,
        # Metadata columns available for only some subclasses.
        "frac_nonnull": np.float64,
        "random_split": int,
        "fidelities": np.dtype("O"),  # Dictionary stored as json
    }

    _df: pd.DataFrame

    def __init__(
        self: TData,
        df: pd.DataFrame | None = None,
        description: str | None = None,
        _skip_ordering_and_validation: bool = False,
    ) -> None:
        """Initialize a ``Data`` object from the given DataFrame.

        Args:
            df: DataFrame with underlying data, and required columns.
            description: Human-readable description of data.
            _skip_ordering_and_validation: If True, uses the given DataFrame
                as is, without ordering its columns or validating its contents.
                Intended only for use in `Data.filter`, where the contents
                of the DataFrame are known to be ordered and valid.
        """
        if df is None:
            # Initialize with barebones DF.
            self._df = pd.DataFrame(columns=list(self.required_columns()))
        elif _skip_ordering_and_validation:
            self._df = df
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
            df = df.dropna(axis=0, how="all", ignore_index=True)
            df = self._safecast_df(df=df)

            # Reorder the columns for easier viewing
            col_order = [c for c in self.column_data_types() if c in df.columns]
            self._df = df.reindex(columns=col_order, copy=False)

        self.description = description

    @classmethod
    def _safecast_df(
        cls: type[TData],
        df: pd.DataFrame,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        extra_column_types: Mapping[str, type] | None = None,
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
        dtypes = df.dtypes
        for col, coltype in cls.column_data_types(
            extra_column_types=extra_column_types
        ).items():
            if col in df.columns.values and coltype is not Any:
                # Pandas timestamp handlng is weird
                dtype = "datetime64[ns]" if coltype is pd.Timestamp else coltype
                if (dtype != dtypes[col]) and not (
                    coltype is int and df.loc[:, col].isnull().any()
                ):
                    df[col] = df[col].astype(dtype)
        return df

    def required_columns(self) -> set[str]:
        """Names of columns that must be present in the underlying ``DataFrame``."""
        return self.REQUIRED_COLUMNS

    @classmethod
    def supported_columns(
        cls, extra_column_names: Iterable[str] | None = None
    ) -> set[str]:
        """Names of columns supported (but not necessarily required) by this class."""
        extra_column_names = set(extra_column_names or [])
        extra_column_types: dict[str, Any] = {name: Any for name in extra_column_names}
        return cls.REQUIRED_COLUMNS.union(
            cls.column_data_types(extra_column_types=extra_column_types)
        )

    @classmethod
    def column_data_types(
        cls,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        extra_column_types: Mapping[str, type] | None = None,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
    ) -> dict[str, type]:
        """Type specification for all supported columns."""
        extra_column_types = extra_column_types or {}
        return {**cls.COLUMN_DATA_TYPES, **extra_column_types}

    @classmethod
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        """Serialize the class-dependent properties needed to initialize this Data.
        Used for storage and to help construct new similar Data.
        """
        data = assert_is_instance(obj, cls)
        return serialize_init_args(
            obj=data, exclude_fields=["_skip_ordering_and_validation"]
        )

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        class_decoder_registry: TClassDecoderRegistry | None = None,
    ) -> dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the object.
        Used for storage.
        """
        # Extract `df` only if present, since certain inputs to this fn, e.g.
        # SQAData.structure_metadata_json, don't have a `df` attribute.
        if "df" in args and not isinstance(args["df"], pd.DataFrame):
            # NOTE: Need dtype=False, otherwise infers arm_names like
            # "4_1" should be int 41.
            args["df"] = pd.read_json(StringIO(args["df"]["value"]), dtype=False)
        return extract_init_args(args=args, class_=cls)

    @property
    def true_df(self) -> pd.DataFrame:
        """Return the ``DataFrame`` being used as the source of truth (avoid using
        except for caching).
        """
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def get_filtered_results(self, **filters: Any) -> pd.DataFrame:
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
    def from_multiple(data: Iterable[TData]) -> TData:
        """Combines multiple objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax objects of this class to combine.
        """
        dfs = []

        cls = None

        for datum in data:
            if cls is None:
                cls = type(datum)
            if type(datum) is not cls:
                raise TypeError(
                    f"All data objects must be instances of the same class. Got "
                    f"{cls} and {type(datum)}."
                )
            if len(datum.df) > 0:
                dfs.append(datum.df)

        cls = cls or cast(type[TData], Data)

        if len(dfs) == 0:
            return cls()

        return cls(df=pd.concat(dfs, axis=0, sort=True))

    @classmethod
    def from_evaluations(
        cls: type[TData],
        evaluations: Mapping[str, TTrialEvaluation],
        trial_index: int,
        sample_sizes: Mapping[str, int] | None = None,
        start_time: int | str | None = None,
        end_time: int | str | None = None,
    ) -> TData:
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to outcomes, which itself is a mapping of
                outcome names to values, means, or tuples of mean and SEM. If SEM is
                not specified, it will be set to None and inferred from data.
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
            Ax object of the enclosing class.
        """
        records = cls._get_records(evaluations=evaluations, trial_index=trial_index)
        records = cls._add_cols_to_records(
            records=records,
            sample_sizes=sample_sizes,
            start_time=start_time,
            end_time=end_time,
        )
        return cls(df=pd.DataFrame(records))

    @staticmethod
    def _add_cols_to_records(
        records: list[dict[str, Any]],
        sample_sizes: Mapping[str, int] | None = None,
        start_time: int | str | None = None,
        end_time: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """Adds to records metadata columns that are available for all
        Data subclasses.
        """
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

        return records

    def __repr__(self) -> str:
        """String representation of the subclass, inheriting from this base."""
        df_markdown = self.df.to_markdown()
        if len(df_markdown) > DF_REPR_MAX_LENGTH:
            df_markdown = df_markdown[:DF_REPR_MAX_LENGTH] + "..."
        return f"{self.__class__.__name__}(df=\n{df_markdown})"

    @staticmethod
    def _get_records(
        evaluations: Mapping[str, TTrialEvaluation], trial_index: int
    ) -> list[dict[str, Any]]:
        return [
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

    @property
    def metric_names(self) -> set[str]:
        """Set of metric names that appear in the underlying dataframe of
        this object.
        """
        return set() if self.df.empty else set(self.df["metric_name"].values)

    def filter(
        self: TData,
        trial_indices: Iterable[int] | None = None,
        metric_names: Iterable[str] | None = None,
    ) -> TData:
        """Construct a new object with the subset of rows corresponding to the
        provided trial indices AND metric names. If either trial_indices or
        metric_names are not provided, that dimension will not be filtered.
        """

        return self.__class__(
            df=_filter_df(
                df=self.df, trial_indices=trial_indices, metric_names=metric_names
            ),
            _skip_ordering_and_validation=True,
        )

    @classmethod
    def from_multiple_data(
        cls, data: Iterable[Data], subset_metrics: Iterable[str] | None = None
    ) -> Data:
        """Combines multiple objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax objects of this class to combine.
            subset_metrics: If specified, combined object will only contain
                metrics, names of which appear in this iterable,
                in the underlying dataframe.
        """
        data_out = cls.from_multiple(data=data)
        if len(data_out.df.index) == 0:
            return data_out
        if subset_metrics:
            data_out._df = data_out.df.loc[
                data_out.df["metric_name"].isin(subset_metrics)
            ]

        return data_out

    def clone(self) -> Data:
        """Returns a new Data object with the same underlying dataframe."""
        return Data(df=deepcopy(self.df), description=self.description)


def clone_without_metrics(data: Data, excluded_metric_names: Iterable[str]) -> Data:
    """Returns a new data object where rows containing the outcomes specified by
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
    column_data_types: Mapping[str, type] | None = None,
    required_columns: set[str] | None = None,
    time_columns: set[str] | None = None,
) -> type[Data]:
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
        def required_columns(cls) -> set[str]:
            return (required_columns or set()).union(Data.REQUIRED_COLUMNS)

        @classmethod
        def column_data_types(
            cls, extra_column_types: Mapping[str, type] | None = None
        ) -> dict[str, type]:
            return super().column_data_types(
                {**(extra_column_types or {}), **(column_data_types or {})}
            )

    return CustomData


def _filter_df(
    df: pd.DataFrame,
    trial_indices: Iterable[int] | None = None,
    metric_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter rows of a dataframe by trial indices and metric names."""
    if trial_indices is not None:
        # Trial indices is not None, metric names is not yet known.
        trial_indices_mask = df["trial_index"].isin(trial_indices)
        if metric_names is None:
            # If metric names is None, we can filter & return.
            return df.loc[trial_indices_mask]
        # Both are given, filter by both.
        metric_names_mask = df["metric_name"].isin(metric_names)
        return df.loc[trial_indices_mask & metric_names_mask]
    if metric_names is not None:
        # Trial indices is None, metric names is not None.
        metric_names_mask = df["metric_name"].isin(metric_names)
        return df.loc[metric_names_mask]
    # Both are None, return the dataframe as is.
    return df
