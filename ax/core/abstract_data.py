#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Type

import numpy as np
import pandas as pd
from ax.utils.common.base import Base
from ax.utils.common.serialization import serialize_init_args, extract_init_args

if TYPE_CHECKING:
    from ax.core.observation import ObservationData


class AbstractData(ABC, Base):
    """Abstract Base Class for storing data for an experiment."""

    def __init__(
        self,
        description: Optional[str] = None,
    ) -> None:
        """Init Data.

        Args:
            description: Human-readable description of data.

        """
        self.description = description

    @staticmethod
    @abstractmethod
    def from_multiple_data(
        data: Iterable[AbstractData], subset_metrics: Optional[Iterable[str]] = None
    ) -> AbstractData:
        """Combines multiple data objects into one (with the concatenated
        underlying dataframe).

        NOTE: if one or more data objects in the iterable is of a custom
        subclass of this Data type, object of that class will be returned. If
        the iterable contains incompatible types of `Data`, an error will be
        raised.

        Args:
            data: Iterable of data objects to combine.
            subset_metrics: If specified, combined data will only contain
                metrics that appear in this iterable.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def metric_names(self) -> Set[str]:
        """Set of metrics contained in this data."""
        pass  # pragma: no cover

    @abstractmethod
    def to_observation_data(self) -> List["ObservationData"]:
        """Convert to ObservationData"""
        pass  # pragma: no cover


class AbstractDataFrameData(AbstractData, Base):
    """Abstract Base Class for storing `DataFrame`-backed Data for an experiment.

    Attributes:
        df: DataFrame with underlying data, and required columns.
        description: Human-readable description of data.

    """

    REQUIRED_COLUMNS = {}
    COLUMN_DATA_TYPES = {
        "arm_name": str,
        "metric_name": str,
        "mean": np.float64,
        "sem": np.float64,
        "trial_index": np.int64,
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
    ) -> None:
        """Init Data.

        Args:
            description: Human-readable description of data.

        """
        super().__init__(description=description)

    @classmethod
    def _safecast_df(
        cls, df: pd.DataFrame, extra_column_types: Optional[Dict[str, Type]] = None
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
        # pyre-fixme[7]: Expected `DataFrame` but got
        #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
        return df.astype(dtype=dtype)

    @classmethod
    def required_columns(cls) -> Set[str]:
        """Names of required columns."""
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
        cls, extra_column_types: Optional[Dict[str, Type]] = None
    ) -> Dict[str, Type]:
        """Type specification for all supported columns."""
        extra_column_types = extra_column_types or {}
        return {**cls.COLUMN_DATA_TYPES, **extra_column_types}

    @property
    def df(self) -> pd.DataFrame:
        """Return a flattened `DataFrame` representation of this data's metrics."""
        # pyre-ignore [16]: Undefined attribute. _df will be defined in subclasses.
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
        # pyre-fixme[16]: `Optional` has no attribute `encode`.
        return md5(self.df.to_json().encode("utf-8")).hexdigest()

    @property
    def metric_names(self) -> Set[str]:
        """Set of metric names that appear in the underlying dataframe of
        this object.
        """
        return set() if self.df.empty else set(self.df["metric_name"].values)

    def get_filtered_results(self, **filters: Dict[str, Any]) -> pd.DataFrame:
        """Return filtered subset of data.

        Args:
            filter: Column names and values they must match.

        Returns
            df: The filtered DataFrame.
        """
        df = self.df.copy()
        columns = df.columns
        for colname, value in filters.items():
            if colname not in columns:
                raise ValueError(
                    f"{colname} not in the set of columns: {columns}"
                    f"in this data object of type: {str(type(self))}."
                )
            df = df[df[colname] == value]
        return df

    def to_observation_data(self) -> List["ObservationData"]:
        """Convert to ObservationData"""
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def serialize_init_args(cls, data: AbstractDataFrameData) -> Dict[str, Any]:
        """Serialize the class-dependent properties needed to initialize this Data.
        Used for storage and to help construct new similar Data. All kwargs
        other than "dataframe" and "description" are considered structural.
        """
        return serialize_init_args(object=data, exclude_fields=["df", "description"])

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.
        """
        return extract_init_args(args=args, class_=cls)

    def copy_structure_with_df(self, df: pd.DataFrame) -> AbstractDataFrameData:
        """Serialize the structural properties needed to initialize this Data.
        Used for storage and to help construct new similar Data. All kwargs
        other than "dataframe" and "description" are considered structural.
        """
        cls = type(self)
        # pyre-ignore[45]: Cannot insantiate abstract class
        return cls(df=df, **cls.serialize_init_args(self))
