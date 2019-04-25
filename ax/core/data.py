#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from hashlib import md5
from typing import Dict, Iterable, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd
from ax.core.base import Base


TPdTimestamp = pd.Timestamp  # pyre-ignore[16]: Pyre doesn't recognize this type

COLUMN_DATA_TYPES = {
    "arm_name": str,
    "metric_name": str,
    "trial_index": np.int64,
    "n": np.int64,
    "frac_nonnull": np.float64,
    "mean": np.float64,
    "sem": np.float64,
    "random_split": np.int64,
    "start_time": TPdTimestamp,
    "end_time": TPdTimestamp,
}
REQUIRED_COLUMNS = {"arm_name", "metric_name", "mean", "sem"}


class Data(Base):
    """Class storing data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to gluster for future use by attaching it to an experiment using
    `experiment.add_data()` (this requires a description to be set.)


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
            missing_columns = self.required_columns() - (
                self.required_columns() & set(df.columns.tolist())
            )
            if len(missing_columns) > 0:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            extra_columns = set(df.columns.tolist()) - (
                set(self.column_data_types().keys()) & set(df.columns.tolist())
            )
            if len(extra_columns) > 0:
                raise ValueError(f"Columns {list(extra_columns)} are not supported.")

            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            self._df = self._safecast_df(df=df)
        self.description = description

    def _safecast_df(self, df: pd.DataFrame) -> pd.DataFrame:
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
            for col, coltype in self.column_data_types().items()
            if col in df.columns.values
            and not (
                self.column_data_types()[col] is np.int64
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
        if sum(1 for _ in data) == 0:  # Return empty data if empty iterable.
            return Data()
        return Data(df=pd.concat([datum.df for datum in data], axis=0, sort=True))

    @staticmethod
    def from_evaluations(
        evaluations: Dict[str, Dict[str, Tuple[float, float]]], trial_index: int
    ) -> "Data":
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from condition name to metric outcomes.

        Returns:
            Ax Data object.
        """
        return Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": name,
                        "metric_name": metric_name,
                        "mean": evaluation[metric_name][0],
                        "sem": evaluation[metric_name][1],
                        "trial_index": trial_index,
                    }
                    for name, evaluation in evaluations.items()
                    for metric_name in evaluation.keys()
                ]
            )
        )

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
        return md5(self.df.to_json().encode("utf-8")).hexdigest()


def set_single_trial(data: Data) -> Data:
    """Returns a new Data object where we set all rows to have the same
    trial index (i.e. 0). This is meant to be used with our IVW transform,
    which will combine multiple observations of the same metric.
    """
    df = data._df.copy()
    if "trial_index" in df:
        df["trial_index"] = 0
    return Data(df=df)


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
        required_columns: Set of additional columns reqiured for this data object.
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
