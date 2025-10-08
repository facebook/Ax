#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from io import StringIO
from logging import Logger
from typing import Any, cast, TypeVar

import numpy as np
import pandas as pd
from ax.exceptions.core import UserInputError
from ax.utils.common.base import Base
from ax.utils.common.equality import dataframe_equals
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import (
    extract_init_args,
    SerializationMixin,
    TClassDecoderRegistry,
    TDecoderRegistry,
)
from ax.utils.stats.math_utils import relativize as relativize_func
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)

TData = TypeVar("TData", bound="Data")
DF_REPR_MAX_LENGTH = 1000
MAP_KEY = "step"


class Data(Base, SerializationMixin):
    """Class storing numerical data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        full_df: DataFrame with underlying data. For Data, the required columns
            are "arm_name", "metric_name", "mean", and "sem", the latter two of
            which must be numeric. This is close to the raw data input by the
            user as ``df``; by contrast, in the ``Data`` subclass ``MapData``,
            the property ``self.df`` may be a subset of the full data used for
            modeling. Constructing ``df`` can be expensive, so it is better to
            reference ``full_df`` than ``df`` for operations that do not require
            scanning the full data, such as accessing the columns of the
            DataFrame.

    Properties:
        df: Potentially smaller representation of the data used for modeling. In
            the base class ``Data``, ``df`` equals ``full_df``. In the subclass
            ``MapData``, ``df`` contains only the most recent ``step`` values
            for each trial-arm-metric. Because constructing ``df`` can be
            expensive, it is recommended to reference ``full_df`` for operations
            that do not require scanning the full data, such as accessing the
            columns of the DataFrame.
    """

    # Note: Although the SEM (standard error of the mean) is a required column,
    # downstream models can infer missing SEMs. Simply specify NaN as the SEM value,
    # either in your Metric class or in Data explicitly.
    REQUIRED_COLUMNS = {
        "trial_index",
        "arm_name",
        "metric_name",
        "mean",
        "sem",
        "metric_signature",
    }

    # Note on text data: https://pandas.pydata.org/docs/user_guide/text.html
    # Its type can either be `numpy.dtypes.ObjectDType` or StringDtype extension
    # type; the later is still experimental. So we are using object.
    COLUMN_DATA_TYPES: dict[str, Any] = {
        # Ubiquitous columns.
        "trial_index": int,
        "arm_name": np.dtype("O"),
        # Metric data-related columns.
        "metric_name": np.dtype("O"),
        "metric_signature": np.dtype("O"),
        "mean": np.float64,
        "sem": np.float64,
        # Metadata columns available for all subclasses.
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "n": int,
        # Metadata columns available for only some subclasses.
        "frac_nonnull": np.float64,
        "random_split": int,
        # Used with MapData
        MAP_KEY: float,
    }

    full_df: pd.DataFrame

    def __init__(
        self: TData,
        df: pd.DataFrame | None = None,
        _skip_ordering_and_validation: bool = False,
    ) -> None:
        """Initialize a ``Data`` object from the given DataFrame.

        Args:
            df: DataFrame with underlying data, and required columns.
            _skip_ordering_and_validation: If True, uses the given DataFrame
                as is, without ordering its columns or validating its contents.
                Intended only for use in `Data.filter`, where the contents
                of the DataFrame are known to be ordered and valid.
        """
        if df is None:
            # Initialize with barebones DF with expected dtypes
            self.full_df = pd.DataFrame.from_dict(
                {
                    col: pd.Series([], dtype=self.COLUMN_DATA_TYPES[col])
                    for col in self.required_columns()
                }
            )
        elif _skip_ordering_and_validation:
            self.full_df = df
        else:
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            # Drop rows where every input is null. Since `dropna` can be slow, first
            # check trial index to see if dropping nulls might be needed.
            if df["trial_index"].isnull().any():
                df = df.dropna(axis=0, how="all", ignore_index=True)
            df = self._safecast_df(df=df)
            self.full_df = self._get_df_with_cols_in_expected_order(df=df)

    @classmethod
    def _get_df_with_cols_in_expected_order(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder the columns for easier viewing"""
        current_order = list(df.columns)  # Surprisingly slow, so do it once
        overall_order = list(cls.COLUMN_DATA_TYPES)
        desired_order = [c for c in overall_order if c in current_order] + [
            c for c in current_order if c not in overall_order
        ]
        if current_order != desired_order:
            return df.reindex(columns=desired_order, copy=False)
        return df

    @classmethod
    def _safecast_df(cls: type[TData], df: pd.DataFrame) -> pd.DataFrame:
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
        for col, coltype in cls.COLUMN_DATA_TYPES.items():
            if col in df.columns.values and coltype is not Any:
                # Pandas timestamp handlng is weird
                dtype = "datetime64[ns]" if coltype is pd.Timestamp else coltype
                if (dtype != dtypes[col]) and not (
                    coltype is int and df.loc[:, col].isnull().any()
                ):
                    df[col] = df[col].astype(dtype)
        df.reset_index(inplace=True, drop=True)
        return df

    def required_columns(self) -> set[str]:
        """Names of columns that must be present in the underlying ``DataFrame``."""
        return self.REQUIRED_COLUMNS

    @classmethod
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        """Serialize the class-dependent properties needed to initialize this Data.
        Used for storage and to help construct new similar Data.
        """
        data = assert_is_instance(obj, cls)
        return {"df": data.full_df}

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
        # Backward compatibility
        return extract_init_args(args=args, class_=cls)

    @property
    def df(self) -> pd.DataFrame:
        return self.full_df

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

    def __repr__(self) -> str:
        """String representation of the subclass, inheriting from this base."""
        df_markdown = self.df.to_markdown()
        if len(df_markdown) > DF_REPR_MAX_LENGTH:
            df_markdown = df_markdown[:DF_REPR_MAX_LENGTH] + "..."
        return f"{self.__class__.__name__}(df=\n{df_markdown})"

    @property
    def metric_names(self) -> set[str]:
        """Set of metric names that appear in the underlying dataframe of
        this object.
        """
        return set() if self.df.empty else set(self.df["metric_name"].values)

    @property
    def metric_signatures(self) -> set[str]:
        """Set of metric signatures that appear in the underlying dataframe of
        this object.
        """
        return set() if self.df.empty else set(self.df["metric_signature"].values)

    def filter(
        self: TData,
        trial_indices: Iterable[int] | None = None,
        metric_names: Iterable[str] | None = None,
        metric_signatures: Iterable[str] | None = None,
    ) -> TData:
        """Construct a new object with the subset of rows corresponding to the
        provided trial indices, metric names, and metric signatures. If trial_indices,
        metric_names, or metric_signatures are not provided, that dimension will not be
        filtered.
        """

        if metric_names and metric_signatures:
            raise UserInputError(
                "Cannot filter by both metric names and metric signatures. "
                "Please filter by one or the other."
            )

        return self.__class__(
            df=_filter_df(
                df=self.full_df,
                trial_indices=trial_indices,
                metric_names=metric_names,
                metric_signatures=metric_signatures,
            ),
            _skip_ordering_and_validation=True,
        )

    @classmethod
    def from_multiple_data(cls, data: Iterable[Data]) -> Data:
        """Combines multiple objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax objects of this class to combine.
            subset_metrics: If specified, combined object will only contain
                metrics, names of which appear in this iterable,
                in the underlying dataframe.
        """
        return cls.from_multiple(data=data)

    def relativize(
        self,
        status_quo_name: str = "status_quo",
        as_percent: bool = False,
        include_sq: bool = False,
        bias_correction: bool = True,
        control_as_constant: bool = False,
    ) -> "Data":
        """Relativize this data object w.r.t. a status_quo arm.

        Args:
            status_quo_name: The name of the status_quo arm.
            as_percent: If True, return results as percentage change.
            include_sq: Include status quo in final df.
            bias_correction: Whether to apply bias correction when computing relativized
                metric values. Uses a second-order Taylor expansion for approximating
                the means and standard errors or the ratios, see
                ax.utils.stats.statstools.relativize for more details.
            control_as_constant: If true, control is treated as a constant.
                bias_correction is ignored when this is true.

        Returns:
            The new data object with the relativized metrics (excluding the
                status_quo arm)

        """

        df = self.df.copy()
        grp_cols = list(
            {"trial_index", "metric_name", "random_split"}.intersection(
                df.columns.values
            )
        )

        grouped_df = df.groupby(grp_cols)
        dfs = []
        for grp in grouped_df.groups.keys():
            subgroup_df = grouped_df.get_group(grp)
            is_sq = subgroup_df["arm_name"] == status_quo_name

            sq_mean, sq_sem = (
                subgroup_df[is_sq][["mean", "sem"]].drop_duplicates().values.flatten()
            )

            # rm status quo from final df to relativize
            if not include_sq:
                subgroup_df = subgroup_df[~is_sq]
            means_rel, sems_rel = relativize_func(
                means_t=subgroup_df["mean"].values,
                sems_t=subgroup_df["sem"].values,
                mean_c=sq_mean,
                sem_c=sq_sem,
                as_percent=as_percent,
                bias_correction=bias_correction,
                control_as_constant=control_as_constant,
            )
            dfs.append(
                pd.concat(
                    [
                        subgroup_df.drop(["mean", "sem"], axis=1),
                        pd.DataFrame(
                            np.array([means_rel, sems_rel]).T,
                            columns=["mean", "sem"],
                            index=subgroup_df.index,
                        ),
                    ],
                    axis=1,
                )
            )
        df_rel = pd.concat(dfs, axis=0)
        if include_sq:
            df_rel.loc[df_rel["arm_name"] == status_quo_name, "sem"] = 0.0
        return Data(df_rel)

    def clone(self: TData) -> TData:
        """Returns a new Data object with the same underlying dataframe."""
        return self.__class__(df=deepcopy(self.full_df))

    def __eq__(self, o: Data) -> bool:
        return type(self) is type(o) and dataframe_equals(self.full_df, o.full_df)


def _filter_df(
    df: pd.DataFrame,
    trial_indices: Iterable[int] | None = None,
    metric_names: Iterable[str] | None = None,
    metric_signatures: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter rows of a dataframe by trial indices, metric names, metric signatures,
    or trial indices and either metric names or metric signatures."""

    if metric_names and metric_signatures:
        raise UserInputError(
            "Cannot filter by both metric names and metric signatures. "
            "Please filter by one or the other."
        )

    if trial_indices is not None:
        # Trial indices is not None, metric names is not yet known.
        trial_indices_mask = df["trial_index"].isin(trial_indices)
        # If metric names is not None, filter by both.
        if metric_names:
            metric_names_mask = df["metric_name"].isin(metric_names)
            return df.loc[trial_indices_mask & metric_names_mask]
        # If metric signatures is not None, filter by both.
        if metric_signatures:
            metric_signatures_mask = df["metric_signature"].isin(metric_signatures)
            return df.loc[trial_indices_mask & metric_signatures_mask]
        # If metric names and signatures are None, filter by trial indices only.
        return df.loc[trial_indices_mask]
    if metric_names is not None:
        # Trial indices is None, metric signatures is None, metric names is not None.
        metric_names_mask = df["metric_name"].isin(metric_names)
        return df.loc[metric_names_mask]
    if metric_signatures is not None:
        # Trial indices is None, metric names is None, metric signatures is not None.
        metric_signatures_mask = df["metric_signature"].isin(metric_signatures)
        return df.loc[metric_signatures_mask]
    # All three are None, return the dataframe as is.
    return df
