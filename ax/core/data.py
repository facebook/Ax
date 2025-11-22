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
from typing import Any, TypeVar

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

    # Indicates whether this data type supports relativization.
    # Subclasses can override this if they don't support relativization.
    supports_relativization: bool = True

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

    @classmethod
    def from_multiple_data(cls: type[TData], data: Iterable[TData]) -> TData:
        """Combines multiple objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax objects of this class to combine.
        """
        dfs = []

        for datum in data:
            if type(datum) is not cls:
                raise TypeError(
                    f"All data objects must be instances of {cls}. Got "
                    f"{cls} and {type(datum)}."
                )
            if not datum.full_df.empty:
                dfs.append(datum.df)

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
            ).reset_index(drop=True),
            _skip_ordering_and_validation=True,
        )

    def clone(self: TData) -> TData:
        """Returns a new Data object with the same underlying dataframe."""
        return self.__class__(df=deepcopy(self.full_df))

    def __eq__(self, o: Data) -> bool:
        return type(self) is type(o) and dataframe_equals(self.full_df, o.full_df)

    def relativize(
        self: TData,
        status_quo_name: str = "status_quo",
        as_percent: bool = False,
        include_sq: bool = False,
        bias_correction: bool = True,
        control_as_constant: bool = False,
    ) -> TData:
        """Relativize a data object w.r.t. a status_quo arm.

        Args:
            data: The data object to be relativized.
            status_quo_name: The name of the status_quo arm.
            as_percent: If True, return results as percentage change.
            include_sq: Include status quo in final df.
            bias_correction: Whether to apply bias correction when computing relativized
                metric values. Uses a second-order Taylor expansion for approximating
                the means and standard errors or the ratios, see
                ax.utils.stats.math_utils.relativize for more details.
            control_as_constant: If true, control is treated as a constant.
                bias_correction is ignored when this is true.

        Returns:
            The new data object with the relativized metrics (excluding the
                status_quo arm)

        """
        df = self.df.copy()
        df_rel = relativize_dataframe(
            df=df,
            status_quo_name=status_quo_name,
            as_percent=as_percent,
            include_sq=include_sq,
            bias_correction=bias_correction,
            control_as_constant=control_as_constant,
        )
        return self.__class__(df=df_rel)


def combine_dfs_favoring_recent(
    last_df: pd.DataFrame, new_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine last_df and new_df.

    Deduplicate in favor of new_df when there are multiple observations with
    the same "trial_index", "metric_name", and "arm_name", and, when
    present, "step." If only one input has a "step," assign a NaN step to the other.

    Args:
        last_df: The DataFrame of data currently attached to a trial
        new_df: A DataFrame containing new data to be attached

    Returns:
        Combined DataFrame
    """
    merge_keys = ["trial_index", "metric_name", "arm_name"]
    # If only one has a "step" column, add a step column to the other one
    if MAP_KEY in last_df.columns:
        merge_keys += [MAP_KEY]
        if MAP_KEY not in new_df.columns:
            new_df["step"] = float("NaN")
    elif MAP_KEY in new_df.columns:
        merge_keys += [MAP_KEY]
        last_df["step"] = float("NaN")

    combined = pd.concat((last_df, new_df), ignore_index=True).drop_duplicates(
        subset=merge_keys, keep="last", ignore_index=True
    )
    return assert_is_instance(combined, pd.DataFrame)


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


def relativize_dataframe(
    df: pd.DataFrame,
    status_quo_name: str = "status_quo",
    as_percent: bool = False,
    include_sq: bool = False,
    bias_correction: bool = True,
    control_as_constant: bool = False,
) -> pd.DataFrame:
    """Relativize a dataframe w.r.t. a status_quo arm.

    Args:
        df: The dataframe to be relativized.
        status_quo_name: The name of the status_quo arm.
        as_percent: If True, return results as percentage change.
        include_sq: Include status quo in final df.
        bias_correction: Whether to apply bias correction when computing relativized
            metric values. Uses a second-order Taylor expansion for approximating
            the means and standard errors or the ratios, see
            ax.utils.stats.math_utils.relativize for more details.
        control_as_constant: If true, control is treated as a constant.
            bias_correction is ignored when this is true.

    Returns:
        The new dataframe with the relativized metrics (excluding the
            status_quo arm)

    """
    grp_cols = list(
        {"trial_index", "metric_name", "random_split"}.intersection(df.columns.values)
    )

    grouped_df = df.groupby(grp_cols)
    dfs = []
    for grp in grouped_df.groups.keys():
        subgroup_df = grouped_df.get_group(grp)
        is_sq = subgroup_df["arm_name"] == status_quo_name

        # Check if status quo exists in this subgroup (trial)
        sq_data = subgroup_df[is_sq][["mean", "sem"]].drop_duplicates().values
        if len(sq_data) == 0:
            # No status quo in this trial - skip relativization and include raw data
            logger.debug(
                "Status quo '%s' not found in trial group %s - "
                "skipping relativization for this group",
                status_quo_name,
                grp,
            )
            dfs.append(subgroup_df)
            continue

        sq_mean, sq_sem = sq_data.flatten()

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
        dfs.append(subgroup_df.assign(mean=means_rel, sem=sems_rel))
    df_rel = pd.concat(dfs, axis=0)
    if include_sq:
        df_rel.loc[df_rel["arm_name"] == status_quo_name, "sem"] = 0.0
    df_rel.reset_index(inplace=True, drop=True)
    # Reorder columns to match expected order (reuses Data class logic)
    df_rel = Data._get_df_with_cols_in_expected_order(df_rel)
    return df_rel
