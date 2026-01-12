#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from copy import deepcopy
from functools import cached_property
from io import StringIO
from logging import Logger
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.exceptions.core import UnsupportedError, UserInputError
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

    The dataframe is retrieved via the ``full_df`` property; when a "step"
    column is present, this indicates progression over time, and data for the
    most recent step per trial-arm-metric can be accessed with the `df`
    property. The data can be stored to an external store for future use by
    attaching it to an experiment using `experiment.attach_data()`.

    IMPORTANT: A ``Data``'s attributes, such as ``full_df``, should not be
    mutated, because cached properties ``df``, ``has_step_column``, and
    ``trial_indices`` will not be updated. If you need to change ``full_df``,
    construct a new ``Data``.


    Attributes:
        full_df: DataFrame with underlying data. The required columns
            are "arm_name", "metric_name", "mean", and "sem", the latter two of
            which must be numeric. This is close to the raw data input by the
            user as ``df``.
        has_step_column: Whether the data contains a "step" column (equivalently,
            `MAP_KEY`).
        _memo_df: Cached representation of ``df`` (see below).

    Properties:
        df: Potentially smaller representation of the data used for modeling.
            When no "step" column is present, ``df`` equals ``full_df``. When a
            ``step`` column is present in ``full_df``, ``df`` contains only the
            most recent ``step`` values for each trial-arm-metric. Because
            constructing ``df`` can be expensive, it is recommended to reference
            ``full_df`` for operations that do not require scanning the full
            data, such as accessing the columns of the DataFrame.
        trial_indices: The distinct values in ``full_df``'s ``trial_index``
            column.
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
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "n": int,
        "frac_nonnull": np.float64,
        "random_split": int,
        MAP_KEY: float,
    }

    # When constructing `data.df` or calling `data.latest`, keep only 1 or N
    # rows from each (trial, arm, metric) group
    DEDUPLICATE_BY_COLUMNS = [
        "trial_index",
        "arm_name",
        "metric_name",
        "metric_signature",
    ]

    full_df: pd.DataFrame

    def __init__(
        self: TData,
        df: pd.DataFrame | None = None,
        _skip_ordering_and_validation: bool = False,
    ) -> None:
        """Initialize a ``Data`` object from the given DataFrame.

        Args:
            df: DataFrame with underlying data, and required columns. Data must
                be unique at the level of ("trial_index", "arm_name",
                "metric_name"), plus "step" if a "step" column is present. A
                lightly processed version of this argument will become the
                `Data`'s `full_df` attribute.
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
                    for col in self.REQUIRED_COLUMNS
                }
            )
        elif _skip_ordering_and_validation:
            self.full_df = df
        else:
            columns = set(df.columns)
            missing_columns = self.REQUIRED_COLUMNS - columns
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
        self._memo_df = None
        self.has_step_column = MAP_KEY in self.full_df.columns

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
        """Given a dictionary, extract the properties needed to initialize the data.
        Used for storage.

        Note: Older Data saved with the former `MapData` class may have been
        stored with progressions represented by a column or columns other than
        "step" and had `MapKeyInfo`s to indicate which columns corresponded to
        progressions. This is no longer supported, and such columns will not be
        recognized as progressions if provided.
        """
        # Extract `df` only if present, since certain inputs to this fn, e.g.
        # SQAData.structure_metadata_json, don't have a `df` attribute.
        if "df" in args and not isinstance(args["df"], pd.DataFrame):
            # NOTE: Need dtype=False, otherwise infers arm_names like
            # "4_1" should be int 41.
            args["df"] = pd.read_json(StringIO(args["df"]["value"]), dtype=False)
        return extract_init_args(args=args, class_=cls)

    @property
    def df(self) -> pd.DataFrame:
        """
        Return data aggregated up to the final step of each trial-arm-metric.

        If there is no "step" column, this is the same as ``full_df``, since it
        is assumed that there is only one row per (trial, arm, metric) group.

        If there is a "step" column, once this function is called, its result is
        written to ``self._memo_df``, and ``self._memo_df`` is returned on any
        subsequent calls, since it is assumed that data has not been mutated in
        place.
        """
        # Case: no step column, so no aggregation needed
        if not self.has_step_column:
            return self.full_df
        # Case: Result already cached
        if self._memo_df is not None:
            return self._memo_df

        # Case: Empty data
        if self.full_df.empty:
            return self.full_df

        idxs = (
            self.full_df.fillna({MAP_KEY: np.inf})
            .groupby(self.DEDUPLICATE_BY_COLUMNS)[MAP_KEY]
            .idxmax()
            # In the case where all MAP_KEY values are NaN for a group we return an
            # arbitrary row from that group.
            .fillna(
                self.full_df.groupby(self.DEDUPLICATE_BY_COLUMNS).apply(
                    lambda group: group.index[0]
                )
            )
        )
        self._memo_df = self.full_df.loc[idxs]

        return self._memo_df

    @classmethod
    def from_multiple_data(cls: type[TData], data: Iterable[Data]) -> TData:
        """Combines multiple objects into one (with the concatenated
        underlying dataframe).

        Args:
            data: Iterable of Ax objects of this class to combine.
        """
        dfs = [datum.full_df for datum in data if not datum.full_df.empty]

        if len(dfs) == 0:
            return cls()

        result_has_step_column = any(d.has_step_column for d in data)
        return cls(df=pd.concat(dfs, axis=0, sort=not result_has_step_column))

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
        self: Data,
        trial_indices: Iterable[int] | None = None,
        metric_names: Iterable[str] | None = None,
        metric_signatures: Iterable[str] | None = None,
    ) -> Data:
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
        if self.has_step_column:
            raise NotImplementedError(
                "Relativization is not supported for data with step columns."
            )
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

    @cached_property
    def trial_indices(self) -> set[int]:
        """Return the set of trial indices in the data."""
        if self._memo_df is not None:
            # Use a smaller df if available
            return set(self.df["trial_index"].unique())
        # If no small df is available, use the full df
        return set(self.full_df["trial_index"].unique())

    def latest(self, rows_per_group: int = 1) -> Data:
        """Return a new Data with the most recently observed `rows_per_group`
        rows for each (trial_index, arm, metric) group, determined by the "step"
        values, where higher implies more recent.

        This function considers only the relative ordering of the "step" values,
        making it most suitable when these values are equally spaced.

        If `rows_per_group` is greater than the number of rows in a given
        (trial_index, arm, metric) group, then all rows are returned.

        If there are no "step" values, then this simply returns the original.
        """
        if not self.has_step_column:
            if rows_per_group != 1:
                raise UnsupportedError(
                    "Cannot have rows_per_group greater than 1 for data without"
                    " a 'step' column, because there should inherently be only "
                    "one row per (trial, arm, metric) group."
                )
            return self
        # Note: Normally, a groupby-apply automatically returns a DataFrame that is
        # sorted by the group keys, but this is not true when using filtrations like
        # "tail."

        # Optimizer beware: This is slow and it has proven difficult to speed it
        # up. `latest` takes up a large portion of the time, and so does the groupby;
        # sorting can take ~40% of the time. If you find this to be a bottleneck, it
        # may be better to avoid unnecessary calls.

        return Data(
            df=(
                self.full_df.sort_values(MAP_KEY)
                .groupby(self.DEDUPLICATE_BY_COLUMNS)
                .tail(rows_per_group)
                .sort_values(self.DEDUPLICATE_BY_COLUMNS)
            )
        )

    def subsample(
        self: TData,
        keep_every: int | None = None,
        limit_rows_per_group: int | None = None,
        limit_rows_per_metric: int | None = None,
        include_first_last: bool = True,
    ) -> TData:
        """Return a new Data that subsamples the `MAP_KEY` column in an
        equally-spaced manner. This function considers only the relative ordering
        of the `MAP_KEY` values, making it most suitable when these values are
        equally spaced.

        There are three ways that this can be done:
            1. If `keep_every = k` is set, then every kth row of the DataFrame in the
                `MAP_KEY` column is kept after grouping by `DEDUPLICATE_BY_COLUMNS`.
                In other words, every kth step of each (arm, metric) will be kept.
            2. If `limit_rows_per_group = n`, the method will find the (arm, metric)
                pair with the largest number of rows in the `MAP_KEY` column and select
                an appropriate `keep_every` such that each (arm, metric) has at most
                `n` rows in the `MAP_KEY` column.
            3. If `limit_rows_per_metric = n`, the method will select an
                appropriate `keep_every` such that the total number of rows per
                metric is less than `n`.
        If multiple of `keep_every`, `limit_rows_per_group`, `limit_rows_per_metric`,
        then the priority is in the order above: 1. `keep_every`,
        2. `limit_rows_per_group`, and 3. `limit_rows_per_metric`.

        Note that we want all curves to be subsampled with nearly the same spacing.
        Internally, the method converts `limit_rows_per_group` and
        `limit_rows_per_metric` to a `keep_every` quantity that will satisfy the
        original request.

        When `include_first_last` is True, then the method will use the `keep_every`
        as a guideline and for each group, produce (nearly) evenly spaced points that
        include the first and last points.
        """
        if (
            keep_every is None
            and limit_rows_per_group is None
            and limit_rows_per_metric is None
        ):
            logger.warning(
                "None of `keep_every`, `limit_rows_per_group`, or "
                "`limit_rows_per_metric` is specified. Returning the original data "
                "without subsampling."
            )
            return self
        if not self.has_step_column:
            # Data should automatically obey all the conditions, since there is
            # just one row per trial-arm-metric group
            return self

        subsampled_metric_dfs = []
        for metric_name in self.full_df["metric_name"].unique():
            metric_full_df = _filter_df(self.full_df, metric_names=[metric_name])
            subsampled_metric_dfs.append(
                _subsample_one_metric(
                    metric_full_df,
                    keep_every=keep_every,
                    limit_rows_per_group=limit_rows_per_group,
                    limit_rows_per_metric=limit_rows_per_metric,
                    include_first_last=include_first_last,
                )
            )
        subsampled_df: pd.DataFrame = pd.concat(subsampled_metric_dfs)
        return self.__class__(df=subsampled_df)


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


def _ceil_divide(
    a: int | np.int_ | npt.NDArray[np.int_], b: int | np.int_ | npt.NDArray[np.int_]
) -> np.int_ | npt.NDArray[np.int_]:
    return -np.floor_divide(-a, b)


def _subsample_rate(
    full_df: pd.DataFrame,
    keep_every: int | None = None,
    limit_rows_per_group: int | None = None,
    limit_rows_per_metric: int | None = None,
) -> int:
    if keep_every is not None:
        return keep_every

    grouped_full_df = full_df.groupby(Data.DEDUPLICATE_BY_COLUMNS)
    group_sizes = grouped_full_df.size()
    max_rows = group_sizes.max()

    if limit_rows_per_group is not None:
        return _ceil_divide(max_rows, limit_rows_per_group).item()

    if limit_rows_per_metric is not None:
        # search for the `keep_every` such that when you apply it to each group,
        # the total number of rows is smaller than `limit_rows_per_metric`.
        ks = np.arange(max_rows, 0, -1)
        # total sizes in ascending order
        total_sizes = np.sum(
            _ceil_divide(group_sizes.values, ks[..., np.newaxis]), axis=1
        )
        # binary search
        i = bisect_right(total_sizes, limit_rows_per_metric)
        # if no such `k` is found, then `derived_keep_every` stays as 1.
        if i > 0:
            return ks[i - 1].item()

    raise ValueError(
        "at least one of `keep_every`, `limit_rows_per_group`, "
        "or `limit_rows_per_metric` must be specified."
    )


def _subsample_one_metric(
    full_df: pd.DataFrame,
    keep_every: int | None = None,
    limit_rows_per_group: int | None = None,
    limit_rows_per_metric: int | None = None,
    include_first_last: bool = True,
) -> pd.DataFrame:
    """Helper function to subsample a dataframe that holds a single metric."""

    grouped_full_df = full_df.groupby(Data.DEDUPLICATE_BY_COLUMNS)

    derived_keep_every = _subsample_rate(
        full_df, keep_every, limit_rows_per_group, limit_rows_per_metric
    )

    if derived_keep_every <= 1:
        return full_df
    filtered_dfs = []
    for _, df_g in grouped_full_df:
        df_g = df_g.sort_values(MAP_KEY)
        if include_first_last:
            rows_per_group = _ceil_divide(len(df_g), derived_keep_every)
            linspace_idcs = np.linspace(0, len(df_g) - 1, rows_per_group)
            idcs = np.round(linspace_idcs).astype(int)
            filtered_df = df_g.iloc[idcs]
        else:
            filtered_df = df_g.iloc[:: int(derived_keep_every)]
        filtered_dfs.append(filtered_df)
    return pd.concat(filtered_dfs)


def sort_by_trial_index_and_arm_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the dataframe by trial index and arm name. The arm names with default patterns
    (e.g. `0_1`, `3_11`) are sorted by trial index part (before underscore) and arm
    number part (after underscore) within trial index. The arm names with non-default
    patterns (e.g. `status_quo`, `control`, `capped_param_1`) are sorted alphabetically
    and will be on the top of the sorted dataframe.

    Args:
        df: The DataFrame to sort.

    Returns:
        The sorted DataFrame.
    """

    # Create new columns for sorting the default arm names
    df["is_default"] = pd.notna(df["arm_name"]) & df["arm_name"].str.count(
        pat=r"^\d+_\d+$"
    )

    df["trial_index_part"] = float("NaN")
    df["arm_name_part"] = float("NaN")

    split_arm_name = df.loc[df["is_default"], "arm_name"].str.split("_")
    df.loc[df["is_default"], "trial_index_part"] = split_arm_name.str.get(0).astype(int)
    df.loc[df["is_default"], "arm_name_part"] = split_arm_name.str.get(1).astype(int)

    # Sort the DataFrame by the new columns (trial_index_part and arm_number_part)
    # for default arm names
    df = (
        df.sort_values(
            by=[
                "trial_index",
                "is_default",
                "trial_index_part",
                "arm_name_part",
                "arm_name",
            ],
            inplace=False,
        ).reset_index(drop=True)
        if not df.empty
        else df
    )

    # Drop the temporary 'trial_index_part' and 'arm_number_part' columns
    df.drop(
        columns=["trial_index_part", "arm_name_part", "is_default"],
        # Ignore errors that occur when dropping columns that do not exist in the
        # dataframe.
        errors="ignore",
        inplace=True,
    )
    return df
