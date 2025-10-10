# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings

from bisect import bisect_right
from collections.abc import Iterable
from logging import Logger
from math import nan
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.data import _filter_df, Data, MAP_KEY
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry

logger: Logger = get_logger(__name__)


class MapData(Data):
    """Class storing mapping-like results for an experiment.

    Data is stored in a dataframe, and auxiliary information is stored in
    DataFrame with column names given by the keys in the passed ``MapKeyInfo``
    objects.

    Mapping-like results occur whenever a metric is reported as a collection
    of results, each element corresponding to a tuple of values.

    The simplest case is a sequence. For instance a time series is
    a mapping from the 1-tuple `(timestamp)` to (mean, sem) results.

    Another example: MultiFidelity results. This is a mapping from
    `(fidelity_feature_1, ..., fidelity_feature_n)` to (mean, sem) results.

    The dataframe is retrieved via the `map_df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        full_df: DataFrame with underlying data. The required columns
            are "arm_name", "metric_name", "mean", "sem", and "step", the latter
            three of which must be numeric. This is close to the raw data input by the
            user as ``df``; by contrast, the property ``self.df`` is be a subset
            of the full data used for modeling. Constructing ``df`` can be
            expensive, so it is better to reference ``full_df`` than ``df`` for
            operations that do not require scanning the full data, such as
            accessing the columns of the DataFrame.
        _memo_df: Either ``None``, if ``self.df`` has never been accessed, or
            equivalent to ``self.df``.

    Properties:
        df: Potentially smaller representation of the data used for modeling,
            containing only the most recent ``step`` values
            for each trial-arm-metric. Because constructing ``df`` can be
            expensive, it is recommended to reference ``full_df`` for operations
            that do not require scanning the full data, such as accessing the
            columns of the DataFrame.
        map_df: Equivalent to ``full_df``. ``map_df`` exists only on
            ``MapData``, whereas ``full_df`` exists for any ``Data`` subclass.
    """

    DEDUPLICATE_BY_COLUMNS = [
        "trial_index",
        "arm_name",
        "metric_name",
        "metric_signature",
    ]

    full_df: pd.DataFrame
    _memo_df: pd.DataFrame | None

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        _skip_ordering_and_validation: bool = False,
    ) -> None:
        """Initialize a ``MapData`` object from the given DataFrame and MapKeyInfos.

        Note: ``MapData`` may also be initialized more simply using
        ``MapData.from_df``, which allows for simpler semantics but may be
        unstable.

        Args:
            df: DataFrame with underlying data, and required columns.
            _skip_ordering_and_validation: If True, uses the given DataFrame
                as is, without ordering its columns or validating its contents.
                Intended only for use in `MapData.filter`, where the contents
                of the DataFrame are known to be ordered and valid.
        """
        if df is not None and MAP_KEY not in df.columns:
            df[MAP_KEY] = nan
        super().__init__(
            df=df, _skip_ordering_and_validation=_skip_ordering_and_validation
        )
        self._memo_df = None

    # true_df is being deprecated after the release of Ax 1.1.2, so it will
    # surface in Ax 1.1.3 or 1.2.0, so it can be removed in the minor release
    # after that.
    @property
    def true_df(self) -> pd.DataFrame:
        warnings.warn(
            "MapData.true_df is deprecated. Use MapData.full_df instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.map_df

    def required_columns(self) -> set[str]:
        return super().required_columns().union({MAP_KEY})

    @property
    def map_df(self) -> pd.DataFrame:
        return self.full_df

    @classmethod
    def from_multiple_data(cls, data: Iterable[Data]) -> MapData:
        """
        Downcast instances of Data into instances of MapData.

        If no "step" column is present, it will be filled in with NaNs.
        """
        map_dfs = [
            datum.full_df
            if isinstance(datum, MapData)
            else datum.df.assign(**{MAP_KEY: nan})
            for datum in data
            if not datum.full_df.empty
        ]

        if len(map_dfs) == 0:
            return MapData()

        return MapData(df=pd.concat(map_dfs))

    @property
    def df(self) -> pd.DataFrame:
        """Returns a DataFrame that only contains the last (determined by map keys)
        observation for each (arm, metric) pair.
        """
        if self._memo_df is not None:
            return self._memo_df

        self._memo_df = _tail(map_df=self.map_df, n=1, sort=True)
        return self._memo_df

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        class_decoder_registry: TClassDecoderRegistry | None = None,
    ) -> dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.

        Most logic here is for backwards compatibility with older MapData that
        may have been stored with multiple map keys and/or a map key with a
        different name.
        """
        # map_key_infos used to be a supported argument; it allowed the column
        # called MAP_KEY to have a different name.
        if "map_key_infos" in args:
            map_keys = {d["key"] for d in args["map_key_infos"]}
        else:
            map_keys = set()

        deserialized = super().deserialize_init_args(args=args)

        bad_keys = map_keys - {MAP_KEY}
        if len(bad_keys) > 0:
            df = deserialized["df"]
            if MAP_KEY in map_keys:
                warnings.warn(
                    f"Received multiple map keys. All except {MAP_KEY}"
                    " will be ignored.",
                    stacklevel=2,
                )
                if df is not None:
                    df.drop(columns=bad_keys, inplace=True)

            else:
                key_to_rename = bad_keys.pop()
                if len(bad_keys) > 0:
                    warnings.warn(
                        "Received multiple map keys. All except for "
                        f"{key_to_rename} will be ignored.",
                        stacklevel=2,
                    )
                    if df is not None:
                        df.drop(columns=bad_keys, inplace=True)

                warnings.warn(
                    f"{key_to_rename} will be renamed to {MAP_KEY} on "
                    "df, since passing custom map keys is no longer supported.",
                    stacklevel=2,
                )
                if df is not None:
                    df.rename(columns={key_to_rename: MAP_KEY}, inplace=True)

        return deserialized

    def latest(self, rows_per_group: int = 1) -> MapData:
        """Return a new MapData with the most recently observed `rows_per_group`
        rows for each (arm, metric) group, determined by the "step" values,
        where higher implies more recent.

        This function considers only the relative ordering of the "step" values,
        making it most suitable when these values are equally spaced.

        If `rows_per_group` is greater than the number of rows in a given
        (arm, metric) group, then all rows are returned.
        """
        return MapData(
            df=_tail(map_df=self.map_df, n=rows_per_group, sort=True),
        )

    def subsample(
        self,
        keep_every: int | None = None,
        limit_rows_per_group: int | None = None,
        limit_rows_per_metric: int | None = None,
        include_first_last: bool = True,
    ) -> MapData:
        """Return a new MapData that subsamples the `MAP_KEY` column in an
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
        subsampled_metric_dfs = []
        for metric_name in self.map_df["metric_name"].unique():
            metric_map_df = _filter_df(self.map_df, metric_names=[metric_name])
            subsampled_metric_dfs.append(
                _subsample_one_metric(
                    metric_map_df,
                    keep_every=keep_every,
                    limit_rows_per_group=limit_rows_per_group,
                    limit_rows_per_metric=limit_rows_per_metric,
                    include_first_last=include_first_last,
                )
            )
        subsampled_df: pd.DataFrame = pd.concat(subsampled_metric_dfs)
        return MapData(df=subsampled_df)


def _ceil_divide(
    a: int | np.int_ | npt.NDArray[np.int_], b: int | np.int_ | npt.NDArray[np.int_]
) -> np.int_ | npt.NDArray[np.int_]:
    return -np.floor_divide(-a, b)


def _subsample_rate(
    map_df: pd.DataFrame,
    keep_every: int | None = None,
    limit_rows_per_group: int | None = None,
    limit_rows_per_metric: int | None = None,
) -> int:
    if keep_every is not None:
        return keep_every

    grouped_map_df = map_df.groupby(MapData.DEDUPLICATE_BY_COLUMNS)
    group_sizes = grouped_map_df.size()
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


def _tail(
    map_df: pd.DataFrame,
    n: int = 1,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Note: Normally, a groupby-apply automatically returns a DataFrame that is
    sorted by the group keys, but this is not true when using filtrations like
    "tail."

    Note: Optimizer beware: This is slow and it has proven difficult to speed it
    up. `tail` takes up a large portion of the time, and so does the groupby;
    sorting can take ~40% of the time. If you find this to be a bottleneck, it
    may be better to avoid unnecessary calls to `.df`.
    """
    if len(map_df) == 0:
        return map_df
    df = map_df.sort_values(MAP_KEY).groupby(MapData.DEDUPLICATE_BY_COLUMNS).tail(n)
    if sort:
        df.sort_values(MapData.DEDUPLICATE_BY_COLUMNS, inplace=True)
    return df


def _subsample_one_metric(
    map_df: pd.DataFrame,
    keep_every: int | None = None,
    limit_rows_per_group: int | None = None,
    limit_rows_per_metric: int | None = None,
    include_first_last: bool = True,
) -> pd.DataFrame:
    """Helper function to subsample a dataframe that holds a single metric."""

    grouped_map_df = map_df.groupby(MapData.DEDUPLICATE_BY_COLUMNS)

    derived_keep_every = _subsample_rate(
        map_df, keep_every, limit_rows_per_group, limit_rows_per_metric
    )

    if derived_keep_every <= 1:
        filtered_map_df = map_df
    else:
        filtered_dfs = []
        for _, df_g in grouped_map_df:
            df_g = df_g.sort_values(MAP_KEY)
            if include_first_last:
                rows_per_group = _ceil_divide(len(df_g), derived_keep_every)
                linspace_idcs = np.linspace(0, len(df_g) - 1, rows_per_group)
                idcs = np.round(linspace_idcs).astype(int)
                filtered_df = df_g.iloc[idcs]
            else:
                filtered_df = df_g.iloc[:: int(derived_keep_every)]
            filtered_dfs.append(filtered_df)
        filtered_map_df: pd.DataFrame = pd.concat(filtered_dfs)
    return filtered_map_df
