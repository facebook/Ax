# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from copy import deepcopy
from logging import Logger
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.data import _filter_df, Data
from ax.core.types import TMapTrialEvaluation
from ax.exceptions.core import UnsupportedError
from ax.utils.common.base import SortableBase
from ax.utils.common.docutils import copy_doc
from ax.utils.common.equality import dataframe_equals
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import (
    serialize_init_args,
    TClassDecoderRegistry,
    TDecoderRegistry,
)
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)


T = TypeVar("T")


class MapKeyInfo(Generic[T], SortableBase):
    """Helper class storing map keys and auxilary info for use in MapData"""

    def __init__(
        self,
        key: str,
        default_value: T,
    ) -> None:
        self._key = key
        self._default_value = default_value

    def __str__(self) -> str:
        return f"MapKeyInfo({self.key}, {self.default_value})"

    def __hash__(self) -> int:
        return hash((self.key, self.default_value))

    def _unique_id(self) -> str:
        return str(self.__hash__())

    @property
    def key(self) -> str:
        return self._key

    @property
    def default_value(self) -> T:
        return self._default_value

    @property
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def value_type(self) -> type:
        return type(self._default_value)

    def clone(self) -> MapKeyInfo[T]:
        """Return a copy of this MapKeyInfo."""
        return MapKeyInfo(key=self.key, default_value=deepcopy(self.default_value))


class MapData(Data):
    """Class storing mapping-like results for an experiment.

    Data is stored in a dataframe, and auxiliary information ((key name,
    default value) pairs) are stored in a collection of MapKeyInfo objects.

    Mapping-like results occur whenever a metric is reported as a collection
    of results, each element corresponding to a tuple of values.

    The simplest case is a sequence. For instance a time series is
    a mapping from the 1-tuple `(timestamp)` to (mean, sem) results.

    Another example: MultiFidelity results. This is a mapping from
    `(fidelity_feature_1, ..., fidelity_feature_n)` to (mean, sem) results.

    The dataframe is retrieved via the `map_df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)
    """

    DEDUPLICATE_BY_COLUMNS = ["arm_name", "metric_name"]

    _map_df: pd.DataFrame
    _memo_df: pd.DataFrame | None

    # pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
    _map_key_infos: list[MapKeyInfo]

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        # pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
        map_key_infos: Iterable[MapKeyInfo] | None = None,
        description: str | None = None,
        _skip_ordering_and_validation: bool = False,
    ) -> None:
        """Initialize a ``MapData`` object from the given DataFrame and MapKeyInfos.

        Args:
            df: DataFrame with underlying data, and required columns.
            map_key_infos: A list of MapKeyInfo objects, each of which contains
                information about the mapping-like structure of the data.
                See the class docstring for additional information.
            description: Human-readable description of data.
            _skip_ordering_and_validation: If True, uses the given DataFrame
                as is, without ordering its columns or validating its contents.
                Intended only for use in `MapData.filter`, where the contents
                of the DataFrame are known to be ordered and valid.
        """
        if map_key_infos is None and df is not None:
            raise ValueError("map_key_infos may be `None` iff `df` is None.")

        # pyre-fixme[4]: Attribute must be annotated.
        self._map_key_infos = list(map_key_infos) if map_key_infos is not None else []

        if df is None:  # If df is None create an empty dataframe with appropriate cols
            self._map_df = pd.DataFrame(
                columns=list(self.required_columns().union(self.map_keys))
            )
        elif _skip_ordering_and_validation:
            self._map_df = df
        else:
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise UnsupportedError(
                    f"Dataframe must contain required columns {missing_columns}."
                )
            extra_columns = columns - self.supported_columns(
                extra_column_names=self.map_keys
            )
            if extra_columns:
                raise UnsupportedError(
                    f"Columns {[mki.key for mki in extra_columns]} are not supported."
                )
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            df = self._safecast_df(df=df, extra_column_types=self.map_key_to_type)

            col_order = [
                c
                for c in self.column_data_types(self.map_key_to_type)
                if c in df.columns
            ]
            self._map_df = df[col_order]

        self.description = description

        self._memo_df = None

    def __eq__(self, o: MapData) -> bool:
        mkis_match = set(self.map_key_infos) == set(o.map_key_infos)
        dfs_match = dataframe_equals(self.map_df, o.map_df)

        return mkis_match and dfs_match

    @property
    def true_df(self) -> pd.DataFrame:
        return self.map_df

    @property
    # pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
    def map_key_infos(self) -> list[MapKeyInfo]:
        return self._map_key_infos

    @property
    def map_keys(self) -> list[str]:
        return [mki.key for mki in self.map_key_infos]

    @property
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def map_key_to_type(self) -> dict[str, type]:
        return {mki.key: mki.value_type for mki in self.map_key_infos}

    @staticmethod
    def from_multiple_map_data(
        data: Sequence[MapData],
        subset_metrics: Iterable[str] | None = None,
    ) -> MapData:
        if len(data) == 0:
            return MapData()

        unique_map_key_infos = []
        for mki in (mki for datum in data for mki in datum.map_key_infos):
            if any(
                mki.key == unique.key
                and not np.isclose(
                    mki.default_value, unique.default_value, equal_nan=True
                )
                for unique in unique_map_key_infos
            ):
                logger.warning(f"MapKeyInfo conflict for {mki.key}, eliding {mki}.")
            else:
                if not any(mki.key == unique.key for unique in unique_map_key_infos):
                    # If there is a key conflict but the mkis are equal, silently do
                    # not add the duplicate.
                    unique_map_key_infos.append(mki)

        df = pd.concat(
            [pd.DataFrame(columns=[mki.key for mki in unique_map_key_infos])]
            + [datum.map_df for datum in data]
        ).fillna(value={mki.key: mki.default_value for mki in unique_map_key_infos})

        if subset_metrics:
            subset_metrics_mask = df["metric_name"].isin(subset_metrics)
            df = df[subset_metrics_mask]

        return MapData(df=df, map_key_infos=unique_map_key_infos)

    @staticmethod
    def from_map_evaluations(
        evaluations: dict[str, TMapTrialEvaluation],
        trial_index: int,
        # pyre-fixme[24]: Generic type `MapKeyInfo` expects 1 type parameter.
        map_key_infos: Iterable[MapKeyInfo] | None = None,
    ) -> MapData:
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else None,
                "trial_index": trial_index,
                **map_dict,
            }
            for name, map_dict_and_metrics_list in evaluations.items()
            for map_dict, evaluation in map_dict_and_metrics_list
            for metric_name, value in evaluation.items()
        ]
        map_keys = {
            key
            for name, map_dict_and_metrics_list in evaluations.items()
            for map_dict, evaluation in map_dict_and_metrics_list
            for key in map_dict.keys()
        }
        map_key_infos = map_key_infos or [
            MapKeyInfo(key=key, default_value=np.nan) for key in map_keys
        ]

        if {mki.key for mki in map_key_infos} != map_keys:
            raise ValueError("Inconsistent map_key sets in evaluations.")

        return MapData(df=pd.DataFrame(records), map_key_infos=map_key_infos)

    @property
    def map_df(self) -> pd.DataFrame:
        return self._map_df

    @map_df.setter
    # pyre-fixme[3]: Return type must be annotated.
    def map_df(self, df: pd.DataFrame):
        raise UnsupportedError(
            "MapData's underlying DataFrame is immutable; create a new"
            + " MapData via `__init__` or `from_multiple_data`."
        )

    @classmethod
    def from_multiple_data(
        cls,
        data: Iterable[Data],
        subset_metrics: Iterable[str] | None = None,
    ) -> MapData:
        """Downcast instances of Data into instances of MapData with empty
        map_key_infos if necessary then combine as usual (filling in empty cells with
        default values).
        """
        map_datas = [
            (
                cls(df=datum.df, map_key_infos=[])
                if not isinstance(datum, MapData)
                else datum
            )
            for datum in data
        ]

        return cls.from_multiple_map_data(data=map_datas, subset_metrics=subset_metrics)

    @property
    def df(self) -> pd.DataFrame:
        """Returns a DataFrame that only contains the last (determined by map keys)
        observation for each (arm, metric) pair.
        """
        if self._memo_df is not None:
            return self._memo_df

        if len(self.map_keys) == 0:
            # If map_keys is empty just return the df
            return self.map_df

        self._memo_df = self.map_df.sort_values(self.map_keys).drop_duplicates(
            MapData.DEDUPLICATE_BY_COLUMNS, keep="last"
        )

        return self._memo_df

    @copy_doc(Data.filter)
    def filter(
        self,
        trial_indices: Iterable[int] | None = None,
        metric_names: Iterable[str] | None = None,
    ) -> MapData:
        return self.__class__(
            df=_filter_df(
                df=self.map_df, trial_indices=trial_indices, metric_names=metric_names
            ),
            map_key_infos=self.map_key_infos,
            _skip_ordering_and_validation=True,
        )

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        map_data = assert_is_instance(obj, MapData)
        properties = serialize_init_args(
            obj=map_data, exclude_fields=["_skip_ordering_and_validation"]
        )
        properties["df"] = map_data.map_df
        properties["map_key_infos"] = [
            serialize_init_args(obj=mki) for mki in properties["map_key_infos"]
        ]
        return properties

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        class_decoder_registry: TClassDecoderRegistry | None = None,
    ) -> dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.
        """
        args["map_key_infos"] = [
            MapKeyInfo(d["key"], d["default_value"]) for d in args["map_key_infos"]
        ]
        return super().deserialize_init_args(args=args)

    def clone(self) -> MapData:
        """Returns a new ``MapData`` object with the same underlying dataframe
        and map key infos.
        """
        return MapData(
            df=deepcopy(self.map_df),
            map_key_infos=[mki.clone() for mki in self.map_key_infos],
            description=self.description,
        )

    def subsample(
        self,
        map_key: str | None = None,
        keep_every: int | None = None,
        limit_rows_per_group: int | None = None,
        limit_rows_per_metric: int | None = None,
        include_first_last: bool = True,
    ) -> MapData:
        """Subsample the `map_key` column in an equally-spaced manner (if there is
        a `self.map_keys` is length one, then `map_key` can be set to None). The
        values of the `map_key` column are not taken into account, so this function
        is most reasonable when those values are equally-spaced. There are three
        ways that this can be done:
            1. If `keep_every = k` is set, then every kth row of the DataFrame in the
                `map_key` column is kept after grouping by `DEDUPLICATE_BY_COLUMNS`.
                In other words, every kth step of each (arm, metric) will be kept.
            2. If `limit_rows_per_group = n`, the method will find the (arm, metric)
                pair with the largest number of rows in the `map_key` column and select
                an approprioate `keep_every` such that each (arm, metric) has at most
                `n` rows in the `map_key` column.
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
        if map_key is None:
            if len(self.map_keys) > 1:
                raise ValueError(
                    "More than one `map_key` found, cannot decide target to subsample."
                )
            map_key = self.map_keys[0]
        subsampled_metric_dfs = []
        for metric_name in self.map_df["metric_name"].unique():
            metric_map_df = _filter_df(self.map_df, metric_names=[metric_name])
            subsampled_metric_dfs.append(
                _subsample_one_metric(
                    metric_map_df,
                    map_key=map_key,
                    keep_every=keep_every,
                    limit_rows_per_group=limit_rows_per_group,
                    limit_rows_per_metric=limit_rows_per_metric,
                    include_first_last=include_first_last,
                )
            )
        subsampled_df: pd.DataFrame = pd.concat(subsampled_metric_dfs)
        return MapData(
            df=subsampled_df,
            map_key_infos=self.map_key_infos,
            description=self.description,
        )


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


def _subsample_one_metric(
    map_df: pd.DataFrame,
    map_key: str | None = None,
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
            df_g = df_g.sort_values(map_key)
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
