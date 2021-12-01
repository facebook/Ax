# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar, Generic, Iterable, Optional

import pandas as pd
from ax.core.data import Data
from ax.core.types import TMapTrialEvaluation
from ax.exceptions.core import UnsupportedError
from ax.utils.common.base import SortableBase
from ax.utils.common.equality import dataframe_equals
from ax.utils.common.logger import get_logger

logger = get_logger(__name__)


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
    def value_type(self) -> Type:
        return type(self._default_value)


class MapData(Data):
    """Class storing mapping-like results for an experiment.

    Data is stored in a dataframe, and axilary information ((key name,
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

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        map_key_infos: Optional[Iterable[MapKeyInfo]] = None,
        description: Optional[str] = None,
    ) -> None:
        if map_key_infos is None and df is not None:
            raise ValueError("map_key_infos may be `None` iff `df` is None.")

        self._map_key_infos = map_key_infos or []

        if df is None:  # If df is None create an empty dataframe with appropriate cols
            self._map_df = pd.DataFrame(
                columns=self.required_columns().union(self.map_keys)
            )
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
    def true_df(self):
        return self.map_df

    @property
    def map_key_infos(self) -> Iterable[MapKeyInfo]:
        return self._map_key_infos

    @property
    def map_keys(self) -> Iterable[str]:
        return [mki.key for mki in self.map_key_infos]

    @property
    def map_key_to_type(self) -> Dict[str, Type]:
        return {mki.key: mki.value_type for mki in self.map_key_infos}

    @staticmethod
    def from_multiple_map_data(
        data: Iterable[MapData],
        subset_metrics: Optional[Iterable[str]] = None,
    ) -> MapData:
        unique_map_key_infos = []
        for mki in (mki for datum in data for mki in datum.map_key_infos):
            if any(
                mki.key == unique.key and mki.default_value != unique.default_value
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

        subset_metrics_mask = df["metric_name"].isin(
            subset_metrics if subset_metrics else df["metric_name"]
        )

        return MapData(df=df[subset_metrics_mask], map_key_infos=unique_map_key_infos)

    @staticmethod
    def from_map_evaluations(
        evaluations: Dict[str, TMapTrialEvaluation],
        trial_index: int,
        map_key_infos: Optional[Iterable[MapKeyInfo]] = None,
    ) -> MapData:
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else 0.0,
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
            MapKeyInfo(key=key, default_value=0.0) for key in map_keys
        ]

        if {mki.key for mki in map_key_infos} != map_keys:
            raise ValueError("Inconsistent map_key sets in evaluations.")

        return MapData(df=pd.DataFrame(records), map_key_infos=map_key_infos)

    @property
    def map_df(self) -> pd.DataFrame:
        return self._map_df

    @map_df.setter
    def map_df(self, df: pd.DataFrame):
        raise UnsupportedError(
            "MapData's underlying DataFrame is immutable; create a new"
            + " MapData via `__init__` or `from_multiple_data`."
        )

    @staticmethod
    def from_multiple_data(
        data: Iterable[Data],
        subset_metrics: Optional[Iterable[str]] = None,
    ) -> MapData:
        """Downcast instances of Data into instances of MapData with empty
        map_key_infos if necessary then combine as usual (filling in empty cells with
        default values).
        """
        map_datas = [
            MapData(df=datum.df, map_key_infos=[])
            if not isinstance(datum, MapData)
            else datum
            for datum in data
        ]

        return MapData.from_multiple_map_data(
            data=map_datas, subset_metrics=subset_metrics
        )

    @property
    def df(self) -> pd.DataFrame:
        """Returns a Data shaped DataFrame"""

        # If map_keys is empty just return the df
        if self._memo_df is not None:
            return self._memo_df

        if not any(True for _ in self.map_keys):
            return self.map_df

        self._memo_df = (
            self.map_df.sort_values(list(self.map_keys))
            .drop_duplicates(MapData.DEDUPLICATE_BY_COLUMNS, keep="last")
            .loc[:, ~self.map_df.columns.isin(self.map_keys)]
        )

        return self._memo_df

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.
        """

        return {
            "map_key_infos": [
                MapKeyInfo(d["key"], d["default_value"]) for d in args["map_key_infos"]
            ]
        }
