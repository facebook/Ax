#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from ax.core.abstract_data import AbstractDataFrameData
from ax.core.data import Data
from ax.core.types import TMapTrialEvaluation


class MapData(AbstractDataFrameData):
    """Class storing mapping-like results for an experiment.

    Mapping-like results occur whenever a metric is reported as a collection
    of results, each element corresponding to a tuple of values.

    The simplest case is a sequence. For instance a time series is
    a mapping from the 1-tuple `(timestamp)` to (mean, sem) results.

    Another example: MultiFidelity results. This is a mapping from
    `(fidelity_feature_1, ..., fidelity_feature_n)` to (mean, sem) results.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)
    """

    # Note: Although the SEM (standard error of the mean) is a required column in data,
    # downstream models can infer missing SEMs. Simply specify NaN as the SEM value,
    # either in your Metric class or in Data explicitly.
    REQUIRED_COLUMNS = {"arm_name", "metric_name", "mean", "sem"}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        map_keys: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Init `MapData`.

        Args:
            df: DataFrame with underlying data, and required columns.
            map_keys: List of all elements of the Tuple that makes up the
                key in MapData.
            description: Human-readable description of data.

        """
        if map_keys is None and df is not None:
            raise ValueError(
                "map_keys may only be `None` when `df` is also None "
                "(an empty `MapData`)."
            )
        self._map_keys = map_keys or []
        # Represent MapData internally as a flat `DataFrame`
        # Make an empty `DataFrame with map_keys if available`
        if df is None:
            self._df = pd.DataFrame(
                columns=self.required_columns().union(self.map_keys)
            )
        else:
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            extra_columns = columns - self.supported_columns(
                extra_column_names=self.map_keys
            )
            if extra_columns:
                raise ValueError(f"Columns {list(extra_columns)} are not supported.")
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            df = self._safecast_df(df=df, extra_column_types=self.map_key_types)

            # Reorder the columns for easier viewing
            col_order = [
                c for c in self.column_data_types(self.map_key_types) if c in df.columns
            ]
            self._df = df[col_order]
        self.description = description

    @staticmethod
    # pyre-ignore [14]: `Iterable[Data]` not a supertype of overridden parameter.
    def from_multiple_data(
        data: Iterable[MapData], subset_metrics: Optional[Iterable[str]] = None
    ) -> MapData:
        """Combines multiple data objects into one (with the concatenated
        underlying dataframe).

        NOTE: if one or more data objects in the iterable is of a custom
        subclass of `MapData`, object of that class will be returned. If
        the iterable contains multiple types of `Data`, an error will be
        raised.

        Args:
            data: Iterable of Ax `MapData` objects to combine.
            subset_metrics: If specified, combined `MapData` will only contain
                metrics, names of which appear in this iterable,
                in the underlying dataframe.
        """
        # Filter out empty dataframes because they may not have correct map_keys.
        data = [datum for datum in data if not datum.df.empty]
        dfs = [datum.df for datum in data if not datum.df.empty]

        if len(dfs) == 0:
            return MapData()

        if subset_metrics:
            dfs = [df.loc[df["metric_name"].isin(subset_metrics)] for df in dfs]

        # cast to list
        data = list(data)
        if not all((type(datum) is MapData) for datum in data):
            # check if all types in iterable match the first type
            raise ValueError("Non-MapData in inputs.")
        # obtain map_keys of first elt in iterable (we know it's not empty)
        map_keys = data[0].map_keys

        if not all((set(datum.map_keys) == set(map_keys)) for datum in data):
            raise ValueError("Inconsistent map_keys found in data iterable.")
        else:
            # if all validation is passed return concatenated data.
            return MapData(df=pd.concat(dfs, axis=0, sort=True), map_keys=map_keys)

    @property
    def map_keys(self):
        """Return the names of fields that together make a map key.

        E.g. ["timestamp"] for a timeseries, ["fidelity_param_1", "fidelity_param_2"]
        for a multi-fidelity set of results.
        """
        return self._map_keys

    @property
    def map_key_types(self):
        return {map_key: Any for map_key in self.map_keys}

    def update(self, new_data: MapData) -> None:
        if not new_data.map_keys == self.map_keys:
            raise ValueError("Inconsistent map_keys found in new data.")
        self._df = self.df.append(new_data.df)

    @staticmethod
    def from_map_evaluations(
        evaluations: Dict[str, TMapTrialEvaluation],
        trial_index: int,
        map_keys: Optional[List[str]] = None,
    ) -> MapData:
        """
        Convert dict of mapped evaluations to an Ax MapData object

        Args:
            evaluations: Map from arm name to metric outcomes (itself a mapping
                of metric names to tuples of mean and optionally a SEM).
            trial_index: Trial index to which this data belongs.
            map_keys: List of all elements of the Tuple that makes up the
                key in MapData.

        Returns:
            Ax MapData object.
        """
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
        map_keys_list = [
            list(map_dict.keys())
            for name, map_dict_and_metrics_list in evaluations.items()
            for map_dict, evaluation in map_dict_and_metrics_list
        ]
        map_keys = map_keys or map_keys_list[0]
        if not all((set(mk) == set(map_keys)) for mk in map_keys_list):
            raise ValueError("Inconsistent map_key sets in evaluations.")
        return MapData(df=pd.DataFrame(records), map_keys=map_keys)

    def to_standard_data(self) -> Data:
        return Data(df=self.df)
