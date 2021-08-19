#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from typing import Dict, Iterable, Optional, Set, Type

import numpy as np
import pandas as pd
from ax.core.abstract_data import AbstractDataFrameData
from ax.core.types import TFidelityTrialEvaluation, TTrialEvaluation


class Data(AbstractDataFrameData):
    """Class storing data for an experiment.

    The dataframe is retrieved via the `df` property. The data can be stored
    to an external store for future use by attaching it to an experiment using
    `experiment.attach_data()` (this requires a description to be set.)


    Attributes:
        df: DataFrame with underlying data, and required columns.
        description: Human-readable description of data.

    """

    # Note: Although the SEM (standard error of the mean) is a required column in data,
    # downstream models can infer missing SEMs. Simply specify NaN as the SEM value,
    # either in your Metric class or in Data explicitly.
    REQUIRED_COLUMNS = {"arm_name", "metric_name", "mean", "sem"}
    # pyre-ignore[15]: Inconsistent override. Adds FieldExperiment-specific fields
    COLUMN_DATA_TYPES = {
        "arm_name": str,
        "metric_name": str,
        "mean": np.float64,
        "sem": np.float64,
        "trial_index": np.int64,
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "n": np.int64,
        "frac_nonnull": np.float64,
        "random_split": np.int64,
        "fidelities": str,  # Dictionary stored as json
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
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
            columns = set(df.columns)
            missing_columns = self.required_columns() - columns
            if missing_columns:
                raise ValueError(
                    f"Dataframe must contain required columns {list(missing_columns)}."
                )
            extra_columns = columns - self.supported_columns()
            if extra_columns:
                raise ValueError(f"Columns {list(extra_columns)} are not supported.")
            df = df.dropna(axis=0, how="all").reset_index(drop=True)
            df = self._safecast_df(df=df)

            # Reorder the columns for easier viewing
            col_order = [c for c in self.column_data_types() if c in df.columns]
            self._df = df[col_order]
        super().__init__(description=description)

    @staticmethod
    # pyre-ignore [14]: `Iterable[Data]` not a supertype of overridden parameter.
    def from_multiple_data(
        data: Iterable[Data], subset_metrics: Optional[Iterable[str]] = None
    ) -> Data:
        """Combines multiple data objects into one (with the concatenated
        underlying dataframe).

        NOTE: if one or more data objects in the iterable is of a custom
        subclass of `Data`, object of that class will be returned. If
        the iterable contains multiple types of `Data`, an error will be
        raised.

        Args:
            data: Iterable of Ax `Data` objects to combine.
            subset_metrics: If specified, combined `Data` will only contain
                metrics, names of which appear in this iterable,
                in the underlying dataframe.
        """
        dfs = [datum.df for datum in data]

        if len(dfs) == 0:
            return Data()

        if subset_metrics:
            dfs = [df.loc[df["metric_name"].isin(subset_metrics)] for df in dfs]

        # obtain type of first elt in iterable (we know it's not empty)
        data_type = type(data[0])

        # check if all types in iterable match the first type
        if all((type(datum) is data_type) for datum in data):
            # if all types in iterable are subclasses of Data, return the subclass
            if issubclass(data_type, Data):
                return data_type(df=pd.concat(dfs, axis=0, sort=True))
            else:
                # if not, return the original Data object
                return Data(df=pd.concat(dfs, axis=0, sort=True))
        else:
            raise ValueError("More than one custom data type found in data iterable")

    @staticmethod
    def from_evaluations(
        evaluations: Dict[str, TTrialEvaluation],
        trial_index: int,
        sample_sizes: Optional[Dict[str, int]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Data:
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to metric outcomes (itself a mapping
                of metric names to tuples of mean and optionally a SEM).
            trial_index: Trial index to which this data belongs.
            sample_sizes: Number of samples collected for each arm.
            start_time: Optional start time of run of the trial that produced this
                data, in milliseconds.
            end_time: Optional end time of run of the trial that produced this
                data, in milliseconds.

        Returns:
            Ax Data object.
        """
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else 0.0,
                "trial_index": trial_index,
            }
            for name, evaluation in evaluations.items()
            for metric_name, value in evaluation.items()
        ]
        if start_time is not None or end_time is not None:
            for record in records:
                record.update({"start_time": start_time, "end_time": end_time})
        if sample_sizes:
            for record in records:
                record["n"] = sample_sizes[str(record["arm_name"])]
        return Data(df=pd.DataFrame(records))

    @staticmethod
    def from_fidelity_evaluations(
        evaluations: Dict[str, TFidelityTrialEvaluation],
        trial_index: int,
        sample_sizes: Optional[Dict[str, int]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Data:
        """
        Convert dict of fidelity evaluations to Ax data object.

        Args:
            evaluations: Map from arm name to list of (fidelity, metric outcomes)
                (where metric outcomes is itself a mapping of metric names to
                tuples of mean and SEM).
            trial_index: Trial index to which this data belongs.
            sample_sizes: Number of samples collected for each arm.
            start_time: Optional start time of run of the trial that produced this
                data, in milliseconds.
            end_time: Optional end time of run of the trial that produced this
                data, in milliseconds.

        Returns:
            Ax Data object.
        """
        records = [
            {
                "arm_name": name,
                "metric_name": metric_name,
                "mean": value[0] if isinstance(value, tuple) else value,
                "sem": value[1] if isinstance(value, tuple) else 0.0,
                "trial_index": trial_index,
                "fidelities": json.dumps(fidelity),
            }
            for name, fidelity_and_metrics_list in evaluations.items()
            for fidelity, evaluation in fidelity_and_metrics_list
            for metric_name, value in evaluation.items()
        ]
        if start_time is not None or end_time is not None:
            for record in records:
                record.update({"start_time": start_time, "end_time": end_time})
        if sample_sizes:
            for record in records:
                record["n"] = sample_sizes[str(record["arm_name"])]
        return Data(df=pd.DataFrame(records))


def set_single_trial(data: Data) -> Data:
    """Returns a new Data object where we set all rows to have the same
    trial index (i.e. 0). This is meant to be used with our IVW transform,
    which will combine multiple observations of the same metric.
    """
    df = data._df.copy()
    if "trial_index" in df:
        df["trial_index"] = 0
    return Data(df=df)


def clone_without_metrics(data: Data, excluded_metric_names: Iterable[str]) -> Data:
    """Returns a new data object where rows containing the metrics specified by
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
        required_columns: Set of additional columns required for this data object.
        time_columns: Set of additional columns to cast to timestamp.

    Returns:
        New data subclass with amended column definitions.
    """

    class CustomData(Data):
        @classmethod
        def required_columns(cls) -> Set[str]:
            return (required_columns or set()).union(Data.REQUIRED_COLUMNS)

        @classmethod
        def column_data_types(
            cls, extra_column_types: Optional[Dict[str, Type]] = None
        ) -> Dict[str, Type]:
            return super().column_data_types(
                {**(extra_column_types or {}), **(column_data_types or {})}
            )

    return CustomData
