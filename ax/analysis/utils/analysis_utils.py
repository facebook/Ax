#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRunType

from ax.core.map_data import MapData
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.trial import BaseTrial
from ax.exceptions.core import DataRequiredError

from ax.service.utils.best_point import _derel_opt_config_wrapper, _is_row_feasible
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from pandas.core.frame import DataFrame

logger: Logger = get_logger(__name__)

FEASIBLE_COL_NAME = "is_feasible"


def _get_generation_method_str(trial: BaseTrial) -> str:
    trial_generation_property = trial._properties.get("generation_model_key")
    if trial_generation_property is not None:
        return trial_generation_property

    generation_methods = {
        not_none(generator_run._model_key)
        for generator_run in trial.generator_runs
        if generator_run._model_key is not None
    }

    # add "Manual" if any generator_runs are manual
    if any(
        generator_run.generator_run_type == GeneratorRunType.MANUAL.name
        for generator_run in trial.generator_runs
    ):
        generation_methods.add("Manual")
    return ", ".join(generation_methods) if generation_methods else "Unknown"


def _merge_trials_dict_with_df(
    df: pd.DataFrame,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    trials_dict: Dict[int, Any],
    column_name: str,
    always_include_field_column: bool = False,
) -> None:
    """Add a column ``column_name`` to a DataFrame ``df`` containing a column
    ``trial_index``. Each value of the new column is given by the element of
    ``trials_dict`` indexed by ``trial_index``.

    Args:
        df: Pandas DataFrame with column ``trial_index``, to be appended with a new
            column.
        trials_dict: Dict mapping each ``trial_index`` to a value. The new column of
            df will be populated with the value corresponding with the
            ``trial_index`` of each row.
        column_name: Name of the column to be appended to ``df``.
        always_include_field_column: Even if all trials have missing values,
            include the column.
    """

    if "trial_index" not in df.columns:
        raise ValueError("df must have trial_index column")

    # field present for some trial
    if always_include_field_column or any(trials_dict.values()):
        if not all(
            v is not None for v in trials_dict.values()
        ):  # not present for all trials
            logger.warning(
                f"Column {column_name} missing for some trials. "
                "Filling with None when missing."
            )
        df[column_name] = [trials_dict[trial_index] for trial_index in df.trial_index]
    else:
        logger.warning(
            f"Column {column_name} missing for all trials. " "Not appending column."
        )


def _merge_results_if_no_duplicates(
    arms_df: pd.DataFrame,
    results: pd.DataFrame,
    key_components: List[str],
    metrics: List[Metric],
) -> DataFrame:
    """Formats ``data.df`` and merges it with ``arms_df`` if all of the following are
    True:
        - ``data.df`` is not empty
        - ``data.df`` contains columns corresponding to ``key_components``
        - after any formatting, ``data.df`` contains no duplicates of the column
            ``results_key_col``
    """
    if len(results.index) == 0:
        logger.info(
            f"No results present for the specified metrics `{metrics}`. "
            "Returning arm parameters and metadata only."
        )
        return arms_df
    if not all(col in results.columns for col in key_components):
        logger.warning(
            f"At least one of key columns `{key_components}` not present in results df "
            f"`{results}`. Returning arm parameters and metadata only."
        )
        return arms_df
    # prepare results for merge by concattenating the trial index with the arm name
    # sparated by a comma
    key_vals = pd.Series(
        results[key_components].values.astype("str").tolist()
    ).str.join(",")

    results_key_col = "-".join(key_components)

    # Reindex so new column isn't set to NaN.
    key_vals.index = results.index
    results[results_key_col] = key_vals
    # Don't return results if duplicates remain
    if any(results.duplicated(subset=[results_key_col, "metric_name"])):
        logger.warning(
            "Experimental results dataframe contains multiple rows with the same "
            f"keys {results_key_col}. Returning dataframe without results."
        )
        return arms_df
    metric_vals = results.pivot(
        index=results_key_col, columns="metric_name", values="mean"
    ).reset_index()

    # dedupe results by key_components
    metadata_cols = key_components + [results_key_col]
    if FEASIBLE_COL_NAME in results.columns:
        metadata_cols.append(FEASIBLE_COL_NAME)
    metadata = results[metadata_cols].drop_duplicates()
    metrics_df = pd.merge(metric_vals, metadata, on=results_key_col)
    # drop synthetic key column
    metrics_df = metrics_df.drop(results_key_col, axis=1)
    # merge and return
    return pd.merge(metrics_df, arms_df, on=key_components, how="outer")


def exp_to_df(
    exp: Experiment,
    metrics: Optional[List[Metric]] = None,
    run_metadata_fields: Optional[List[str]] = None,
    trial_properties_fields: Optional[List[str]] = None,
    trial_attribute_fields: Optional[List[str]] = None,
    additional_fields_callables: Optional[
        Dict[str, Callable[[Experiment], Dict[int, Union[str, float]]]]
    ] = None,
    always_include_field_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transforms an experiment to a DataFrame with rows keyed by trial_index
    and arm_name, metrics pivoted into one row. If the pivot results in more than
    one row per arm (or one row per ``arm * map_keys`` combination if ``map_keys`` are
    present), results are omitted and warning is produced. Only supports
    ``Experiment``.

    Transforms an ``Experiment`` into a ``pd.DataFrame``.

    Args:
        exp: An ``Experiment`` that may have pending trials.
        metrics: Override list of metrics to return. Return all metrics if ``None``.
        run_metadata_fields: Fields to extract from ``trial.run_metadata`` for trial
            in ``experiment.trials``. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial.
        trial_properties_fields: Fields to extract from ``trial._properties`` for trial
            in ``experiment.trials``. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial. Output columns names
            will be prepended with ``"trial_properties_"``.
        trial_attribute_fields: Fields to extract from trial attributes for each trial
            in ``experiment.trials``. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial.
        additional_fields_callables: A dictionary of field names to callables, with
            each being a function from `experiment` to a `trials_dict` of the form
            {trial_index: value}. An example of a custom callable like this is the
            function `compute_maximum_map_values`.
        always_include_field_columns: If `True`, even if all trials have missing
            values, include field columns anyway. Such columns are by default
            omitted (False).
    Returns:
        DataFrame: A dataframe of inputs, metadata and metrics by trial and arm (and
        ``map_keys``, if present). If no trials are available, returns an empty
        dataframe. If no metric ouputs are available, returns a dataframe of inputs and
        metadata. Columns include:
            * trial_index
            * arm_name
            * trial_status
            * generation_method
            * any elements of exp.runner.run_metadata_report_keys that are present in
              the trial.run_metadata of each trial
            * one column per metric (named after the metric.name)
            * one column per parameter (named after the parameter.name)
    """

    if len(kwargs) > 0:
        logger.warning(
            "`kwargs` in exp_to_df is deprecated. Please remove extra arguments."
        )

    # Accept Experiment and SimpleExperiment
    if isinstance(exp, MultiTypeExperiment):
        raise ValueError("Cannot transform MultiTypeExperiments to DataFrames.")

    key_components = ["trial_index", "arm_name"]

    # Get each trial-arm with parameters
    arms_df = pd.DataFrame(
        [
            {
                "arm_name": arm.name,
                "trial_index": trial_index,
                **arm.parameters,
            }
            for trial_index, trial in exp.trials.items()
            for arm in trial.arms
        ]
    )
    # Fetch results.
    data = exp.lookup_data()
    results = data.df

    # Filter metrics.
    if metrics is not None:
        metric_names = [m.name for m in metrics]
        results = results[results["metric_name"].isin(metric_names)]

    # Add `FEASIBLE_COL_NAME` column according to constraints if any.
    if (
        exp.optimization_config is not None
        and len(not_none(exp.optimization_config).all_constraints) > 0
    ):
        optimization_config = not_none(exp.optimization_config)
        try:
            if any(oc.relative for oc in optimization_config.all_constraints):
                optimization_config = _derel_opt_config_wrapper(
                    optimization_config=optimization_config,
                    experiment=exp,
                )
            results[FEASIBLE_COL_NAME] = _is_row_feasible(
                df=results,
                optimization_config=optimization_config,
            )
        except (KeyError, ValueError, DataRequiredError) as e:
            logger.warning(f"Feasibility calculation failed with error: {e}")

    # If arms_df is empty, return empty results (legacy behavior)
    if len(arms_df.index) == 0:
        if len(results.index) != 0:
            raise ValueError(
                "exp.lookup_data().df returned more rows than there are experimental "
                "arms. This is an inconsistent experimental state. Please report to "
                "Ax support."
            )
        return results

    # Create key column from key_components
    arms_df["trial_index"] = arms_df["trial_index"].astype(int)

    # Add trial status
    trials = exp.trials.items()
    trial_to_status = {index: trial.status.name for index, trial in trials}
    _merge_trials_dict_with_df(
        df=arms_df, trials_dict=trial_to_status, column_name="trial_status"
    )

    # Add trial reason for failed or abandoned trials
    trial_to_reason = {
        index: (
            f"{trial.failed_reason[:15]}..."
            if trial.status.is_failed and trial.failed_reason is not None
            else f"{trial.abandoned_reason[:15]}..."
            if trial.status.is_abandoned and trial.abandoned_reason is not None
            else None
        )
        for index, trial in trials
    }

    _merge_trials_dict_with_df(
        df=arms_df,
        trials_dict=trial_to_reason,
        column_name="reason",
    )

    # Add generation_method, accounting for the generic case that generator_runs is of
    # arbitrary length. Repeated methods within a trial are condensed via `set` and an
    # empty set will yield "Unknown" as the method.
    trial_to_generation_method = {
        trial_index: _get_generation_method_str(trial) for trial_index, trial in trials
    }

    _merge_trials_dict_with_df(
        df=arms_df,
        trials_dict=trial_to_generation_method,
        column_name="generation_method",
    )

    # Add any trial properties fields to arms_df
    if trial_properties_fields is not None:
        # add trial._properties fields
        for field in trial_properties_fields:
            trial_to_properties_field = {
                trial_index: (
                    trial._properties[field] if field in trial._properties else None
                )
                for trial_index, trial in trials
            }
            _merge_trials_dict_with_df(
                df=arms_df,
                trials_dict=trial_to_properties_field,
                column_name="trial_properties_" + field,
                always_include_field_column=always_include_field_columns,
            )

    # Add any run_metadata fields to arms_df
    if run_metadata_fields is not None:
        # add run_metadata fields
        for field in run_metadata_fields:
            trial_to_metadata_field = {
                trial_index: (
                    trial.run_metadata[field] if field in trial.run_metadata else None
                )
                for trial_index, trial in trials
            }
            _merge_trials_dict_with_df(
                df=arms_df,
                trials_dict=trial_to_metadata_field,
                column_name=field,
                always_include_field_column=always_include_field_columns,
            )

    # Add any trial attributes fields to arms_df
    if trial_attribute_fields is not None:
        # add trial attribute fields
        for field in trial_attribute_fields:
            trial_to_attribute_field = {
                trial_index: (getattr(trial, field) if hasattr(trial, field) else None)
                for trial_index, trial in trials
            }
            _merge_trials_dict_with_df(
                df=arms_df,
                trials_dict=trial_to_attribute_field,
                column_name=field,
                always_include_field_column=always_include_field_columns,
            )

    # Add additional fields to arms_df
    if additional_fields_callables is not None:
        for field, func in additional_fields_callables.items():
            trial_to_additional_field = func(exp)
            _merge_trials_dict_with_df(
                df=arms_df,
                trials_dict=trial_to_additional_field,
                column_name=field,
                always_include_field_column=always_include_field_columns,
            )

    exp_df = _merge_results_if_no_duplicates(
        arms_df=arms_df,
        results=results,
        key_components=key_components,
        metrics=metrics or list(exp.metrics.values()),
    )

    exp_df = not_none(not_none(exp_df).sort_values(["trial_index"]))
    initial_column_order = (
        ["trial_index", "arm_name", "trial_status", "reason", "generation_method"]
        + (run_metadata_fields or [])
        + (trial_properties_fields or [])
        + ([FEASIBLE_COL_NAME] if FEASIBLE_COL_NAME in exp_df.columns else [])
    )
    for column_name in reversed(initial_column_order):
        if column_name in exp_df.columns:
            # pyre-ignore[6]: In call `DataFrame.insert`, for 3rd positional argument,
            # expected `Union[int, Series, Variable[ArrayLike <: [ExtensionArray,
            # ndarray]]]` but got `Union[DataFrame, Series]`]
            exp_df.insert(0, column_name, exp_df.pop(column_name))
    return exp_df.reset_index(drop=True)


def compute_maximum_map_values(
    experiment: Experiment, map_key: Optional[str] = None
) -> Dict[int, float]:
    """A function that returns a map from trial_index to the maximum map value
    reached. If map_key is not specified, it uses the first map_key."""
    data = experiment.lookup_data()
    if not isinstance(data, MapData):
        raise ValueError("`compute_maximum_map_values` requires `MapData`.")
    if map_key is None:
        map_key = data.map_keys[0]
    map_df = data.map_df
    maximum_map_value_df = (
        map_df[["trial_index"] + data.map_keys]
        .groupby("trial_index")
        .max()
        .reset_index()
    )
    trials_dict = {}
    for trial_index in experiment.trials:
        value = None
        if trial_index in maximum_map_value_df["trial_index"].values:
            value = maximum_map_value_df[
                maximum_map_value_df["trial_index"] == trial_index
            ][map_key].iloc[0]
        trials_dict[trial_index] = value
    return trials_dict
