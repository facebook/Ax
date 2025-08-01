#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import timedelta
from logging import Logger
from typing import Any, cast, TYPE_CHECKING

import gpytorch
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from ax.adapter import Adapter
from ax.adapter.cross_validation import (
    compute_model_fit_metrics_from_adapter,
    cross_validate,
)
from ax.adapter.random import RandomAdapter
from ax.adapter.torch import TorchAdapter
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRunType
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import Parameter
from ax.core.trial import BaseTrial
from ax.core.trial_status import TrialStatus
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.plot.contour import interact_contour_plotly
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.plot.feature_importances import plot_feature_importance_by_feature_plotly
from ax.plot.helper import get_range_parameters_from_list
from ax.plot.pareto_frontier import (
    _pareto_frontier_plot_input_processing,
    _validate_experiment_and_get_optimization_config,
    scatter_plot_with_hypervolume_trace_plotly,
    scatter_plot_with_pareto_frontier_plotly,
)
from ax.plot.pareto_utils import _extract_observed_pareto_2d
from ax.plot.scatter import interact_fitted_plotly, plot_multiple_metrics
from ax.plot.slice import interact_slice_plotly
from ax.plot.trace import (
    map_data_multiple_metrics_dropdown_plotly,
    plot_objective_value_vs_trial_index,
)
from ax.service.utils.best_point import _is_row_feasible, derelativize_opt_config
from ax.service.utils.best_point_utils import select_baseline_name_default_first_trial
from ax.service.utils.early_stopping import get_early_stopping_metrics
from ax.utils.common.logger import get_logger
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from pandas.core.frame import DataFrame
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    from ax.service.orchestrator import Orchestrator


logger: Logger = get_logger(__name__)
FEATURE_IMPORTANCE_CAPTION = (
    "<b>NOTE:</b> This plot is intended for advanced users. Specifically,<br>"
    "it is a measure of sensitivity/smoothness, so parameters of<br>"
    "relatively low importance may still be important to tune."
)
CROSS_VALIDATION_CAPTION = (
    "<b>NOTE:</b> We have tried our best to only plot the region of interest.<br>"
    "This may hide outliers. You can autoscale the axes to see all trials."
)
FEASIBLE_COL_NAME = "is_feasible"
BASELINE_ARM_NAME = "baseline_arm"
UNPREDICTABLE_METRICS_MESSAGE = (
    "The following metric(s) are behaving unpredictably and may be noisy or "
    "misconfigured: {}. Please check that they are measuring the intended quantity, "
    "and are expected to vary reliably as a function of your parameters."
)


def _get_cross_validation_plots(model: Adapter) -> list[go.Figure]:
    cv = cross_validate(model=model)
    return [
        interact_cross_validation_plotly(
            cv_results=cv, caption=CROSS_VALIDATION_CAPTION
        )
    ]


def _get_objective_trace_plot(
    experiment: Experiment,
    data: Data,
    true_objective_metric_name: str | None = None,
) -> Iterable[go.Figure]:
    if experiment.is_moo_problem:
        return [
            scatter_plot_with_hypervolume_trace_plotly(experiment=experiment),
            *_pairwise_pareto_plotly_scatter(experiment=experiment),
        ]
    runner = experiment.runner
    run_metadata_report_keys = None
    if runner is not None:
        run_metadata_report_keys = runner.run_metadata_report_keys
    exp_df = exp_to_df(exp=experiment, run_metadata_fields=run_metadata_report_keys)

    optimization_config = experiment.optimization_config
    if optimization_config is None:
        return []

    metric_names = (
        metric_name
        for metric_name in [
            optimization_config.objective.metric.name,
            true_objective_metric_name,
        ]
        if metric_name is not None
    )

    plots = [
        plot_objective_value_vs_trial_index(
            exp_df=exp_df,
            metric_colname=metric_name,
            minimize=none_throws(
                optimization_config.objective.minimize
                if optimization_config.objective.metric.name == metric_name
                else experiment.metrics[metric_name].lower_is_better
            ),
            title=f"Best {metric_name} found vs. trial index",
            hover_data_colnames=run_metadata_report_keys,
        )
        for metric_name in metric_names
    ]

    return [plot for plot in plots if plot is not None]


def _get_objective_v_param_plots(
    experiment: Experiment,
    model: Adapter,
    importance: None
    | (dict[str, dict[str, npt.NDArray]] | dict[str, dict[str, float]]) = None,
    # Chosen to take ~1min on local benchmarks.
    max_num_slice_plots: int = 200,
    # Chosen to take ~2min on local benchmarks.
    max_num_contour_plots: int = 20,
) -> list[go.Figure]:
    search_space = experiment.search_space

    range_params = [
        assert_is_instance(param, Parameter)
        for param in search_space.range_parameters.values()
    ]
    range_params = get_range_parameters_from_list(range_params, min_num_values=5)
    if len(range_params) < 1:
        # if search space contains no range params
        logger.warning(
            "`_get_objective_v_param_plot` requires a search space with at least one "
            "`RangeParameter`. Returning an empty list."
        )
        return []
    range_param_names = [param.name for param in range_params]
    num_range_params = len(range_params)
    num_metrics = len(model.metric_names)
    num_slice_plots = num_range_params * num_metrics
    output_plots = []
    if num_slice_plots <= max_num_slice_plots:
        # parameter slice plot
        output_plots += [
            interact_slice_plotly(
                model=model,
            )
        ]
    else:
        warning_msg = (
            f"Skipping creation of {num_slice_plots} slice plots since that "
            f"exceeds <br>`max_num_slice_plots = {max_num_slice_plots}`."
            "<br>Users can plot individual slice plots with the <br>python "
            "function ax.plot.slice.plot_slice_plotly."
        )
        # TODO: return a warning here then convert to a plot/message/etc. downstream.
        warning_plot = _warn_and_create_warning_plot(warning_msg=warning_msg)
        output_plots.append(warning_plot)

    # contour plots
    num_contour_per_metric = max_num_contour_plots // num_metrics
    if num_contour_per_metric < 2:
        warning_msg = (
            "Skipping creation of contour plots since that requires <br>"
            "`max_num_contour_plots >= 2 * num_metrics`. Got "
            f"{max_num_contour_plots=} and {num_metrics=}."
            "<br>Users can plot individual contour plots with the <br>python "
            "function ax.plot.contour.plot_contour_plotly."
        )
        # TODO: return a warning here then convert to a plot/message/etc. downstream.
        warning_plot = _warn_and_create_warning_plot(warning_msg=warning_msg)
        output_plots.append(warning_plot)
    elif num_range_params > 1:
        # Using n params yields n * (n - 1) contour plots, so we use the number of
        # params that yields the desired number of plots (solved using quadratic eqn)
        num_params_per_metric = int(0.5 + (0.25 + num_contour_per_metric) ** 0.5)
        try:
            for metric_name in model.metric_names:
                if importance is not None:
                    range_params_sens_for_metric = {
                        k: v
                        for k, v in importance[metric_name].items()
                        if k in range_param_names
                    }
                    # sort the params by their sensitivity
                    params_to_use = sorted(
                        range_params_sens_for_metric,
                        # pyre-fixme[6]: For 2nd argument expected `None` but got
                        #  `(x: Any) -> Union[ndarray[typing.Any, typing.Any], float]`.
                        key=lambda x: range_params_sens_for_metric[x],
                        reverse=True,
                    )[:num_params_per_metric]
                # if sens is not available, just use the first num_features_per_metric.
                else:
                    params_to_use = range_param_names[:num_params_per_metric]
                with gpytorch.settings.max_eager_kernel_size(float("inf")):
                    output_plots.append(
                        interact_contour_plotly(
                            model=none_throws(model),
                            metric_name=metric_name,
                            parameters_to_use=params_to_use,
                        )
                    )
                logger.debug(
                    f"Created contour plots for metric {metric_name} and parameters "
                    f"{params_to_use}."
                )
        # `mean shape torch.Size` RunTimeErrors, pending resolution of
        # https://github.com/cornellius-gp/gpytorch/issues/1853
        except RuntimeError as e:
            logger.warning(f"Contour plotting failed with error: {e}.")
    return output_plots


def _get_suffix(input_str: str, delim: str = ".", n_chunks: int = 1) -> str:
    return delim.join(input_str.split(delim)[-n_chunks:])


def _get_shortest_unique_suffix_dict(
    input_str_list: list[str], delim: str = "."
) -> dict[str, str]:
    """Maps a list of strings to their shortest unique suffixes

    Maps all original strings to the smallest number of chunks, as specified by
    delim, that are not a suffix of any other original string. If the original
    string was a suffix of another string, map it to its unaltered self.

    Args:
        input_str_list: a list of strings to create the suffix mapping for
        delim: the delimiter used to split up the strings into meaningful chunks

    Returns:
        dict: A dict with the original strings as keys and their abbreviations as
            values
    """

    # all input strings must be unique
    assert len(input_str_list) == len(set(input_str_list))
    if delim == "":
        raise ValueError("delim must be a non-empty string.")
    suffix_dict = defaultdict(list)
    # initialize suffix_dict with last chunk
    for istr in input_str_list:
        suffix_dict[_get_suffix(istr, delim=delim, n_chunks=1)].append(istr)
    max_chunks = max(len(istr.split(delim)) for istr in input_str_list)
    if max_chunks == 1:
        return {istr: istr for istr in input_str_list}
    # the upper range of this loop is `max_chunks + 2` because:
    #     - `i` needs to take the value of `max_chunks`, hence one +1
    #     - the contents of the loop are run one more time to check if `all_unique`,
    #           hence the other +1
    for i in range(2, max_chunks + 2):
        new_dict = defaultdict(list)
        all_unique = True
        for suffix, suffix_str_list in suffix_dict.items():
            if len(suffix_str_list) > 1:
                all_unique = False
                for istr in suffix_str_list:
                    new_dict[_get_suffix(istr, delim=delim, n_chunks=i)].append(istr)
            else:
                new_dict[suffix] = suffix_str_list
        if all_unique:
            if len(set(input_str_list)) != len(suffix_dict.keys()):
                break
            return {
                suffix_str_list[0]: suffix
                for suffix, suffix_str_list in suffix_dict.items()
            }
        suffix_dict = new_dict
    # If this function has not yet exited, some input strings still share a suffix.
    # This is not expected, but in this case, the function will return the identity
    # mapping, i.e., a dict with the original strings as both keys and values.
    logger.warning(
        "Something went wrong. Returning dictionary with original strings as keys and "
        "values."
    )
    return {istr: istr for istr in input_str_list}


def get_standard_plots(
    experiment: Experiment,
    model: Adapter | None,
    data: Data | None = None,
    true_objective_metric_name: str | None = None,
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
    limit_points_per_plot: int | None = None,
    global_sensitivity_analysis: bool = True,
) -> list[go.Figure]:
    """Extract standard plots for single-objective optimization.

    Extracts a list of plots from an ``Experiment`` and ``Adapter`` of general
    interest to an Ax user. Currently not supported are
    - TODO: multi-objective optimization
    - TODO: ChoiceParameter plots

    Args:
        - experiment: The ``Experiment`` from which to obtain standard plots.
        - model: The ``Adapter`` used to suggest trial parameters.
        - true_objective_metric_name: Name of the metric to use as the true objective.
        - early_stopping_strategy: Early stopping strategy used throughout the
            experiment; used for visualizing when curves are stopped.
        - limit_points_per_plot: Limit the number of points used per metric in
            each curve plot. Passed to `_get_curve_plot_dropdown`.
        - global_sensitivity_analysis: If True, plot total Variance-based sensitivity
            analysis for the model parameters. If False, plot sensitivities based on
            GP kernel lengthscales. Defaults to True.
    Returns:
        - a plot of objective value vs. trial index, to show experiment progression
        - a plot of objective value vs. range parameter values, only included if the
          model associated with generation_strategy can create predictions. This
          consists of:

            - a plot_slice plot if the search space contains one range parameter
            - an interact_contour plot if the search space contains multiple
              range parameters

    """
    if (
        true_objective_metric_name is not None
        and true_objective_metric_name not in experiment.metrics.keys()
    ):
        raise ValueError(
            f"true_objective_metric_name='{true_objective_metric_name}' is not present "
            f"in experiment.metrics={experiment.metrics}. Please add a valid "
            "true_objective_metric_name or remove the optional parameter to get "
            "standard plots."
        )

    objective = none_throws(experiment.optimization_config).objective
    if isinstance(objective, ScalarizedObjective):
        logger.warning(
            "get_standard_plots does not currently support ScalarizedObjective "
            "optimization experiments. Returning an empty list."
        )
        return []

    if data is None:
        data = experiment.fetch_data()

    if data.df.empty:
        logger.info(f"Experiment {experiment} does not yet have data, nothing to plot.")
        return []

    output_plot_list = []
    try:
        output_plot_list.extend(
            _get_objective_trace_plot(
                experiment=experiment,
                data=data,
                true_objective_metric_name=true_objective_metric_name,
            )
        )
    except Exception as e:
        # Allow model-based plotting to proceed if objective_trace plotting fails.
        logger.exception(f"Plotting `objective_trace` failed with error {e}")

    # Objective vs. parameter plot requires a `Model`, so add it only if model
    # is alrady available. In cases where initially custom trials are attached,
    # model might not yet be set on the generation strategy. Additionally, if
    # the model is a RandomAdapter, skip plots that require predictions.
    if model is not None and not isinstance(model, RandomAdapter):
        try:
            if true_objective_metric_name is not None:
                logger.debug("Starting objective vs. true objective scatter plot.")
                output_plot_list.append(
                    _objective_vs_true_objective_scatter(
                        model=model,
                        objective_metric_name=objective.metric_names[0],
                        true_objective_metric_name=true_objective_metric_name,
                    )
                )
                logger.debug("Finished with objective vs. true objective scatter plot.")
        except Exception as e:
            logger.exception(f"Scatter plot failed with error: {e}")

        # Compute feature importance ("sensitivity") to select most important
        # features to plot.
        sens = None
        importance_measure = ""
        if global_sensitivity_analysis and isinstance(model, TorchAdapter):
            try:
                logger.debug("Starting global sensitivity analysis.")
                sens = ax_parameter_sens(model, order="total")
                importance_measure = (
                    '<a href="https://en.wikipedia.org/wiki/Variance-based_'
                    'sensitivity_analysis">Variance-based sensitivity analysis</a>'
                )
                logger.debug("Finished global sensitivity analysis.")
            except Exception as e:
                logger.debug(
                    f"Failed to compute signed global feature sensitivities: {e}"
                    "Trying to get unsigned feature sensitivities."
                )
                try:
                    sens = ax_parameter_sens(model, order="total", signed=False)
                except Exception as e:
                    logger.exception(f"Failed to compute feature sensitivities: {e}")
        if sens is None:
            try:
                sens = {
                    metric_name: model.feature_importances(metric_name)
                    for i, metric_name in enumerate(sorted(model.metric_names))
                }
            except Exception as e:
                logger.warning(f"Failed to compute feature importances: {e}")

        try:
            logger.debug("Starting objective vs. param plots.")
            # importance is the absolute value of sensitivity.
            importance = None
            if sens is not None:
                importance = {
                    k: {j: np.absolute(sens[k][j]) for j in sens[k].keys()}
                    for k in sens.keys()
                }
            output_plot_list.extend(
                _get_objective_v_param_plots(
                    experiment=experiment,
                    model=model,
                    importance=importance,
                )
            )
            logger.debug("Finished objective vs. param plots.")
        except Exception as e:
            logger.exception(f"Slice plot failed with error: {e}")

        try:
            logger.debug("Starting cross validation plot.")
            output_plot_list.extend(_get_cross_validation_plots(model=model))
            logger.debug("Finished cross validation plot.")
        except Exception as e:
            logger.exception(f"Cross-validation plot failed with error: {e}")

        # sensitivity plot
        try:
            logger.debug("Starting feature importance plot.")
            feature_importance_plot = plot_feature_importance_by_feature_plotly(
                model=model,
                # pyre-ignore [6]:
                # In call for argument `sensitivity_values`, expected
                # `Optional[Dict[str, Dict[str, Union[float, ndarray]]]]`
                # but got `Dict[str, Dict[str, ndarray]]`.
                sensitivity_values=sens,
                relative=False,
                caption=FEATURE_IMPORTANCE_CAPTION if importance_measure == "" else "",
                importance_measure=importance_measure,
            )
            logger.debug("Finished feature importance plot.")
            feature_importance_plot.layout.title = "[ADVANCED] " + str(
                feature_importance_plot.layout.title.text
            )
            output_plot_list.append(feature_importance_plot)
            output_plot_list.append(interact_fitted_plotly(model=model, rel=False))
        except Exception as e:
            logger.exception(f"Feature importance plot failed with error: {e}")

    # Get plots for MapMetrics
    try:
        logger.debug("Starting MapMetric plots.")
        map_metrics = [
            m for m in experiment.metrics.values() if isinstance(m, MapMetric)
        ]
        if map_metrics:
            # Sort so that objective metrics appear first
            map_metrics.sort(
                key=lambda e: e.name in [m.name for m in objective.metrics],
                reverse=True,
            )
            for by_walltime in [False, True]:
                logger.debug(f"Starting MapMetric plot {by_walltime=}.")
                output_plot_list.append(
                    _get_curve_plot_dropdown(
                        experiment=experiment,
                        map_metrics=map_metrics,
                        data=data,  # pyre-ignore
                        early_stopping_strategy=early_stopping_strategy,
                        by_walltime=by_walltime,
                        limit_points_per_plot=limit_points_per_plot,
                    )
                )
                logger.debug(f"Finished MapMetric plot {by_walltime=}.")
        logger.debug("Finished MapMetric plots.")
    except Exception as e:
        logger.exception(f"Curve plot failed with error: {e}")
    logger.debug("Returning plots.")
    return [plot for plot in output_plot_list if plot is not None]


def _transform_progression_to_walltime(
    progressions: npt.NDArray,
    exp_df: pd.DataFrame,
    trial_idx: int,
) -> npt.NDArray | None:
    try:
        trial_df = exp_df[exp_df["trial_index"] == trial_idx]
        time_run_started = trial_df["time_run_started"].iloc[0]
        time_completed = trial_df["time_completed"].iloc[0]
        runtime_seconds = (time_completed - time_run_started).total_seconds()
        intermediate_times = runtime_seconds * progressions / progressions.max()
        transformed_times = np.array(
            [time_run_started + timedelta(seconds=t) for t in intermediate_times]
        )
        return transformed_times
    except Exception as e:
        logger.error(f"Failed to transform progression to walltime: {e}")
        return None


def _get_curve_plot_dropdown(
    experiment: Experiment,
    map_metrics: Iterable[MapMetric],
    data: MapData,
    early_stopping_strategy: BaseEarlyStoppingStrategy | None,
    by_walltime: bool = False,
    limit_points_per_plot: int | None = None,
) -> go.Figure | None:
    """Plot curve metrics by either progression or walltime.

    Args:
        experiment: The experiment to generate plots for.
        map_metrics: The list of metrics to generate plots for. Each metric
            will be one entry in the dropdown.
        data: The map data used to generate the plots.
        early_stopping_strategy: An instance of ``BaseEarlyStoppingStrategy``. This
            is used to check which metrics are being used for early stopping.
        by_walltime: If true, the x-axis will be walltime. If false, the x-axis is
            the progression of the trials (trials are 'stacked').
        limit_points_per_plot: Limit the total number of data points used per plot
            (i.e., per metric). This is passed down to `MapData.subsample(...)` to
            subsample the data. Useful for keeping the plots of manageable size.
    """
    early_stopping_metrics = get_early_stopping_metrics(
        experiment=experiment, early_stopping_strategy=early_stopping_strategy
    )
    xs_by_metric = {}
    ys_by_metric = {}
    legend_labels_by_metric = {}
    stopping_markers_by_metric = {}
    exp_df = pd.DataFrame()
    if by_walltime:
        exp_df = exp_to_df(
            exp=experiment,
            trial_attribute_fields=["time_run_started", "time_completed"],
            always_include_field_columns=True,
        )
    for m in map_metrics:
        map_key = m.map_key_info.key
        subsampled_data = (
            data
            if limit_points_per_plot is None
            else data.subsample(
                limit_rows_per_metric=limit_points_per_plot, map_key=map_key
            )
        )
        map_df = subsampled_data.map_df
        metric_df = map_df[map_df["metric_name"] == m.name]
        xs, ys, legend_labels, plot_stopping_markers = [], [], [], []
        is_early_stopping_metric = m.name in early_stopping_metrics
        for trial_idx, df_g in metric_df.groupby("trial_index"):
            if experiment.trials[trial_idx].status not in (
                TrialStatus.COMPLETED,
                TrialStatus.EARLY_STOPPED,
            ):
                continue
            if by_walltime:
                x = _transform_progression_to_walltime(
                    progressions=df_g[map_key].to_numpy(),
                    exp_df=exp_df,
                    trial_idx=trial_idx,
                )
                if x is None:
                    continue
            else:
                x = df_g[map_key].to_numpy()
            xs.append(x)
            ys.append(df_g["mean"].to_numpy())
            legend_labels.append(f"Trial {trial_idx}")
            plot_stopping_markers.append(
                is_early_stopping_metric
                and experiment.trials[trial_idx].status == TrialStatus.EARLY_STOPPED
            )

        if len(xs) > 0:
            xs_by_metric[m.name] = xs
            ys_by_metric[m.name] = ys
            legend_labels_by_metric[m.name] = legend_labels
            stopping_markers_by_metric[m.name] = plot_stopping_markers

    if len(xs_by_metric.keys()) == 0:
        return None

    title = (
        "Curve metrics (i.e., learning curves) by walltime"
        if by_walltime
        else "Curve metrics (i.e., learning curves) by progression"
    )
    return map_data_multiple_metrics_dropdown_plotly(
        metric_names=[m.name for m in map_metrics],
        xs_by_metric=xs_by_metric,
        ys_by_metric=ys_by_metric,
        legend_labels_by_metric=legend_labels_by_metric,
        stopping_markers_by_metric=stopping_markers_by_metric,
        title=title,
        xlabels_by_metric={
            m.name: "wall time" if by_walltime else m.map_key_info.key
            for m in map_metrics
        },
        lower_is_better_by_metric={m.name: m.lower_is_better for m in map_metrics},
    )


def _merge_trials_dict_with_df(
    df: pd.DataFrame,
    trials_dict: dict[int, Any],
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
            logger.debug(
                f"Column {column_name} missing for some trials. "
                "Filling with None when missing."
            )
        df[column_name] = [trials_dict[trial_index] for trial_index in df.trial_index]


def _get_generation_method_str(trial: BaseTrial) -> str:
    trial_generation_property = trial._properties.get("generation_model_key")
    if trial_generation_property is not None:
        return trial_generation_property

    generation_methods = {
        none_throws(generator_run._model_key)
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


def _merge_results_if_no_duplicates(
    arms_df: pd.DataFrame,
    results: pd.DataFrame,
    key_components: list[str],
    metrics: list[Metric],
) -> DataFrame:
    """Formats ``data.df`` and merges it with ``arms_df`` if all of the following are
    True:
        - ``data.df`` is not empty
        - ``data.df`` contains columns corresponding to ``key_components``
        - after any formatting, ``data.df`` contains no duplicates of the column
            ``results_key_col``
    """
    if len(results.index) == 0:
        logger.debug(
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


def _get_relative_results(
    results_df: pd.DataFrame, status_quo_arm_name: str
) -> pd.DataFrame:
    """Returns a dataframe with relative results, i.e. % change in metric values
    relative to the status quo arm.
    """
    baseline_df = results_df[results_df["arm_name"] == status_quo_arm_name]
    relative_results_df = pd.merge(
        results_df,
        baseline_df[["metric_name", "mean"]],
        on="metric_name",
        suffixes=("", "_baseline"),
    )
    relative_results_df["mean"] = (
        1.0 * relative_results_df["mean"] / relative_results_df["mean_baseline"] - 1.0
    ) * 100.0
    relative_results_df["metric_name"] = relative_results_df["metric_name"] + "_%CH"
    return relative_results_df


def exp_to_df(
    exp: Experiment,
    metrics: list[Metric] | None = None,
    run_metadata_fields: list[str] | None = None,
    trial_properties_fields: list[str] | None = None,
    trial_attribute_fields: list[str] | None = None,
    additional_fields_callables: None
    | (dict[str, Callable[[Experiment], dict[int, str | float]]]) = None,
    always_include_field_columns: bool = False,
    show_relative_metrics: bool = False,
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
        show_relative_metrics: If `True`, show % metric changes relative to the provided
            status quo arm. If no status quo arm is provided, raise a warning and show
            raw metric values. If `False`, show raw metric values (default).
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
        and len(none_throws(exp.optimization_config).all_constraints) > 0
    ):
        optimization_config = none_throws(exp.optimization_config)
        try:
            if any(oc.relative for oc in optimization_config.all_constraints):
                optimization_config = derelativize_opt_config(
                    optimization_config=optimization_config,
                    experiment=exp,
                )
            # Will return None for those rows whose feasibility cannot be determined.
            results[FEASIBLE_COL_NAME] = _is_row_feasible(
                df=results,
                optimization_config=optimization_config,
                undetermined_value=None,
            )
        except (KeyError, ValueError, DataRequiredError) as e:
            logger.warning(f"Feasibility calculation failed with error: {e}")

    # Calculate relative metrics if `show_relative_metrics` is True.
    if show_relative_metrics:
        if exp.status_quo is None:
            logger.warning(
                "No status quo arm found. Showing raw metric values instead of "
                "relative metric values."
            )
        else:
            status_quo_arm_name = exp.status_quo.name
            try:
                results = _get_relative_results(results, status_quo_arm_name)
            except Exception:
                logger.warning(
                    "Failed to calculate relative metrics. Showing raw metric values "
                    "instead of relative metric values."
                )

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
            else (
                f"{trial.abandoned_reason[:15]}..."
                if trial.status.is_abandoned and trial.abandoned_reason is not None
                else None
            )
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

    exp_df = none_throws(none_throws(exp_df).sort_values(["trial_index"]))
    initial_column_order = (
        ["trial_index", "arm_name", "trial_status", "reason", "generation_method"]
        + (run_metadata_fields or [])
        + (trial_properties_fields or [])
        + ([FEASIBLE_COL_NAME] if FEASIBLE_COL_NAME in exp_df.columns else [])
    )
    for column_name in reversed(initial_column_order):
        if column_name in exp_df.columns:
            exp_df.insert(0, column_name, exp_df.pop(column_name))
    return exp_df.reset_index(drop=True)


def compute_maximum_map_values(
    experiment: Experiment, map_key: str | None = None
) -> dict[int, float]:
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


def _pairwise_pareto_plotly_scatter(
    experiment: Experiment,
    metric_names: tuple[str, str] | None = None,
    reference_point: tuple[float, float] | None = None,
    minimize: bool | tuple[bool, bool] | None = None,
) -> Iterable[go.Figure]:
    metric_name_pairs = _get_metric_name_pairs(experiment=experiment)
    return [
        _pareto_frontier_scatter_2d_plotly(
            experiment=experiment,
            metric_names=metric_name_pair,
        )
        for metric_name_pair in metric_name_pairs
    ]


def _get_metric_name_pairs(
    experiment: Experiment, use_first_n_metrics: int = 4
) -> Iterable[tuple[str, str]]:
    optimization_config = _validate_experiment_and_get_optimization_config(
        experiment=experiment
    )
    if none_throws(optimization_config).is_moo_problem:
        multi_objective = assert_is_instance(
            none_throws(optimization_config).objective, MultiObjective
        )
        metric_names = [obj.metric.name for obj in multi_objective.objectives]
        if len(metric_names) > use_first_n_metrics:
            logger.info(
                f"Got `metric_names = {metric_names}` of length {len(metric_names)}. "
                f"Creating pairwise Pareto plots for the first `use_n_metrics = "
                f"{use_first_n_metrics}` of these and disregarding the remainder."
            )
            metric_names = metric_names[:use_first_n_metrics]
        metric_name_pairs = itertools.combinations(metric_names, 2)
        return metric_name_pairs
    raise UserInputError(
        "Inference of `metric_names` failed. Expected `MultiObjective` but "
        f"got {none_throws(optimization_config).objective}. Please provide an "
        "experiment with a MultiObjective `optimization_config`."
    )


def _pareto_frontier_scatter_2d_plotly(
    experiment: Experiment,
    metric_names: tuple[str, str] | None = None,
    reference_point: tuple[float, float] | None = None,
    minimize: bool | tuple[bool, bool] | None = None,
) -> go.Figure:
    # Determine defaults for unspecified inputs using `optimization_config`
    metric_names, reference_point, minimize = _pareto_frontier_plot_input_processing(
        experiment=experiment,
        metric_names=metric_names,
        reference_point=reference_point,
        minimize=minimize,
    )

    return pareto_frontier_scatter_2d_plotly(
        experiment, metric_names, reference_point, minimize
    )


def pareto_frontier_scatter_2d_plotly(
    experiment: Experiment,
    metric_names: tuple[str, str],
    reference_point: tuple[float, float] | None = None,
    minimize: bool | tuple[bool, bool] | None = None,
) -> go.Figure:
    df = exp_to_df(experiment)
    Y = df[list(metric_names)].to_numpy()
    Y_pareto = (
        _extract_observed_pareto_2d(
            Y=Y, reference_point=reference_point, minimize=minimize
        )
        if minimize is not None
        else None
    )

    hovertext = [f"Arm name: {arm_name}" for arm_name in df["arm_name"]]

    return scatter_plot_with_pareto_frontier_plotly(
        Y=Y,
        Y_pareto=Y_pareto,
        metric_x=metric_names[0],
        metric_y=metric_names[1],
        reference_point=reference_point,
        minimize=minimize,
        hovertext=hovertext,
    )


def _objective_vs_true_objective_scatter(
    model: Adapter,
    objective_metric_name: str,
    true_objective_metric_name: str,
) -> go.Figure:
    plot = plot_multiple_metrics(
        model=model,
        metric_x=objective_metric_name,
        metric_y=true_objective_metric_name,
        rel_x=False,
        rel_y=False,
    )

    fig = go.Figure(plot.data)
    fig.layout.title.text = (
        f"Objective {objective_metric_name} vs. True Objective "
        f"Metric {true_objective_metric_name}"
    )
    return fig


# TODO: may want to have a way to do this with a plot_fn
# that returns a list of plots, such as get_standard_plots
def get_figure_and_callback(
    plot_fn: Callable[[Orchestrator], go.Figure],
) -> tuple[go.Figure, Callable[[Orchestrator], None]]:
    """
    Produce a figure and a callback for updating the figure in place.

    A likely use case is that `plot_fn` takes a Orchestrator instance and
    returns a plotly Figure. Then `get_figure_and_callback` will produce a
    figure and callback that updates that figure according to `plot_fn`
    when the callback is passed to `orchestrator.run_n_trials` or
    `orchestrator.run_all_trials`.

    Args:
        plot_fn: A function for producing a Plotly figure from a orchestrator.
            If `plot_fn` raises a `RuntimeError`, the update will be skipped
            and optimization will proceed.

    Example:
        >>> def _plot(orchestrator:Orchestrator):
        >>>     standard_plots = get_standard_plots(orchestrator.experiment)
        >>>     return standard_plots[0]
        >>>
        >>> fig, callback = get_figure_and_callback(_plot)
    """
    fig = go.FigureWidget(layout=go.Layout())

    # pyre-fixme[53]: Captured variable `fig` is not annotated.
    def _update_fig_in_place(orchestrator: Orchestrator) -> None:
        try:
            new_fig = plot_fn(orchestrator)
        except RuntimeError as e:
            logging.warning(
                f"Plotting function called via callback failed with error {e}."
                "Skipping plot update."
            )
            return
        fig.update(
            data=new_fig._data,
            layout=new_fig._layout,
            overwrite=True,
        )

    # pyre-fixme[7]: Expected `Tuple[Figure, typing.Callable[[Orchestrator], None]]`
    #  but got `Tuple[FigureWidget,
    #  typing.Callable(get_figure_and_callback._update_fig_in_place)[[Named(orchestrator,
    #  Orchestrator)], None]]`.
    return fig, _update_fig_in_place


def _warn_and_create_warning_plot(warning_msg: str) -> go.Figure:
    logger.warning(warning_msg)
    return (
        go.Figure()
        .add_annotation(text=warning_msg, showarrow=False, font={"size": 20})
        .update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        .update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    )


def _format_comparison_string(
    comparison_arm_name: str,
    baseline_arm_name: str,
    objective_name: str,
    percent_change: float,
    baseline_value: float,
    comparison_value: float,
    digits: int,
) -> str:
    return (
        "**Metric "
        f"`{objective_name}` improved {percent_change:.{digits}f}%** "
        f"from `{baseline_value:.{digits}f}` in arm `'{baseline_arm_name}'` "
        f"to `{comparison_value:.{digits}f}` in arm `'{comparison_arm_name}'`.\n "
    )


def _construct_comparison_message(
    objective_name: str,
    objective_minimize: bool,
    baseline_arm_name: str,
    baseline_value: float,
    comparison_arm_name: str,
    comparison_value: float,
    digits: int | None = None,
) -> str | None:
    if baseline_value == 0:
        logger.debug(
            "compare_to_baseline: baseline has value of 0"
            + ", can't compute percent change."
        )
        return None

    if (objective_minimize and (baseline_value <= comparison_value)) or (
        not objective_minimize and (baseline_value >= comparison_value)
    ):
        logger.debug(
            f"compare_to_baseline: comparison arm {comparison_arm_name}"
            + f" did not beat baseline arm {baseline_arm_name}. "
        )
        return None
    percent_change = ((abs(comparison_value - baseline_value)) / baseline_value) * 100
    digits = _find_sigfigs(baseline_value, comparison_value)

    return _format_comparison_string(
        comparison_arm_name=comparison_arm_name,
        baseline_arm_name=baseline_arm_name,
        objective_name=objective_name,
        percent_change=percent_change,
        baseline_value=baseline_value,
        comparison_value=comparison_value,
        digits=digits,
    )


def _find_sigfigs(
    baseline_value: float,
    comparison_value: float,
    max_precision: int = 10,
    default_digits: int = 2,
) -> int:
    """Find the number of significant figures to display in a comparison message.
    This is done by finding the number of significant figures in the difference
    between baseline_value and comparison_value, so that the displayed values
    have at least one differing digit after rounding (if the abs difference is > 1).
    This compares up to max_precision digits after the decimal point,
    and defaults to default_digits if no differing digits are found.

    e.g.
    0.4 and 0.5 => 2 (round to 0.40 and 0.50)
    0.04390 and 0.03947 => 3 (round to 0.043 and 0.039)
    0.111122 and 0.111100 -> 0.111122 and 0.111100
    50.0 and 50.0001 -> 50.00 and 50.0001
    49.1 and 50.001 => 2 (round to 49.10 and 50.00)
    """
    diff = abs(baseline_value - comparison_value)
    str_diff = f"{diff:.{max_precision}f}"
    ints, decimals = str_diff.split(".")
    if int(ints) > 0:
        return default_digits
    for i, d in enumerate(decimals):
        if d != "0":
            return max(i + 1, default_digits)

    return default_digits


def _build_result_tuple(
    objective_name: str,
    objective_minimize: bool,
    baseline_arm_name: str,
    baseline_value: float,
    comparison_row: pd.DataFrame,
) -> tuple[str, bool, str, float, str, float]:
    """Formats inputs into a tuple for use in creating
    the comparison message.

    Returns:
        (metric_name,
        minimize,
        baseline_arm_name,
        baseline_value,
        comparison_arm_name,
        comparison_arm_value,)
    """
    comparison_arm_name = assert_is_instance(comparison_row["arm_name"], str)
    comparison_value = assert_is_instance(comparison_row[objective_name], float)

    result = (
        objective_name,
        objective_minimize,
        baseline_arm_name,
        baseline_value,
        comparison_arm_name,
        comparison_value,
    )
    return result


def maybe_extract_baseline_comparison_values(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None,
    comparison_arm_names: list[str] | None,
    baseline_arm_name: str | None,
) -> list[tuple[str, bool, str, float, str, float]] | None:
    """
    Extracts the baseline values from the experiment, for use in
    comparing the baseline arm to the optimal results.
    Requires the user specifies the names of the arms to compare to.

    Returns:
        List of tuples containing:
        (metric_name,
        minimize,
        baseline_arm_name,
        baseline_value,
        comparison_arm_name,
        comparison_arm_value,
        )
    """
    # TODO: incorporate model uncertainty when available
    # TODO: extract and use best arms if comparison_arm_names is not provided.
    #   Can do this automatically using optimization_config.
    if not comparison_arm_names:
        logger.debug(
            "compare_to_baseline: comparison_arm_names not provided. Returning None."
        )
        return None
    if not optimization_config:
        if experiment.optimization_config is None:
            logger.debug(
                "compare_to_baseline: optimization_config neither"
                + " provided in inputs nor present on experiment."
            )
            return None
        optimization_config = experiment.optimization_config

    arms_df = exp_to_df(experiment)
    if arms_df is None:
        logger.debug("compare_to_baseline: arms_df is None.")
        return None

    comparison_arm_df = arms_df[arms_df["arm_name"].isin(comparison_arm_names)]

    if comparison_arm_df is None or len(comparison_arm_df) == 0:
        logger.debug("compare_to_baseline: comparison_arm_df has no rows.")
        return None

    try:
        baseline_arm_name, _ = select_baseline_name_default_first_trial(
            experiment=experiment, baseline_arm_name=baseline_arm_name
        )
    except Exception as e:
        logger.debug(f"compare_to_baseline: could not select baseline arm. Reason: {e}")
        return None

    baseline_rows = arms_df[arms_df["arm_name"] == baseline_arm_name]

    if experiment.is_moo_problem:
        multi_objective = assert_is_instance(
            optimization_config.objective, MultiObjective
        )
        result_list = []
        for objective in multi_objective.objectives:
            name = objective.metric.name
            minimize = objective.minimize
            opt_index = (
                comparison_arm_df[name].idxmin()
                if minimize
                else comparison_arm_df[name].idxmax()
            )
            comparison_row = arms_df.iloc[opt_index]
            baseline_value = baseline_rows.iloc[0][name]

            result_tuple = _build_result_tuple(
                objective_name=name,
                objective_minimize=minimize,
                baseline_arm_name=baseline_arm_name,
                baseline_value=baseline_value,
                comparison_row=comparison_row,
            )

            result_list.append(result_tuple)
        return result_list if result_list else None

    objective_name = optimization_config.objective.metric.name
    baseline_value = baseline_rows.iloc[0][objective_name]
    comparison_row = comparison_arm_df.iloc[0]

    return [
        _build_result_tuple(
            objective_name=objective_name,
            objective_minimize=optimization_config.objective.minimize,
            baseline_arm_name=baseline_arm_name,
            baseline_value=baseline_value,
            comparison_row=comparison_row,
        )
    ]


def compare_to_baseline_impl(
    comparison_list: list[tuple[str, bool, str, float, str, float]],
) -> str | None:
    """Implementation of compare_to_baseline, taking in a
    list of arm comparisons.
    Can be used directly with the output of
    'maybe_extract_baseline_comparison_values'"""
    result_message = ""
    if len(comparison_list) > 1:
        result_message = (
            "Below is the greatest improvement, if any,"
            " achieved for each objective metric \n"
        )

    for _, result_tuple in enumerate(comparison_list):
        comparison_message = _construct_comparison_message(*result_tuple)
        if comparison_message:
            result_message = (
                result_message
                + (" \n* " if len(comparison_list) > 1 else "")
                + none_throws(comparison_message)
            )

    return result_message if result_message else None


def compare_to_baseline(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None,
    comparison_arm_names: list[str] | None,
    baseline_arm_name: str | None = None,
) -> str | None:
    """Calculate metric improvement of the experiment against baseline.
    Returns the message(s) added to markdown_messages."""

    comparison_list = maybe_extract_baseline_comparison_values(
        experiment=experiment,
        optimization_config=optimization_config,
        comparison_arm_names=comparison_arm_names,
        baseline_arm_name=baseline_arm_name,
    )
    if not comparison_list:
        return None
    comparison_list = none_throws(comparison_list)
    return compare_to_baseline_impl(comparison_list)


def warn_if_unpredictable_metrics(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    model_fit_threshold: float,
    metric_names: list[str] | None = None,
    model_fit_metric_name: str = "coefficient_of_determination",
) -> str | None:
    """Warn if any optimization config metrics are considered unpredictable,
    i.e., their coefficient of determination is less than model_fit_threshold.
    Args:
        experiment: The experiment containing the data and optimization_config.
            If there is no optimization config, this function checks all metrics
            attached to the experiment.
        generation_strategy: The generation strategy containing the model.
        model_fit_threshold: If a model's coefficient of determination is below
            this threshold, that metric is considered unpredictable.
        metric_names: If specified, only check these metrics.
        model_fit_metric_name: Name of the metric to apply the model fit threshold to.

    Returns:
        A string warning the user about unpredictable metrics, if applicable.
    """
    # Get fit quality dict.
    adapter = generation_strategy.adapter  # Optional[Adapter]
    if adapter is None:  # Need to re-fit the model.
        generation_strategy._curr._fit(experiment=experiment)
        adapter = cast(Adapter, generation_strategy.adapter)
    if isinstance(adapter, RandomAdapter):
        logger.debug(
            "Current adapter on GenerationStrategy is RandomAdapter. "
            "Not checking metric predictability."
        )
        return None
    model_fit_dict = compute_model_fit_metrics_from_adapter(
        adapter=adapter,
        generalization=True,  # use generalization metrics for user warning
        untransform=False,
    )
    fit_quality_dict = model_fit_dict[model_fit_metric_name]

    # Extract salient metrics from experiment.
    if metric_names is None:
        if experiment.optimization_config is None:
            metric_names = list(experiment.metrics.keys())
        else:
            metric_names = list(
                none_throws(experiment.optimization_config).metrics.keys()
            )
    else:
        # Raise a ValueError if any metric names are invalid.
        bad_metric_names = set(metric_names) - set(experiment.metrics.keys())
        if len(bad_metric_names) > 0:
            raise ValueError(
                f"Invalid metric names: {bad_metric_names}. Please only use "
                "metric_names that are available on the present experiment, "
                f"which are: {list(experiment.metrics.keys())}."
            )

    # Flag metrics whose coefficient of determination is below the threshold.
    unpredictable_metrics = {
        k: v
        for k, v in fit_quality_dict.items()
        if k in metric_names and v < model_fit_threshold
    }

    if len(unpredictable_metrics) > 0:
        return UNPREDICTABLE_METRICS_MESSAGE.format(list(unpredictable_metrics.keys()))
