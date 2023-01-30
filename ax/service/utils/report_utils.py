#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from collections import defaultdict
from logging import Logger
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import gpytorch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRunType
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.trial import BaseTrial
from ax.exceptions.core import UserInputError
from ax.modelbridge import ModelBridge
from ax.modelbridge.cross_validation import cross_validate
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
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.service.utils.best_point import _derel_opt_config_wrapper, _is_row_feasible
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from pandas.core.frame import DataFrame

if TYPE_CHECKING:
    from ax.service.scheduler import Scheduler


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


# pyre-ignore[11]: Annotation `go.Figure` is not defined as a type.
def _get_cross_validation_plots(model: ModelBridge) -> List[go.Figure]:
    cv = cross_validate(model=model)
    return [
        interact_cross_validation_plotly(
            cv_results=cv, caption=CROSS_VALIDATION_CAPTION
        )
    ]


def _get_objective_trace_plot(
    experiment: Experiment,
    data: Data,
    model_transitions: List[int],
    true_objective_metric_name: Optional[str] = None,
) -> Iterable[go.Figure]:
    if experiment.is_moo_problem:
        return [
            scatter_plot_with_hypervolume_trace_plotly(experiment=experiment),
            *_pairwise_pareto_plotly_scatter(experiment=experiment),
        ]

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
        optimization_trace_single_method_plotly(
            y=np.array([data.df[data.df["metric_name"] == metric_name]["mean"]]),
            title=f"Best {metric_name} found vs. # of iterations",
            ylabel=metric_name,
            model_transitions=model_transitions,
            # Try and use the metric's lower_is_better property, but fall back on
            # objective's minimize property if relevent
            optimization_direction=(
                (
                    "minimize"
                    if experiment.metrics[metric_name].lower_is_better is True
                    else "maximize"
                )
                if experiment.metrics[metric_name].lower_is_better is not None
                else (
                    "minimize" if optimization_config.objective.minimize else "maximize"
                )
            ),
            plot_trial_points=True,
        )
        for metric_name in metric_names
    ]

    return [plot for plot in plots if plot is not None]


def _get_objective_v_param_plots(
    experiment: Experiment,
    model: ModelBridge,
    max_num_slice_plots: int = 100,
    max_num_contour_plots: int = 100,
) -> List[go.Figure]:
    search_space = experiment.search_space

    range_params = get_range_parameters_from_list(
        list(search_space.range_parameters.values()), min_num_values=5
    )
    if len(range_params) < 1:
        # if search space contains no range params
        logger.warning(
            "`_get_objective_v_param_plot` requires a search space with at least one "
            "`RangeParameter`. Returning an empty list."
        )
        return []
    num_range_params = len(range_params)
    num_metrics = len(experiment.metrics)
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

    num_contour_plots = num_range_params * (num_range_params - 1) * num_metrics
    if num_range_params > 1 and num_contour_plots <= max_num_contour_plots:
        # contour plots
        try:
            with gpytorch.settings.max_eager_kernel_size(float("inf")):
                output_plots += [
                    interact_contour_plotly(
                        model=not_none(model),
                        metric_name=metric_name,
                    )
                    for metric_name in model.metric_names
                ]
        # `mean shape torch.Size` RunTimeErrors, pending resolution of
        # https://github.com/cornellius-gp/gpytorch/issues/1853
        except RuntimeError as e:
            logger.warning(f"Contour plotting failed with error: {e}.")
    elif num_contour_plots > max_num_contour_plots:
        warning_msg = (
            f"Skipping creation of {num_contour_plots} contour plots since that "
            f"exceeds <br>`max_num_contour_plots = {max_num_contour_plots}`."
            "<br>Users can plot individual contour plots with the <br>python "
            "function ax.plot.contour.plot_contour_plotly."
        )
        # TODO: return a warning here then convert to a plot/message/etc. downstream.
        warning_plot = _warn_and_create_warning_plot(warning_msg=warning_msg)
        output_plots.append(warning_plot)
    return output_plots


def _get_suffix(input_str: str, delim: str = ".", n_chunks: int = 1) -> str:
    return delim.join(input_str.split(delim)[-n_chunks:])


def _get_shortest_unique_suffix_dict(
    input_str_list: List[str], delim: str = "."
) -> Dict[str, str]:
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
    model: Optional[ModelBridge],
    data: Optional[Data] = None,
    model_transitions: Optional[List[int]] = None,
    true_objective_metric_name: Optional[str] = None,
) -> List[go.Figure]:
    """Extract standard plots for single-objective optimization.

    Extracts a list of plots from an ``Experiment`` and ``ModelBridge`` of general
    interest to an Ax user. Currently not supported are
    - TODO: multi-objective optimization
    - TODO: ChoiceParameter plots

    Args:
        - experiment: The ``Experiment`` from which to obtain standard plots.
        - model: The ``ModelBridge`` used to suggest trial parameters.
        - data: If specified, data, to which to fit the model before generating plots.
        - model_transitions: The arm numbers at which shifts in generation_strategy
            occur.

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

    objective = not_none(experiment.optimization_config).objective
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
                model_transitions=model_transitions
                if model_transitions is not None
                else [],
                true_objective_metric_name=true_objective_metric_name,
            )
        )
    except Exception as e:
        # Allow model-based plotting to proceed if objective_trace plotting fails.
        logger.exception(f"Plotting `objective_trace` failed with error {e}")
        pass

    # Objective vs. parameter plot requires a `Model`, so add it only if model
    # is alrady available. In cases where initially custom trials are attached,
    # model might not yet be set on the generation strategy.
    if model:
        # TODO: Check if model can predict in favor of try/catch.
        try:
            if true_objective_metric_name is not None:
                output_plot_list.append(
                    _objective_vs_true_objective_scatter(
                        model=model,
                        objective_metric_name=objective.metric_names[0],
                        true_objective_metric_name=true_objective_metric_name,
                    )
                )
            output_plot_list.extend(
                _get_objective_v_param_plots(
                    experiment=experiment,
                    model=model,
                )
            )
            output_plot_list.extend(_get_cross_validation_plots(model=model))
            feature_importance_plot = plot_feature_importance_by_feature_plotly(
                model=model, relative=False, caption=FEATURE_IMPORTANCE_CAPTION
            )
            feature_importance_plot.layout.title = "[ADVANCED] " + str(
                # pyre-fixme[16]: go.Figure has no attribute `layout`
                feature_importance_plot.layout.title.text
            )
            output_plot_list.append(feature_importance_plot)
            output_plot_list.append(interact_fitted_plotly(model=model, rel=False))
        except NotImplementedError:
            # Model does not implement `predict` method.
            pass

    return [plot for plot in output_plot_list if plot is not None]


def _merge_trials_dict_with_df(
    df: pd.DataFrame,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    trials_dict: Dict[int, Any],
    column_name: str,
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
    """

    if "trial_index" not in df.columns:
        raise ValueError("df must have trial_index column")
    if any(trials_dict.values()):  # field present for any trial
        if not all(trials_dict.values()):  # not present for all trials
            logger.warning(
                f"Column {column_name} missing for some trials. "
                "Filling with None when missing."
            )
        df[column_name] = [trials_dict[trial_index] for trial_index in df.trial_index]
    else:
        logger.warning(
            f"Column {column_name} missing for all trials. " "Not appending column."
        )


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
    # prepare results for merge
    key_vals = results[key_components[0]].astype("str")
    for key_component in key_components[1:]:
        key_vals += results[key_component].astype("str")
    results_key_col = "-".join(key_components)

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
        run_metadata_fields: fields to extract from ``trial.run_metadata`` for trial
            in ``experiment.trials``. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial.
        trial_properties_fields: fields to extract from ``trial._properties`` for trial
            in ``experiment.trials``. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial. Output columns names
            will be prepended with ``"trial_properties_"``.

    Returns:
        DataFrame: A dataframe of inputs, metadata and metrics by trial and arm (and
        ``map_keys``, if present). If no trials are available, returns an empty
        dataframe. If no metric ouputs are available, returns a dataframe of inputs and
        metadata.
    """

    if len(kwargs) > 0:
        logger.warn(
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
        except ValueError as e:
            logger.warning(e)

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
            )
    exp_df = _merge_results_if_no_duplicates(
        arms_df=arms_df,
        results=results,
        key_components=key_components,
        metrics=metrics or list(exp.metrics.values()),
    )

    exp_df = not_none(not_none(exp_df).sort_values(["trial_index"]))
    initial_column_order = (
        ["trial_index", "arm_name", "trial_status", "generation_method"]
        + (run_metadata_fields or [])
        + (trial_properties_fields or [])
    )
    for column_name in reversed(initial_column_order):
        if column_name in exp_df.columns:
            # pyre-ignore[6]: In call `DataFrame.insert`, for 3rd positional argument,
            # expected `Union[int, Series, Variable[ArrayLike <: [ExtensionArray,
            # ndarray]]]` but got `Union[DataFrame, Series]`]
            exp_df.insert(0, column_name, exp_df.pop(column_name))
    return exp_df.reset_index(drop=True)


def _pairwise_pareto_plotly_scatter(
    experiment: Experiment,
    metric_names: Optional[Tuple[str, str]] = None,
    reference_point: Optional[Tuple[float, float]] = None,
    minimize: Optional[Union[bool, Tuple[bool, bool]]] = None,
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
) -> Iterable[Tuple[str, str]]:
    optimization_config = _validate_experiment_and_get_optimization_config(
        experiment=experiment
    )
    if not_none(optimization_config).is_moo_problem:
        multi_objective = checked_cast(
            MultiObjective, not_none(optimization_config).objective
        )
        metric_names = [obj.metric.name for obj in multi_objective.objectives]
        if len(metric_names) > use_first_n_metrics:
            logger.warning(
                f"Got `metric_names = {metric_names}` of length {len(metric_names)}. "
                f"Creating pairwise Pareto plots for the first `use_n_metrics = "
                f"{use_first_n_metrics}` of these and disregarding the remainder."
            )
            metric_names = metric_names[:use_first_n_metrics]
        metric_name_pairs = itertools.combinations(metric_names, 2)
        return metric_name_pairs
    raise UserInputError(
        "Inference of `metric_names` failed. Expected `MultiObjective` but "
        f"got {not_none(optimization_config).objective}. Please provide an experiment "
        "with a MultiObjective `optimization_config`."
    )


def _pareto_frontier_scatter_2d_plotly(
    experiment: Experiment,
    metric_names: Optional[Tuple[str, str]] = None,
    reference_point: Optional[Tuple[float, float]] = None,
    minimize: Optional[Union[bool, Tuple[bool, bool]]] = None,
) -> go.Figure:

    # Determine defaults for unspecified inputs using `optimization_config`
    metric_names, reference_point, minimize = _pareto_frontier_plot_input_processing(
        experiment=experiment,
        metric_names=metric_names,
        reference_point=reference_point,
        minimize=minimize,
    )

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
    model: ModelBridge,
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
    plot_fn: Callable[["Scheduler"], go.Figure]
) -> Tuple[go.Figure, Callable[["Scheduler"], None]]:
    """
    Produce a figure and a callback for updating the figure in place.

    A likely use case is that `plot_fn` takes a Scheduler instance and
    returns a plotly Figure. Then `get_figure_and_callback` will produce a
    figure and callback that updates that figure according to `plot_fn`
    when the callback is passed to `Scheduler.run_n_trials` or
    `Scheduler.run_all_trials`.

    Args:
        plot_fn: A function for producing a Plotly figure from a scheduler.
            If `plot_fn` raises a `RuntimeError`, the update wil be skipped
            and optimization will proceed.

    Example:
        >>> def _plot(scheduler: Scheduler):
        >>>     standard_plots = get_standard_plots(scheduler.experiment)
        >>>     return standard_plots[0]
        >>>
        >>> fig, callback = get_figure_and_callback(_plot)
    """
    fig = go.FigureWidget(layout=go.Layout())

    # pyre-fixme[53]: Captured variable `fig` is not annotated.
    def _update_fig_in_place(scheduler: "Scheduler") -> None:
        try:
            new_fig = plot_fn(scheduler)
        except RuntimeError as e:
            logging.warn(
                f"Plotting function called via callback failed with error {e}."
                "Skipping plot update."
            )
            return
        fig.update(data=new_fig._data, layout=new_fig._layout, overwrite=True)

    return fig, _update_fig_in_place


def _warn_and_create_warning_plot(warning_msg: str) -> go.Figure:
    logger.warning(warning_msg)
    return (
        go.Figure()
        .add_annotation(text=warning_msg, showarrow=False, font={"size": 20})
        .update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        .update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    )
