#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objs as go
from ae.lazarus.ae.core.batch_trial import BatchTrial
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.observation import Observation
from ae.lazarus.ae.generator.cross_validation import CVResult
from ae.lazarus.ae.plot.base import (
    AEPlotConfig,
    AEPlotTypes,
    PlotData,
    PlotInSampleCondition,
    PlotMetric,
    Z,
)
from ae.lazarus.ae.plot.scatter import _error_scatter_data, _error_scatter_trace
from plotly import tools


# type alias
FloatList = List[float]


# Helper functions for plotting model fits
def _get_min_max_with_errors(
    x: FloatList, y: FloatList, sd_x: FloatList, sd_y: FloatList
) -> Tuple[float, float]:
    """Get min and max of a bivariate dataset (across variables).

    Args:
        x (List[float]): point estimate of x variable.
        y (List[float]): point estimate of y variable.
        sd_x (List[float]): standard deviation of x variable.
        sd_y (List[float]): standard deviation of y variable.

    Returns:
        min_ (float): minimum of points, including uncertainty.
        max_ (float): maximum of points, including uncertainty.

    """
    min_ = min(
        min(np.array(x) - np.multiply(sd_x, Z)), min(np.array(y) - np.multiply(sd_y, Z))
    )
    max_ = max(
        max(np.array(x) + np.multiply(sd_x, Z)), max(np.array(y) + np.multiply(sd_y, Z))
    )
    return min_, max_


def _diagonal_trace(min_: float, max_: float, visible: bool = True) -> Dict[str, Any]:
    """Diagonal line trace from (min_, min_) to (max_, max_).

    Args:
        min_ (float): minimum to be used for starting point of line.
        max_ (float): maximum to be used for ending point of line.
        visible (bool): if True, trace is set to visible.

    """
    return go.Scatter(  # pyre-ignore[16]
        x=[min_, max_],
        y=[min_, max_],
        line=go.Line(color="black", width=2, dash="dot"),  # pyre-ignore[16]
        mode="lines",
        hoverinfo="none",
        visible=visible,
        showlegend=False,
    )


def _obs_vs_pred_dropdown_plot(
    data: PlotData,
    rel: bool,
    show_context: bool = False,
    xlabel: str = "Actual Outcome",
    ylabel: str = "Predicted Outcome",
) -> Dict[str, Any]:
    """Plot a dropdown plot of observed vs. predicted values from a model.

    Args:
        data (PlotData): a name tuple storing observed and predicted data
            from a model.
        rel (bool): if True, plot metrics relative to the status quo.
        show_context (bool, optional): Show context on hover.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.

    """
    traces = []
    metric_dropdown = []

    if rel and data.status_quo_name is not None:
        if show_context:
            raise ValueError(
                "This plot does not support both context and relativization at "
                "the same time."
            )
        status_quo_condition = data.in_sample[data.status_quo_name]
    else:
        status_quo_condition = None

    for i, metric in enumerate(data.metrics):
        y_raw, se_raw, y_hat, se_hat = _error_scatter_data(
            # Expected `List[typing.Union[PlotInSampleCondition,
            # ae.lazarus.ae.plot.base.PlotOutOfSampleCondition]]` for 1st anonymous
            # parameter to call `ae.lazarus.ae.plot.scatter._error_scatter_data` but got
            # `List[PlotInSampleCondition]`.
            # pyre-fixme[6]:
            list(data.in_sample.values()),
            y_axis_var=PlotMetric(metric, True),
            x_axis_var=PlotMetric(metric, False),
            rel=rel,
            status_quo_condition=status_quo_condition,
        )
        min_, max_ = _get_min_max_with_errors(y_raw, y_hat, se_raw or [], se_hat)
        traces.append(_diagonal_trace(min_, max_, visible=(i == 0)))
        traces.append(
            _error_scatter_trace(
                # Expected `List[typing.Union[PlotInSampleCondition,
                # ae.lazarus.ae.plot.base.PlotOutOfSampleCondition]]` for 1st parameter
                # `conditions` to call `ae.lazarus.ae.plot.scatter._error_scatter_trace`
                # but got `List[PlotInSampleCondition]`.
                # pyre-fixme[6]:
                conditions=list(data.in_sample.values()),
                hoverinfo="text",
                rel=rel,
                show_condition_details_on_hover=True,
                show_CI=True,
                show_context=show_context,
                status_quo_condition=status_quo_condition,
                visible=(i == 0),
                x_axis_label=xlabel,
                x_axis_var=PlotMetric(metric, False),
                y_axis_label=ylabel,
                y_axis_var=PlotMetric(metric, True),
            )
        )

        # only the first two traces are visible (corresponding to first outcome
        # in dropdown)
        is_visible = [False] * (len(data.metrics) * 2)
        is_visible[2 * i] = True
        is_visible[2 * i + 1] = True

        # on dropdown change, restyle
        metric_dropdown.append(
            {"args": ["visible", is_visible], "label": metric, "method": "restyle"}
        )

    updatemenus = [
        {
            "x": 0,
            "y": 1.125,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": metric_dropdown,
        },
        {
            "buttons": [
                {
                    "args": [
                        {
                            "error_x.width": 4,
                            "error_x.thickness": 2,
                            "error_y.width": 4,
                            "error_y.thickness": 2,
                        }
                    ],
                    "label": "Yes",
                    "method": "restyle",
                },
                {
                    "args": [
                        {
                            "error_x.width": 0,
                            "error_x.thickness": 0,
                            "error_y.width": 0,
                            "error_y.thickness": 0,
                        }
                    ],
                    "label": "No",
                    "method": "restyle",
                },
            ],
            "x": 1.125,
            "xanchor": "left",
            "y": 0.8,
            "yanchor": "middle",
        },
    ]

    layout = go.Layout(  # pyre-ignore[16]
        annotations=[
            {
                "showarrow": False,
                "text": "Show CI",
                "x": 1.125,
                "xanchor": "left",
                "xref": "paper",
                "y": 0.9,
                "yanchor": "middle",
                "yref": "paper",
            }
        ],
        xaxis={
            "title": xlabel,
            "zeroline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 0.5,
        },
        yaxis={
            "title": ylabel,
            "zeroline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 0.5,
        },
        showlegend=False,
        hovermode="closest",
        updatemenus=updatemenus,
        width=530,
        height=500,
    )

    return go.Figure(data=traces, layout=layout)  # pyre-ignore[16]


def _get_batch_comparison_plot_data(
    observations: List[Observation],
    batch_x: int,
    batch_y: int,
    rel: bool = False,
    status_quo_name: Optional[str] = None,
) -> PlotData:
    """Compute PlotData for comparing repeated arms across trials.

    Args:
        observations (List[Observation]): List of observations.
        batch_x (int): Batch for x-axis.
        batch_y (int): Batch for y-axis.
        rel (bool): Whether to relativize data against status_quo condition.
        status_quo_name (bool): Name of the status_quo condition.

    Returns:
        PlotData: a plot data object.
    """
    if rel and status_quo_name is None:
        raise ValueError("Experiment status quo must be set for rel=True")
    x_observations = {
        observation.condition_name: observation
        for observation in observations
        if observation.features.trial_index == batch_x
    }
    y_observations = {
        observation.condition_name: observation
        for observation in observations
        if observation.features.trial_index == batch_y
    }

    # Assume input is well formed and metric_names are consistent across observations
    metric_names = observations[0].data.metric_names
    insample_data: Dict[str, PlotInSampleCondition] = {}
    for condition_name, x_observation in x_observations.items():
        # Restrict to conditions present in both trials
        if condition_name not in y_observations:
            continue

        y_observation = y_observations[condition_name]
        condition_data = {
            "name": condition_name,
            "y": {},
            "se": {},
            "params": x_observation.features.parameters,
            "y_hat": {},
            "se_hat": {},
            "context_stratum": None,
        }
        for i, mname in enumerate(x_observation.data.metric_names):
            condition_data["y"][mname] = x_observation.data.means[i]
            condition_data["se"][mname] = np.sqrt(x_observation.data.covariance[i][i])
        for i, mname in enumerate(y_observation.data.metric_names):
            condition_data["y_hat"][mname] = y_observation.data.means[i]
            condition_data["se_hat"][mname] = np.sqrt(
                y_observation.data.covariance[i][i]
            )
        # Expected `str` for 2nd anonymous parameter to call `dict.__setitem__` but got
        # `Optional[str]`.
        # pyre-fixme[6]:
        insample_data[condition_name] = PlotInSampleCondition(**condition_data)

    return PlotData(
        metrics=metric_names,
        in_sample=insample_data,
        out_of_sample=None,
        status_quo_name=status_quo_name,
    )


def _get_cv_plot_data(cv_results: List[CVResult]) -> PlotData:
    if len(cv_results) == 0:
        return PlotData(
            metrics=[], in_sample={}, out_of_sample=None, status_quo_name=None
        )

    # condition_name -> Condition data
    insample_data: Dict[str, PlotInSampleCondition] = {}

    # Assume input is well formed and this is consistent
    metric_names = cv_results[0].observed.data.metric_names

    for cv_result in cv_results:
        condition_name = cv_result.observed.condition_name
        condition_data = {
            "name": cv_result.observed.condition_name,
            "y": {},
            "se": {},
            "params": cv_result.observed.features.parameters,
            "y_hat": {},
            "se_hat": {},
            "context_stratum": None,
        }
        for i, mname in enumerate(cv_result.observed.data.metric_names):
            condition_data["y"][mname] = cv_result.observed.data.means[i]
            condition_data["se"][mname] = np.sqrt(
                cv_result.observed.data.covariance[i][i]
            )
        for i, mname in enumerate(cv_result.predicted.metric_names):
            condition_data["y_hat"][mname] = cv_result.predicted.means[i]
            condition_data["se_hat"][mname] = np.sqrt(
                cv_result.predicted.covariance[i][i]
            )

        # Expected `str` for 2nd anonymous parameter to call `dict.__setitem__` but got
        # `Optional[str]`.
        # pyre-fixme[6]:
        insample_data[condition_name] = PlotInSampleCondition(**condition_data)
    return PlotData(
        metrics=metric_names,
        in_sample=insample_data,
        out_of_sample=None,
        status_quo_name=None,
    )


def interact_empirical_model_validation(batch: BatchTrial, data: Data) -> AEPlotConfig:
    """Compare the model predictions for the batch conditions against observed data.

    Relies on the model predictions stored on the generator_runs of batch.

    Args:
        batch (Batch): Batch on which to perform analysis.
        data (Data): Observed data for the batch.
    Returns:
        AEPlotConfig for the plot.
    """
    insample_data: Dict[str, PlotInSampleCondition] = {}
    metric_names = list(data.df["metric_name"].unique())
    for struct in batch.generator_run_structs:
        generator_run = struct.generator_run
        if generator_run.model_predictions is None:
            continue
        for i, condition in enumerate(generator_run.conditions):
            condition_data = {
                "name": condition.name_or_short_signature,
                "y": {},
                "se": {},
                "params": condition.params,
                "y_hat": {},
                "se_hat": {},
                "context_stratum": None,
            }
            predictions = generator_run.model_predictions
            for _, row in data.df[
                data.df["condition_name"] == condition.name_or_short_signature
            ].iterrows():
                metric_name = row["metric_name"]
                condition_data["y"][metric_name] = row["mean"]
                condition_data["se"][metric_name] = row["sem"]
                condition_data["y_hat"][metric_name] = predictions[0][metric_name][i]
                condition_data["se_hat"][metric_name] = predictions[1][metric_name][
                    metric_name
                ][i]
            insample_data[condition.name_or_short_signature] = PlotInSampleCondition(
                **condition_data
            )
    plot_data = PlotData(
        metrics=metric_names,
        in_sample=insample_data,
        out_of_sample=None,
        status_quo_name=None,
    )

    fig = _obs_vs_pred_dropdown_plot(data=plot_data, rel=False)
    fig["layout"]["title"] = "Cross-validation"
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def interact_cross_validation(
    cv_results: List[CVResult], show_context: bool = True
) -> AEPlotConfig:
    """Interactive cross-validation (CV) plotting; select metric via dropdown.

    Note: uses the Plotly version of dropdown (which means that all data is
    stored within the notebook).

    Args:
        cv_results (List[CVResult]): cross-validation results.
        show_context (bool, optional): if True, show context on hover.

    """
    data = _get_cv_plot_data(cv_results)
    fig = _obs_vs_pred_dropdown_plot(data=data, rel=False, show_context=show_context)
    fig["layout"]["title"] = "Cross-validation"
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def tile_cross_validation(
    cv_results: List[CVResult],
    show_condition_details_on_hover: bool = True,
    show_context: bool = True,
) -> AEPlotConfig:
    """Tile version of CV plots; sorted by 'best fitting' outcomes.

    Plots are sorted in decreasing order using the p-value of a Fisher exact
    test statistic.

    Args:
        cv_results (List[CVResult]): cross-validation results.
        include_measurement_error (bool, optional): if True, include
            measurement_error metrics in plot.
        show_condition_details_on_hover (bool, optional): if True, display
            parameterizations of conditions on hover. Default is True.
        show_context (bool, optional): if True (default), display context on
            hover.

    """
    data = _get_cv_plot_data(cv_results)
    metrics = data.metrics

    # make subplots (2 plots per row)
    nrows = int(np.ceil(len(metrics) / 2))
    ncols = min(len(metrics), 2)
    fig = tools.make_subplots(
        rows=nrows,
        cols=ncols,
        print_grid=False,
        subplot_titles=tuple(metrics),
        horizontal_spacing=0.15,
        vertical_spacing=0.30 / nrows,
    )

    for i, metric in enumerate(metrics):
        y_hat = []
        se_hat = []
        y_raw = []
        se_raw = []
        for condition in data.in_sample.values():
            y_hat.append(condition.y_hat[metric])
            se_hat.append(condition.se_hat[metric])
            y_raw.append(condition.y[metric])
            se_raw.append(condition.se[metric])
        min_, max_ = _get_min_max_with_errors(y_raw, y_hat, se_raw, se_hat)
        fig.append_trace(
            _diagonal_trace(min_, max_), int(np.floor(i / 2)) + 1, i % 2 + 1
        )
        fig.append_trace(
            _error_scatter_trace(
                # Expected `List[typing.Union[PlotInSampleCondition,
                # ae.lazarus.ae.plot.base.PlotOutOfSampleCondition]]` for 1st anonymous
                # parameter to call `ae.lazarus.ae.plot.scatter._error_scatter_trace` but
                # got `List[PlotInSampleCondition]`.
                # pyre-fixme[6]:
                list(data.in_sample.values()),
                y_axis_var=PlotMetric(metric, True),
                x_axis_var=PlotMetric(metric, False),
                y_axis_label="Predicted",
                x_axis_label="Actual",
                hoverinfo="text",
                show_condition_details_on_hover=show_condition_details_on_hover,
                show_context=show_context,
            ),
            int(np.floor(i / 2)) + 1,
            i % 2 + 1,
        )

    # if odd number of plots, need to manually remove the last blank subplot
    # generated by `tools.make_subplots`
    if len(metrics) % 2 == 1:
        del fig["layout"]["xaxis{}".format(nrows * ncols)]
        del fig["layout"]["yaxis{}".format(nrows * ncols)]

    # allocate 400 px per plot (equal aspect ratio)
    fig["layout"].update(
        title="Cross-Validation",  # What should I replace this with?
        hovermode="closest",
        width=800,
        height=400 * nrows,
        font={"size": 10},
        showlegend=False,
    )

    # update subplot title size and the axis labels
    for i, ant in enumerate(fig["layout"]["annotations"]):
        ant["font"].update(size=12)
        fig["layout"]["xaxis{}".format(i + 1)].update(
            title="Actual Outcome", mirror=True, linecolor="black", linewidth=0.5
        )
        fig["layout"]["yaxis{}".format(i + 1)].update(
            title="Predicted Outcome", mirror=True, linecolor="black", linewidth=0.5
        )

    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)


def interact_batch_comparison(
    observations: List[Observation],
    batch_x: int,
    batch_y: int,
    rel: bool = False,
    status_quo_name: Optional[str] = None,
) -> AEPlotConfig:
    """Compare repeated conditions from two trials; select metric via dropdown.

    Args:
        observations: List of observations to compute comparison.
        batch_x: Index of batch for x-axis.
        batch_y: Index of bach for y-axis.
        rel: Whether to relativize data against status_quo condition.
        status_quo_name: Name of the status_quo condition.
    """
    plot_data = _get_batch_comparison_plot_data(
        observations, batch_x, batch_y, rel=rel, status_quo_name=status_quo_name
    )
    fig = _obs_vs_pred_dropdown_plot(
        data=plot_data,
        rel=rel,
        xlabel="Batch {}".format(batch_x),
        ylabel="Batch {}".format(batch_y),
    )
    fig["layout"]["title"] = "Repeated conditions across trials"
    return AEPlotConfig(data=fig, plot_type=AEPlotTypes.GENERIC)
