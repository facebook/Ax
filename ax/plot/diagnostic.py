#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objs as go
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.observation import Observation
from ax.modelbridge.cross_validation import CVResult
from ax.modelbridge.transforms.convert_metric_names import convert_mt_observations
from ax.plot.base import (
    AxPlotConfig,
    AxPlotTypes,
    PlotData,
    PlotInSampleArm,
    PlotMetric,
    Z,
)
from ax.plot.scatter import _error_scatter_data, _error_scatter_trace
from ax.utils.common.typeutils import not_none
from plotly import subplots


# type alias
FloatList = List[float]


# Helper functions for plotting model fits
def _get_min_max_with_errors(
    x: FloatList, y: FloatList, sd_x: FloatList, sd_y: FloatList
) -> Tuple[float, float]:
    """Get min and max of a bivariate dataset (across variables).

    Args:
        x: point estimate of x variable.
        y: point estimate of y variable.
        sd_x: standard deviation of x variable.
        sd_y: standard deviation of y variable.

    Returns:
        min_: minimum of points, including uncertainty.
        max_: maximum of points, including uncertainty.

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
        min_: minimum to be used for starting point of line.
        max_: maximum to be used for ending point of line.
        visible: if True, trace is set to visible.

    """
    return go.Scatter(
        x=[min_, max_],
        y=[min_, max_],
        line=dict(color="black", width=2, dash="dot"),  # noqa: C408
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
        data: a name tuple storing observed and predicted data
            from a model.
        rel: if True, plot metrics relative to the status quo.
        show_context: Show context on hover.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.

    """
    traces = []
    metric_dropdown = []

    if rel and data.status_quo_name is not None:
        if show_context:
            raise ValueError(
                "This plot does not support both context and relativization at "
                "the same time."
            )
        status_quo_arm = data.in_sample[data.status_quo_name]
    else:
        status_quo_arm = None

    for i, metric in enumerate(data.metrics):
        y_raw, se_raw, y_hat, se_hat = _error_scatter_data(
            # Expected `List[typing.Union[PlotInSampleArm,
            # ax.plot.base.PlotOutOfSampleArm]]` for 1st anonymous
            # parameter to call `ax.plot.scatter._error_scatter_data` but got
            # `List[PlotInSampleArm]`.
            # pyre-fixme[6]:
            list(data.in_sample.values()),
            y_axis_var=PlotMetric(metric, pred=True, rel=rel),
            x_axis_var=PlotMetric(metric, pred=False, rel=rel),
            status_quo_arm=status_quo_arm,
        )
        min_, max_ = _get_min_max_with_errors(y_raw, y_hat, se_raw or [], se_hat)
        traces.append(_diagonal_trace(min_, max_, visible=(i == 0)))
        traces.append(
            _error_scatter_trace(
                # Expected `List[typing.Union[PlotInSampleArm,
                # ax.plot.base.PlotOutOfSampleArm]]` for 1st parameter
                # `arms` to call `ax.plot.scatter._error_scatter_trace`
                # but got `List[PlotInSampleArm]`.
                # pyre-fixme[6]:
                arms=list(data.in_sample.values()),
                hoverinfo="text",
                show_arm_details_on_hover=True,
                show_CI=True,
                show_context=show_context,
                status_quo_arm=status_quo_arm,
                visible=(i == 0),
                x_axis_label=xlabel,
                x_axis_var=PlotMetric(metric, pred=False, rel=rel),
                y_axis_label=ylabel,
                y_axis_var=PlotMetric(metric, pred=True, rel=rel),
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

    layout = go.Layout(
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

    return go.Figure(data=traces, layout=layout)


def _get_batch_comparison_plot_data(
    observations: List[Observation],
    batch_x: int,
    batch_y: int,
    rel: bool = False,
    status_quo_name: Optional[str] = None,
) -> PlotData:
    """Compute PlotData for comparing repeated arms across trials.

    Args:
        observations: List of observations.
        batch_x: Batch for x-axis.
        batch_y: Batch for y-axis.
        rel: Whether to relativize data against status_quo arm.
        status_quo_name: Name of the status_quo arm.

    Returns:
        PlotData: a plot data object.
    """
    if rel and status_quo_name is None:
        raise ValueError("Experiment status quo must be set for rel=True")
    x_observations = {
        observation.arm_name: observation
        for observation in observations
        if observation.features.trial_index == batch_x
    }
    y_observations = {
        observation.arm_name: observation
        for observation in observations
        if observation.features.trial_index == batch_y
    }

    # Assume input is well formed and metric_names are consistent across observations
    metric_names = observations[0].data.metric_names
    insample_data: Dict[str, PlotInSampleArm] = {}
    for arm_name, x_observation in x_observations.items():
        # Restrict to arms present in both trials
        if arm_name not in y_observations:
            continue

        y_observation = y_observations[arm_name]
        arm_data = {
            "name": arm_name,
            "y": {},
            "se": {},
            "parameters": x_observation.features.parameters,
            "y_hat": {},
            "se_hat": {},
            "context_stratum": None,
        }
        for i, mname in enumerate(x_observation.data.metric_names):
            # pyre-fixme[16]: Optional type has no attribute `__setitem__`.
            arm_data["y"][mname] = x_observation.data.means[i]
            arm_data["se"][mname] = np.sqrt(x_observation.data.covariance[i][i])
        for i, mname in enumerate(y_observation.data.metric_names):
            arm_data["y_hat"][mname] = y_observation.data.means[i]
            arm_data["se_hat"][mname] = np.sqrt(y_observation.data.covariance[i][i])
        # Expected `str` for 2nd anonymous parameter to call `dict.__setitem__` but got
        # `Optional[str]`.
        # pyre-fixme[6]:
        insample_data[arm_name] = PlotInSampleArm(**arm_data)

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

    # arm_name -> Arm data
    insample_data: Dict[str, PlotInSampleArm] = {}

    # Assume input is well formed and this is consistent
    metric_names = cv_results[0].predicted.metric_names

    for rid, cv_result in enumerate(cv_results):
        arm_name = cv_result.observed.arm_name
        arm_data = {
            "name": cv_result.observed.arm_name,
            "y": {},
            "se": {},
            "parameters": cv_result.observed.features.parameters,
            "y_hat": {},
            "se_hat": {},
            "context_stratum": None,
        }
        for i, mname in enumerate(cv_result.observed.data.metric_names):
            # pyre-fixme[16]: Optional type has no attribute `__setitem__`.
            arm_data["y"][mname] = cv_result.observed.data.means[i]
            arm_data["se"][mname] = np.sqrt(cv_result.observed.data.covariance[i][i])
        for i, mname in enumerate(cv_result.predicted.metric_names):
            arm_data["y_hat"][mname] = cv_result.predicted.means[i]
            arm_data["se_hat"][mname] = np.sqrt(cv_result.predicted.covariance[i][i])

        # Expected `str` for 2nd anonymous parameter to call `dict.__setitem__` but got
        # `Optional[str]`.
        # pyre-fixme[6]:
        insample_data[f"{arm_name}_{rid}"] = PlotInSampleArm(**arm_data)
    return PlotData(
        metrics=metric_names,
        in_sample=insample_data,
        out_of_sample=None,
        status_quo_name=None,
    )


def interact_empirical_model_validation(batch: BatchTrial, data: Data) -> AxPlotConfig:
    """Compare the model predictions for the batch arms against observed data.

    Relies on the model predictions stored on the generator_runs of batch.

    Args:
        batch: Batch on which to perform analysis.
        data: Observed data for the batch.
    Returns:
        AxPlotConfig for the plot.
    """
    insample_data: Dict[str, PlotInSampleArm] = {}
    metric_names = list(data.df["metric_name"].unique())
    for struct in batch.generator_run_structs:
        generator_run = struct.generator_run
        if generator_run.model_predictions is None:
            continue
        for i, arm in enumerate(generator_run.arms):
            arm_data = {
                "name": arm.name_or_short_signature,
                "y": {},
                "se": {},
                "parameters": arm.parameters,
                "y_hat": {},
                "se_hat": {},
                "context_stratum": None,
            }
            predictions = generator_run.model_predictions
            for _, row in data.df[
                data.df["arm_name"] == arm.name_or_short_signature
            ].iterrows():
                metric_name = row["metric_name"]
                # pyre-fixme[16]: Optional type has no attribute `__setitem__`.
                arm_data["y"][metric_name] = row["mean"]
                arm_data["se"][metric_name] = row["sem"]
                arm_data["y_hat"][metric_name] = predictions[0][metric_name][i]
                arm_data["se_hat"][metric_name] = predictions[1][metric_name][
                    metric_name
                ][i]
            # pyre-fixme[6]: Expected `Optional[Dict[str, Union[float, str]]]` for 1s...
            insample_data[arm.name_or_short_signature] = PlotInSampleArm(**arm_data)
    if not insample_data:
        raise ValueError("No model predictions present on the batch.")
    plot_data = PlotData(
        metrics=metric_names,
        in_sample=insample_data,
        out_of_sample=None,
        status_quo_name=None,
    )

    fig = _obs_vs_pred_dropdown_plot(data=plot_data, rel=False)
    fig["layout"]["title"] = "Cross-validation"
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_cross_validation(
    cv_results: List[CVResult], show_context: bool = True
) -> AxPlotConfig:
    """Interactive cross-validation (CV) plotting; select metric via dropdown.

    Note: uses the Plotly version of dropdown (which means that all data is
    stored within the notebook).

    Args:
        cv_results: cross-validation results.
        show_context: if True, show context on hover.

    """
    data = _get_cv_plot_data(cv_results)
    fig = _obs_vs_pred_dropdown_plot(data=data, rel=False, show_context=show_context)
    fig["layout"]["title"] = "Cross-validation"
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def tile_cross_validation(
    cv_results: List[CVResult],
    show_arm_details_on_hover: bool = True,
    show_context: bool = True,
) -> AxPlotConfig:
    """Tile version of CV plots; sorted by 'best fitting' outcomes.

    Plots are sorted in decreasing order using the p-value of a Fisher exact
    test statistic.

    Args:
        cv_results: cross-validation results.
        include_measurement_error: if True, include
            measurement_error metrics in plot.
        show_arm_details_on_hover: if True, display
            parameterizations of arms on hover. Default is True.
        show_context: if True (default), display context on
            hover.

    """
    data = _get_cv_plot_data(cv_results)
    metrics = data.metrics

    # make subplots (2 plots per row)
    nrows = int(np.ceil(len(metrics) / 2))
    ncols = min(len(metrics), 2)
    fig = subplots.make_subplots(
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
        for arm in data.in_sample.values():
            y_hat.append(arm.y_hat[metric])
            se_hat.append(arm.se_hat[metric])
            y_raw.append(arm.y[metric])
            se_raw.append(arm.se[metric])
        min_, max_ = _get_min_max_with_errors(y_raw, y_hat, se_raw, se_hat)
        fig.append_trace(
            _diagonal_trace(min_, max_), int(np.floor(i / 2)) + 1, i % 2 + 1
        )
        fig.append_trace(
            _error_scatter_trace(
                # Expected `List[typing.Union[PlotInSampleArm,
                # ax.plot.base.PlotOutOfSampleArm]]` for 1st anonymous
                # parameter to call `ax.plot.scatter._error_scatter_trace` but
                # got `List[PlotInSampleArm]`.
                # pyre-fixme[6]:
                list(data.in_sample.values()),
                y_axis_var=PlotMetric(metric, pred=True, rel=False),
                x_axis_var=PlotMetric(metric, pred=False, rel=False),
                y_axis_label="Predicted",
                x_axis_label="Actual",
                hoverinfo="text",
                show_arm_details_on_hover=show_arm_details_on_hover,
                show_context=show_context,
            ),
            int(np.floor(i / 2)) + 1,
            i % 2 + 1,
        )

    # if odd number of plots, need to manually remove the last blank subplot
    # generated by `subplots.make_subplots`
    if len(metrics) % 2 == 1:
        fig["layout"].pop("xaxis{}".format(nrows * ncols))
        fig["layout"].pop("yaxis{}".format(nrows * ncols))

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

    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_batch_comparison(
    observations: List[Observation],
    experiment: Experiment,
    batch_x: int,
    batch_y: int,
    rel: bool = False,
    status_quo_name: Optional[str] = None,
) -> AxPlotConfig:
    """Compare repeated arms from two trials; select metric via dropdown.

    Args:
        observations: List of observations to compute comparison.
        batch_x: Index of batch for x-axis.
        batch_y: Index of bach for y-axis.
        rel: Whether to relativize data against status_quo arm.
        status_quo_name: Name of the status_quo arm.
    """
    if isinstance(experiment, MultiTypeExperiment):
        observations = convert_mt_observations(observations, experiment)
    if not status_quo_name and experiment.status_quo:
        status_quo_name = not_none(experiment.status_quo).name
    plot_data = _get_batch_comparison_plot_data(
        observations, batch_x, batch_y, rel=rel, status_quo_name=status_quo_name
    )
    fig = _obs_vs_pred_dropdown_plot(
        data=plot_data,
        rel=rel,
        xlabel="Batch {}".format(batch_x),
        ylabel="Batch {}".format(batch_y),
    )
    fig["layout"]["title"] = "Repeated arms across trials"
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)
