#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes, PlotData
from ax.plot.helper import (
    TNullableGeneratorRunsDict,
    axis_range,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
    slice_config_to_trace,
)
from plotly import graph_objs as go


# type aliases
SlicePredictions = Tuple[
    PlotData,
    List[Dict[str, Union[str, float]]],
    List[float],
    np.ndarray,
    np.ndarray,
    str,
    str,
    bool,
    Dict[str, Optional[Union[str, bool, float, int]]],
    np.ndarray,
    bool,
]


def _get_slice_predictions(
    model: ModelBridge,
    param_name: str,
    metric_name: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    trial_index: Optional[int] = None,
) -> SlicePredictions:
    """Computes slice prediction configuration values for a single metric name.

    Args:
        model: ModelBridge that contains model for predictions
        param_name: Name of parameter that will be sliced
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose arms will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters. Ignored if
            fixed_features is specified.
        fixed_features: An ObservationFeatures object containing the values of
            features (including non-parameter features like context) to be set
            in the slice.

    Returns: Configruation values for AxPlotConfig.
    """
    if generator_runs_dict is None:
        generator_runs_dict = {}

    parameter = get_range_parameter(model, param_name)
    grid = get_grid_for_parameter(parameter, density)

    plot_data, raw_data, cond_name_to_parameters = get_plot_data(
        model=model, generator_runs_dict=generator_runs_dict, metric_names={metric_name}
    )

    if fixed_features is not None:
        slice_values = fixed_features.parameters
    else:
        fixed_features = ObservationFeatures(parameters={})
    fixed_values = get_fixed_values(model, slice_values, trial_index)

    prediction_features = []
    for x in grid:
        predf = deepcopy(fixed_features)
        predf.parameters = fixed_values.copy()
        predf.parameters[param_name] = x
        prediction_features.append(predf)

    f, cov = model.predict(prediction_features)
    f_plt = f[metric_name]
    sd_plt = np.sqrt(cov[metric_name][metric_name])
    # pyre-fixme[7]: Expected `Tuple[PlotData, List[Dict[str, Union[float, str]]],
    #  List[float], np.ndarray, np.ndarray, str, str, bool, Dict[str, Union[None, bool,
    #  float, int, str]], np.ndarray, bool]` but got `Tuple[PlotData, Dict[str,
    #  Dict[str, Union[None, bool, float, int, str]]], List[float], List[Dict[str,
    #  Union[float, str]]], np.ndarray, str, str, bool, Dict[str, Union[None, bool,
    #  float, int, str]], typing.Any, bool]`.
    return (
        plot_data,
        cond_name_to_parameters,
        f_plt,
        raw_data,
        grid,
        metric_name,
        param_name,
        relative,
        fixed_values,
        sd_plt,
        parameter.log_scale,
    )


def plot_slice(
    model: ModelBridge,
    param_name: str,
    metric_name: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    trial_index: Optional[int] = None,
) -> AxPlotConfig:
    """Plot predictions for a 1-d slice of the parameter space.

    Args:
        model: ModelBridge that contains model for predictions
        param_name: Name of parameter that will be sliced
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose arms will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters. Ignored if
            fixed_features is specified.
        fixed_features: An ObservationFeatures object containing the values of
            features (including non-parameter features like context) to be set
            in the slice.
    """
    pd, cntp, f_plt, rd, grid, _, _, _, fv, sd_plt, ls = _get_slice_predictions(
        model=model,
        param_name=param_name,
        metric_name=metric_name,
        generator_runs_dict=generator_runs_dict,
        relative=relative,
        density=density,
        slice_values=slice_values,
        fixed_features=fixed_features,
        trial_index=trial_index,
    )

    config = {
        "arm_data": pd,
        "arm_name_to_parameters": cntp,
        "f": f_plt,
        "fit_data": rd,
        "grid": grid,
        "metric": metric_name,
        "param": param_name,
        "rel": relative,
        "setx": fv,
        "sd": sd_plt,
        "is_log": ls,
    }
    config = AxPlotConfig(config, plot_type=AxPlotTypes.GENERIC).data

    arm_data = config["arm_data"]
    arm_name_to_parameters = config["arm_name_to_parameters"]
    f = config["f"]
    fit_data = config["fit_data"]
    grid = config["grid"]
    metric = config["metric"]
    param = config["param"]
    rel = config["rel"]
    setx = config["setx"]
    sd = config["sd"]
    is_log = config["is_log"]

    traces = slice_config_to_trace(
        arm_data,
        arm_name_to_parameters,
        f,
        fit_data,
        grid,
        metric,
        param,
        rel,
        setx,
        sd,
        is_log,
        True,
    )

    # layout
    xrange = axis_range(grid, is_log)
    xtype = "log" if is_log else "linear"

    layout = {
        "hovermode": "closest",
        "xaxis": {
            "anchor": "y",
            "autorange": False,
            "exponentformat": "e",
            "range": xrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": param,
            "type": xtype,
        },
        "yaxis": {
            "anchor": "x",
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": metric,
        },
    }

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)


def interact_slice(
    model: ModelBridge,
    param_name: str,
    metric_name: str = "",
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
    trial_index: Optional[int] = None,
) -> AxPlotConfig:
    """Create interactive plot with predictions for a 1-d slice of the parameter
    space.

    Args:
        model: ModelBridge that contains model for predictions
        param_name: Name of parameter that will be sliced
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose arms will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters. Ignored if
            fixed_features is specified.
        fixed_features: An ObservationFeatures object containing the values of
            features (including non-parameter features like context) to be set
            in the slice.
    """
    if generator_runs_dict is None:
        generator_runs_dict = {}

    metric_names = list(model.metric_names)

    parameter = get_range_parameter(model, param_name)
    grid = get_grid_for_parameter(parameter, density)

    plot_data_dict = {}
    raw_data_dict = {}
    sd_plt_dict: Dict[str, Dict[str, np.ndarray]] = {}

    cond_name_to_parameters_dict = {}
    is_log_dict: Dict[str, bool] = {}

    if fixed_features is not None:
        slice_values = fixed_features.parameters
    else:
        fixed_features = ObservationFeatures(parameters={})
    fixed_values = get_fixed_values(model, slice_values, trial_index)

    prediction_features = []
    for x in grid:
        predf = deepcopy(fixed_features)
        predf.parameters = fixed_values.copy()
        predf.parameters[param_name] = x
        prediction_features.append(predf)

    f, cov = model.predict(prediction_features)

    for metric_name in metric_names:
        pd, cntp, f_plt, rd, _, _, _, _, _, sd_plt, ls = _get_slice_predictions(
            model=model,
            param_name=param_name,
            metric_name=metric_name,
            generator_runs_dict=generator_runs_dict,
            relative=relative,
            density=density,
            slice_values=slice_values,
            fixed_features=fixed_features,
        )

        plot_data_dict[metric_name] = pd
        raw_data_dict[metric_name] = rd
        cond_name_to_parameters_dict[metric_name] = cntp

        sd_plt_dict[metric_name] = np.sqrt(cov[metric_name][metric_name])
        is_log_dict[metric_name] = ls

    config = {
        "arm_data": plot_data_dict,
        "arm_name_to_parameters": cond_name_to_parameters_dict,
        "f": f,
        "fit_data": raw_data_dict,
        "grid": grid,
        "metrics": metric_names,
        "param": param_name,
        "rel": relative,
        "setx": fixed_values,
        "sd": sd_plt_dict,
        "is_log": is_log_dict,
    }
    config = AxPlotConfig(config, plot_type=AxPlotTypes.GENERIC).data

    arm_data = config["arm_data"]
    arm_name_to_parameters = config["arm_name_to_parameters"]
    f = config["f"]
    fit_data = config["fit_data"]
    grid = config["grid"]
    metrics = config["metrics"]
    param = config["param"]
    rel = config["rel"]
    setx = config["setx"]
    sd = config["sd"]
    is_log = config["is_log"]

    traces = []

    for i, metric in enumerate(metrics):
        cur_visible = i == 0
        metric = metrics[i]
        traces.extend(
            slice_config_to_trace(
                arm_data[metric],
                arm_name_to_parameters[metric],
                f[metric],
                fit_data[metric],
                grid,
                metric,
                param,
                rel,
                setx,
                sd[metric],
                is_log[metric],
                cur_visible,
            )
        )

    # layout
    xrange = axis_range(grid, is_log[metrics[0]])
    xtype = "log" if is_log[metrics[0]] else "linear"

    buttons = []
    for i, metric in enumerate(metrics):
        trace_cnt = 3 + len(arm_data[metric]["out_of_sample"].keys()) * 2
        visible = [False] * (len(metrics) * trace_cnt)
        for j in range(i * trace_cnt, (i + 1) * trace_cnt):
            visible[j] = True
        buttons.append(
            {
                "method": "update",
                "args": [{"visible": visible}, {"yaxis.title": metric}],
                "label": metric,
            }
        )

    layout = {
        "title": "Predictions for a 1-d slice of the parameter space",
        "annotations": [
            {
                "showarrow": False,
                "text": "Choose metric:",
                "x": 0.225,
                "xanchor": "center",
                "xref": "paper",
                "y": 1.005,
                "yanchor": "bottom",
                "yref": "paper",
            }
        ],
        "updatemenus": [{"y": 1.1, "x": 0.5, "yanchor": "top", "buttons": buttons}],
        "hovermode": "closest",
        "xaxis": {
            "anchor": "y",
            "autorange": False,
            "exponentformat": "e",
            "range": xrange,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": param,
            "type": xtype,
        },
        "yaxis": {
            "anchor": "x",
            "autorange": True,
            "tickfont": {"size": 11},
            "tickmode": "auto",
            "title": metrics[0],
        },
    }

    fig = go.Figure(data=traces, layout=layout)
    return AxPlotConfig(data=fig, plot_type=AxPlotTypes.GENERIC)
