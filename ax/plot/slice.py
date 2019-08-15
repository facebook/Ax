#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes, PlotData
from ax.plot.helper import (
    TNullableGeneratorRunsDict,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
)


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
    fixed_values = get_fixed_values(model, slice_values)

    prediction_features = []
    for x in grid:
        predf = deepcopy(fixed_features)
        predf.parameters = fixed_values.copy()
        predf.parameters[param_name] = x
        prediction_features.append(predf)

    f, cov = model.predict(prediction_features)
    f_plt = f[metric_name]
    sd_plt = np.sqrt(cov[metric_name][metric_name])
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
    return AxPlotConfig(config, plot_type=AxPlotTypes.SLICE)


def interact_slice(
    model: ModelBridge,
    param_name: str,
    metric_name: str = "",
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    relative: bool = False,
    density: int = 50,
    slice_values: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[ObservationFeatures] = None,
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
    fixed_values = get_fixed_values(model, slice_values)

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
    return AxPlotConfig(config, plot_type=AxPlotTypes.INTERACT_SLICE)
