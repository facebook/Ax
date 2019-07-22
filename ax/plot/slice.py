#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.helper import (
    TNullableGeneratorRunsDict,
    get_fixed_values,
    get_grid_for_parameter,
    get_plot_data,
    get_range_parameter,
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

    config = {
        "arm_data": plot_data,
        "arm_name_to_parameters": cond_name_to_parameters,
        "f": f_plt,
        "fit_data": raw_data,
        "grid": grid,
        "metric": metric_name,
        "param": param_name,
        "rel": relative,
        "setx": fixed_values,
        "sd": sd_plt,
        "is_log": parameter.log_scale,
    }
    return AxPlotConfig(config, plot_type=AxPlotTypes.SLICE)
