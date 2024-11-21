# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import math
from typing import Any

import numpy as np
import numpy.typing as npt

import pandas as pd

import torch
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.observation import ObservationFeatures
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.plot.contour import _get_contour_predictions
from ax.plot.feature_importances import plot_feature_importance_by_feature_plotly
from ax.plot.helper import TNullableGeneratorRunsDict
from ax.plot.slice import _get_slice_predictions
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel

from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
from plotly import graph_objects as go, io as pio
from plotly.subplots import make_subplots
from pyre_extensions import none_throws


TOP_K_TOO_LARGE_ERROR = (
    "Interaction Analysis only supports visualizing the slice/contour for"
    " up to 6 component defined by the `top_k` argument, but received"
    " {} as input."
)
MAX_NUM_PLOT_COMPONENTS: int = 6
PLOT_SIZE: int = 380


def get_model_kwargs(
    use_interaction: bool,
    num_parameters: int,
    torch_device: torch.device | None = None,
) -> dict[str, Any]:
    """Method to get the specific OAK kernel used to identify parameter interactions
    in an Ax experiment. The kernel is an Orthogonal Additive Kernel (OAK), which
    decomposes the objective into an additive sum of main parameter effects and
    pairwise interaction effects. The kernel comes with a sparsity-inducing prior,
    which attempts explain the data with as few components as possible. The
    smoothness of the components is regularized by a lengthscale prior to guard
    against excessicely short lengthscales being fit.

    Args:
        use_interaction: Whether to use interaction effects.
        num_parameters: Number of parameters in the experiment.
        torch_device: The type of torch device to use for the model.
    """
    # A fairly restrictive prior on the lengthscale avoids spurious
    # fits, where a single component is fit to explain all variability.
    # NOTE (hvarfner) Imposing a calibrated sparsity-inducing prior is
    # probably a good add, but second-order components tend to break down
    # even for weak priors.
    return {
        "covar_module_class": OrthogonalAdditiveKernel,
        "covar_module_options": {
            "base_kernel": RBFKernel(
                ard_num_dims=num_parameters,
                lengthscale_prior=LogNormalPrior(2, 1),
            ),
            "dim": num_parameters,
            "dtype": torch.float64,
            "device": torch_device,
            "second_order": use_interaction,
            "coeff_constraint": Positive(transform=torch.exp, inv_transform=torch.log),
        },
        "allow_batched_models": False,
    }


def sort_and_filter_top_k_components(
    indices: dict[str, dict[str, npt.NDArray]],
    k: int,
    most_important: bool = True,
) -> dict[str, dict[str, npt.NDArray]]:
    """Sorts and filter the top k components according to Sobol indices, per metric.

    Args:
        indices: A dictionary of {metric: {component: sobol_index}} Sobol indices.
        k: The number of components to keep.
        most_important: Whether to keep the most or least important components.

    Returns:
        A dictionary of the top k components.
    """
    metrics = list(indices.keys())
    sorted_indices = {
        metric: dict(
            sorted(
                metric_indices.items(),
                key=lambda x: x[1],
                reverse=most_important,
            )
        )
        for metric, metric_indices in indices.items()
    }

    # filter to top k components
    sorted_indices = {
        metric: {
            key: value
            for _, (key, value) in zip(range(k), sorted_indices[metric].items())
        }
        for metric in metrics
    }
    return sorted_indices


class InteractionPlot(PlotlyAnalysis):
    """
    Analysis class which tries to explain the data of an experiment as one- or two-
    dimensional additive components with a level of sparsity in the components. The
    relative importance of each component is quantified by its Sobol index. Each
    component may be visualized through slice or contour plots depending on if it is
    a first order or second order component, respectively.
    """

    def __init__(
        self,
        metric_name: str,
        fit_interactions: bool = True,
        most_important: bool = True,
        seed: int = 0,
        torch_device: torch.device | None = None,
    ) -> None:
        """Constructor for InteractionAnalysis.

        Args:
            metric_name: The metric to analyze.
            fit_interactions: Whether to fit interaction effects in addition to main
                effects.
            most_important: Whether to sort by most or least important features in the
                bar subplot. Also controls whether the six most or least important
                features are plotted in the surface subplots.
            seed: The seed with which to fit the model. Defaults to 0. Used
                to ensure that the model fit is identical across the generation of
                various plots.
            torch_device: The torch device to use for the model.
        """

        self.metric_name = metric_name
        self.fit_interactions = fit_interactions
        self.most_important = most_important
        self.seed = seed
        self.torch_device = torch_device

    def get_model(
        self, experiment: Experiment, metric_names: list[str] | None = None
    ) -> TorchModelBridge:
        """
        Retrieves the modelbridge used for the analysis. The model uses an OAK
        (Orthogonal Additive Kernel) with a sparsity-inducing prior,
        which decomposes the objective into an additive sum of components.
        """
        covar_module_kwargs = get_model_kwargs(
            use_interaction=self.fit_interactions,
            num_parameters=len(experiment.search_space.tunable_parameters),
            torch_device=self.torch_device,
        )
        data = experiment.lookup_data()
        if metric_names:
            data = data.filter(metric_names=metric_names)

        model_bridge = Models.BOTORCH_MODULAR(
            search_space=experiment.search_space,
            experiment=experiment,
            data=data,
            surrogate=Surrogate(**covar_module_kwargs),
        )
        return model_bridge  # pyre-ignore[7] Return type is always a TorchModelBridge

    # pyre-ignore[14] Must pass in an Experiment (not Experiment | None)
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> PlotlyAnalysisCard:
        model_bridge = self.get_model(
            experiment=none_throws(experiment), metric_names=[self.metric_name]
        )
        """
        Compute Sobol index sensitivity for one metric of an experiment. Sensitivity
        is comptuted by component, where a compoent may be either one variable
        (main effect) or two variables (interaction effect). The sensitivity is
        computed using a model fit with an OAK kernel, which decomposes the objective
        to be a sum of components, and where marginal effects can be computed
        accurately.
        """
        experiment = none_throws(experiment)
        model_bridge = self.get_model(experiment, [self.metric_name])
        sens = ax_parameter_sens(
            model_bridge=model_bridge,
            metrics=[self.metric_name],
            order="second" if self.fit_interactions else "first",
            signed=not self.fit_interactions,
        )
        sens = sort_and_filter_top_k_components(
            indices=sens,
            k=6,
        )
        # reformat the keys from tuple to a proper "x1 & x2" string
        interaction_name = "Interaction" if self.fit_interactions else "Main Effect"
        return PlotlyAnalysisCard(
            name="Interaction Analysis",
            title="Feature Importance Analysis",
            subtitle=f"{interaction_name} Analysis for {self.metric_name}",
            level=AnalysisCardLevel.MID,
            df=pd.DataFrame(sens),
            blob=pio.to_json(
                plot_feature_importance_by_feature_plotly(
                    sensitivity_values=sens,  # pyre-ignore[6]
                )
            ),
        )
        if not self.display_components:
            return PlotlyAnalysisCard(
                name="Interaction Analysis",
                title=f"Feature Importance Analysis for {self.metric_name}",
                subtitle=(
                    "Displays the most important features "
                    f"for {self.metric_name} by order of importance."
                ),
                level=AnalysisCardLevel.MID,
                df=pd.DataFrame(sens),
                blob=pio.to_json(
                    plot_feature_importance_by_feature_plotly(
                        sensitivity_values=sens,  # pyre-ignore[6]
                    )
                ),
            )
        else:
            metric_sens = list(sens[self.metric_name].keys())
            return PlotlyAnalysisCard(
                name="OAK Interaction Analysis",
                title=(
                    "Additive Component Feature Importance Analysis "
                    f"for {self.metric_name}"
                ),
                subtitle=(
                    "Displays the most important features' effects "
                    f"on {self.metric_name} by order of importance."
                ),
                level=AnalysisCardLevel.MID,
                df=pd.DataFrame(sens),
                blob=pio.to_json(
                    plot_component_surfaces_plotly(
                        features=metric_sens,
                        model=model_bridge,
                        metric=self.metric_name,
                        plots_share_range=self.plots_share_range,
                    )
                ),
            )


def update_plot_range(max_range: list[float], new_range: list[float]) -> list[float]:
    """Updates the range to include the value.
    Args:
        max_range: Current max_range among all considered ranges.
        new_range: New range to consider to be included.

    Returns:
        The updated max_range.
    """
    if max_range[0] > new_range[0]:
        max_range[0] = new_range[0]
    if max_range[1] < new_range[1]:
        max_range[1] = new_range[1]
    return max_range


def plot_component_surfaces_plotly(
    features: list[str],
    model: TorchModelBridge,
    metric: str,
    plots_share_range: bool = True,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    density: int = 50,
    slice_values: dict[str, Any] | None = None,
    fixed_features: ObservationFeatures | None = None,
    trial_index: int | None = None,
    renormalize: bool = True,
) -> go.Figure:
    """Plots the interaction surfaces for the given features.

    Args:
        features: The features to plot. Can be either 1D or 2D, where 2D features are
            specified as "x1 & x2".
        model: The modelbridge used for prediction.
        metric: The name of the metric to plot.
        plots_share_range: Whether to have all plots should share the same output range.
        generator_runs_dict: The generator runs dict to use.
        density: The density of the grid, i.e. the number of points evaluated in each
            dimension.
        slice_values: The slice values to use for the parameters that are not plotted.
        fixed_features: The fixed features to use.
        trial_index: The trial index to include in the plot.
        renormalize: Whether to renormalize the surface to have zero mean.

    Returns:
        A plotly figure of all the interaction surfaces.
    """
    traces = []
    titles = []
    param_names = []

    # tracks the maximal value range so that all plots of the same type share the same
    # signal range in the final visualization. We cannot just check the largest
    # component's sobol index, as it may not have the largest co-domain.
    surface_range = [float("inf"), -float("inf")]
    slice_range = [float("inf"), -float("inf")]
    first_surface = True
    for feature in features:
        if " & " in feature:
            component_x, component_y = feature.split(" & ")
            trace, minval, maxval = generate_interaction_component(
                model=model,
                component_x=component_x,
                component_y=component_y,
                metric=metric,
                generator_runs_dict=generator_runs_dict,
                density=density,
                slice_values=slice_values,
                fixed_features=fixed_features,
                trial_index=trial_index,
                first_surface=first_surface,
            )
            first_surface = False
            traces.append(trace)
            param_names.append((component_x, component_y))
            titles.append(f"Total effect, {component_x} & {component_y}")
            surface_range = update_plot_range(surface_range, [minval, maxval])
        else:
            trace, minval, maxval = generate_main_effect_component(
                model=model,
                component=feature,
                metric=metric,
                generator_runs_dict=generator_runs_dict,
                density=density,
                slice_values=slice_values,
                fixed_features=fixed_features,
                trial_index=trial_index,
            )
            traces.append(trace)
            param_names.append(feature)
            titles.append(f"Main Effect, {feature}")
            slice_range = update_plot_range(slice_range, [minval, maxval])

    # 1x3 plots if 3 total, 2x2 plots if 4 total, 3x2 plots if 6 total
    num_rows = 1 if len(traces) <= (MAX_NUM_PLOT_COMPONENTS / 2) else 2
    num_cols = math.ceil(len(traces) / num_rows)

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles)
    for plot_idx, trace in enumerate(traces):
        row = plot_idx // num_cols + 1
        col = plot_idx % num_cols + 1
        fig.add_trace(trace, row=row, col=col)
        fig = set_axis_names(
            figure=fig, trace=trace, row=row, col=col, param_names=param_names[plot_idx]
        )

    fig = scale_traces(
        figure=fig,
        traces=traces,
        surface_range=surface_range,
        slice_range=slice_range,
        plots_share_range=plots_share_range,
    )
    fig.update_layout({"width": PLOT_SIZE * num_cols, "height": PLOT_SIZE * num_rows})
    return fig


def generate_main_effect_component(
    model: TorchModelBridge,
    component: str,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    density: int = 50,
    slice_values: dict[str, Any] | None = None,
    fixed_features: ObservationFeatures | None = None,
    trial_index: int | None = None,
) -> tuple[go.Scatter, float, float]:
    """Plots a slice "main effect" of the model for a given component. The values are
    relative to the mean of all predictions, so that the magnitude of the component is
    communicated.

    Args:
        model: The modelbridge used for prediction.
        component: The name of the component to plot.
        metric: The name of the metric to plot.
        generator_runs_dict: The generator runs dict to use.
        density: The density of the grid, i.e. the number of points evaluated in each
            dimension.
        slice_values: The slice values to use for the parameters that are not plotted.
        fixed_features: The fixed features to use.
        trial_index: The trial index to include in the plot.

    Returns:
        A contour plot of the component interaction, and the range of the plot.
    """
    _, _, slice_mean, _, grid, _, _, _, _, slice_stdev, _ = _get_slice_predictions(
        model=model,
        param_name=component,
        metric_name=metric,
        generator_runs_dict=generator_runs_dict,
        density=density,
        slice_values=slice_values,
        fixed_features=fixed_features,
        trial_index=trial_index,
    )
    # renormalize the slice to have zero mean (done for each component)
    slice_mean = np.array(slice_mean) - np.array(slice_mean).mean()

    trace = go.Scatter(
        x=grid,
        y=slice_mean,
        name=component,
        line_shape="spline",
        showlegend=False,
        error_y={
            "type": "data",
            "array": slice_stdev,
            "visible": True,
            "thickness": 0.8,
        },
    )

    return trace, np.min(slice_mean).astype(float), np.max(slice_mean).astype(float)


def generate_interaction_component(
    model: TorchModelBridge,
    component_x: str,
    component_y: str,
    metric: str,
    generator_runs_dict: TNullableGeneratorRunsDict = None,
    density: int = 50,
    slice_values: dict[str, Any] | None = None,
    fixed_features: ObservationFeatures | None = None,
    trial_index: int | None = None,
    renormalize: bool = True,
    first_surface: bool = True,
) -> tuple[go.Contour, float, float]:
    """Plots a slice "main effect" of the model for a given component. The values are
    relative to the mean of all predictions, so that the magnitude of the component is
    communicated.

    Args:
        model: The modelbridge used for prediction.
        component_x: The name of the component to plot along the x-axis.
        component_y: The name of the component to plot along the y-axis.
        metric: The name of the metric to plot.
        subtract_main_effects: Whether to subtract the main effects from the 2D surface.
            If main effects are not subtracted, the 2D surface is the output of
            plot_contour and models the effect of each parameter in isolation and their
            interaction. If main effects are subtracted, the 2D surface visualizes only
            the interaction effect of the two parameters.
        generator_runs_dict: The generator runs dict to use.
        density: The density of the grid, i.e. the number of points evaluated in each
            dimension.
        slice_values: The slice values to use for the parameters that are not plotted.
        fixed_features: The fixed features to use.
        trial_index: The trial index to include in the plot.
        renormalize: Whether to renormalize the surface to have zero mean.
        first_surface: Whether this is the first surface to be plotted. If so, we plot
            its colorbar.

    Returns:
        A contour plot of the component interaction, and the range of the plot.
    """
    comp_name: str = f"{component_x} & {component_y}"
    fixed_kwargs: dict[str, Any] = {
        "model": model,
        "generator_runs_dict": generator_runs_dict,
        "density": density,
        "slice_values": slice_values,
        "fixed_features": fixed_features,
    }
    _, contour_mean, _, grid_x, grid_y, _ = _get_contour_predictions(
        x_param_name=component_x,
        y_param_name=component_y,
        metric=metric,
        **fixed_kwargs,
    )
    contour_mean = np.reshape(contour_mean, (density, density))
    contour_mean = contour_mean - contour_mean.mean()
    return (
        go.Contour(
            z=contour_mean,
            x=grid_x,
            y=grid_y,
            name=comp_name,
            ncontours=50,
            showscale=first_surface,
        ),
        np.min(contour_mean).astype(float),
        np.max(contour_mean).astype(float),
    )


def scale_traces(
    figure: go.Figure,
    traces: list[go.Scatter | go.Contour],
    surface_range: list[float],
    slice_range: list[float],
    plots_share_range: bool = True,
) -> go.Figure:
    """Scales the traces to have the same range.

    Args:
        figure: The main plotly figure to update the traces on.
        traces: The traces to scale.
        surface_range: The range of the surface traces.
        slice_range: The range of the slice traces.
        plots_share_range: Whether to have all plots (and not just plots
        of the same type) share the same output range.

    Returns:
        A figure with the traces of the same type are scaled to have the same range.
    """
    if plots_share_range:
        total_range = update_plot_range(surface_range, slice_range)
        slice_range = total_range
        surface_range = total_range

    # plotly axis names in layout are of the form "xaxis{idx}" and "yaxis{idx}" except
    # for the first one, which is "xaxis" and "yaxis". We need to keep track of the
    # indices of the traces and then use the correct axis names when updating ranges.
    axis_names = ["yaxis"] + [f"yaxis{idx}" for idx in range(2, len(traces) + 1)]
    slice_axes = [
        axis_name
        for trace, axis_name in zip(traces, axis_names)
        if isinstance(trace, go.Scatter)
    ]

    # scale the surface traces to have the same range
    for trace_idx in range(len(figure["data"])):
        trace = figure["data"][trace_idx]
        if isinstance(trace, go.Contour):
            trace["zmin"] = surface_range[0]
            trace["zmax"] = surface_range[1]

    # and scale the slice traces to have the same range
    figure.update_layout({ax: {"range": slice_range} for ax in slice_axes})
    return figure


def set_axis_names(
    figure: go.Figure,
    trace: go.Contour | go.Scatter,
    row: int,
    col: int,
    param_names: str | tuple[str, str],
) -> go.Figure:
    """Sets the axis names for the given row and column.

    Args:
        figure: The figure to update the axes on.
        trace: The trace of the plot whose axis labels to update.
        row: The row index of the trace in `figure`.
        col: The column index of the trace in `figure`.
        param_names: The parameter names to use for the axis names.

    Returns:
        A figure where the trace at (row, col) has its axis names set.
    """
    if isinstance(trace, go.Contour):
        X_name, Y_name = param_names
        figure.update_xaxes(title_text=X_name, row=row, col=col)
        figure.update_yaxes(title_text=Y_name, row=row, col=col)
    else:
        figure.update_xaxes(title_text=param_names, row=row, col=col)
    return figure
