# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import Logger

import pandas as pd
import torch
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard

from ax.analysis.plotly.surface.contour import (
    _prepare_data as _prepare_contour_data,
    _prepare_plot as _prepare_contour_plot,
)
from ax.analysis.plotly.surface.slice import (
    _prepare_data as _prepare_slice_data,
    _prepare_plot as _prepare_slice_plot,
)
from ax.analysis.plotly.surface.utils import is_axis_log_scale
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.logger import get_logger
from ax.utils.sensitivity.sobol_measures import ax_parameter_sens
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel

from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)


class InteractionPlot(PlotlyAnalysis):
    """
    Analysis class which tries to explain the data of an experiment as one- or two-
    dimensional additive components with a level of sparsity in the components. The
    relative importance of each component is quantified by its Sobol index. Each
    component may be visualized through slice or contour plots depending on if it is
    a first order or second order component, respectively.

    The DataFrame computed will contain just the sensitivity analyisis with one row per
    parameter and the following columns:
        - feature: The name of the first or second order component
        - sensitivity: The sensitivity of the component
    """

    def __init__(
        self,
        metric_name: str | None = None,
        fit_interactions: bool = True,
        most_important: bool = True,
        seed: int = 0,
        torch_device: torch.device | None = None,
    ) -> None:
        """
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

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> PlotlyAnalysisCard:
        """
        Compute Sobol index sensitivity for one metric of an experiment. Sensitivity
        is comptuted by component, where a compoent may be either one variable
        (main effect) or two variables (interaction effect). The sensitivity is
        computed using a model fit with an OAK kernel, which decomposes the objective
        to be a sum of components, and where marginal effects can be computed
        accurately.
        """

        if experiment is None:
            raise UserInputError("InteractionPlot requires an Experiment")

        metric_name = self.metric_name or select_metric(experiment=experiment)

        # Fix the seed to ensure that the model is fit identically across different
        # analyses of the same experiment.
        with torch.random.fork_rng():
            torch.torch.manual_seed(self.seed)

            # Fit the OAK model.
            oak_model = self._get_oak_model(
                experiment=experiment, metric_name=metric_name
            )

            # Calculate first- or second-order Sobol indices.
            sens = ax_parameter_sens(
                model_bridge=oak_model,
                metrics=[metric_name],
                order="second" if self.fit_interactions else "first",
                signed=not self.fit_interactions,
            )[metric_name]

        sensitivity_df = pd.DataFrame(
            [*sens.items()], columns=["feature", "sensitivity"]
        ).sort_values(by="sensitivity", key=abs, ascending=self.most_important)

        # Calculate feature importance bar plot. Only plot the top 15 features.
        # Plot the absolute value of the sensitivity but color by the sign.
        plotting_df = sensitivity_df.head(15).copy()
        plotting_df["direction"] = plotting_df["sensitivity"].apply(
            lambda x: "Increases Metric" if x >= 0 else "Decreases Metric"
        )
        plotting_df["sensitivity"] = plotting_df["sensitivity"].abs()
        plotting_df.sort_values(
            by="sensitivity", ascending=self.most_important, inplace=True
        )

        sensitivity_fig = px.bar(
            plotting_df,
            x="sensitivity",
            y="feature",
            color="direction",
            # Increase gets blue, decrease gets orange.
            color_discrete_sequence=["orange", "blue"],
            orientation="h",
        )

        # Calculate surface plots for six most or least important features
        # Note: We use tail and reverse here because the bar plot is sorted from top to
        # bottom.
        top_features = [*reversed(plotting_df.tail(6)["feature"].to_list())]
        surface_figs = []
        for feature_name in top_features:
            try:
                surface_figs.append(
                    _prepare_surface_plot(
                        experiment=experiment,
                        model=oak_model,
                        feature_name=feature_name,
                        metric_name=metric_name,
                    )
                )
            # Not all features will be able to be plotted, skip them gracefully.
            except Exception as e:
                logger.error(f"Failed to generate surface plot for {feature_name}: {e}")

        # Create a plotly figure to hold the bar plot in the top row and surface plots
        # in a 2x3 grid below.
        fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}],
                [{}, {}],
            ],
        )

        for trace in sensitivity_fig.data:
            fig.add_trace(trace, row=1, col=1)
        # Fix order of the features in the bar plot.
        fig.update_yaxes(
            categoryorder="array", categoryarray=plotting_df["feature"], row=1, col=1
        )

        for i in range(len(surface_figs)):
            feature_name = top_features[i]
            surface_fig = surface_figs[i]

            row = (i // 2) + 2
            col = (i % 2) + 1
            for trace in surface_fig.data:
                fig.add_trace(trace, row=row, col=col)

            # Label and fix axes
            if "&" in feature_name:  # If the feature is a second-order component
                x, y = feature_name.split(" & ")

                # Reapply log scale if necessary
                fig.update_xaxes(
                    title_text=x,
                    type=(
                        "log"
                        if is_axis_log_scale(
                            parameter=experiment.search_space.parameters[x]
                        )
                        else "linear"
                    ),
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    title_text=y,
                    type=(
                        "log"
                        if is_axis_log_scale(
                            parameter=experiment.search_space.parameters[y]
                        )
                        else "linear"
                    ),
                    row=row,
                    col=col,
                )
            else:  # If the feature is a first-order component
                fig.update_xaxes(
                    title_text=feature_name,
                    type=(
                        "log"
                        if is_axis_log_scale(
                            parameter=experiment.search_space.parameters[feature_name]
                        )
                        else "linear"
                    ),
                    row=row,
                    col=col,
                )

        # Expand layout since default rendering in most notebooks is too small.
        fig.update_layout(
            height=1500,
            width=1000,
        )

        subtitle_substring = (
            "one- or two-dimensional" if self.fit_interactions else "one-dimensional"
        )

        return self._create_plotly_analysis_card(
            title=f"Interaction Analysis for {metric_name}",
            subtitle=(
                f"Understand an Experiment's data as {subtitle_substring} additive "
                "components with sparsity. Important components are visualized through "
                "slice or contour plots"
            ),
            level=AnalysisCardLevel.MID,
            df=sensitivity_df,
            fig=fig,
        )

    def _get_oak_model(
        self, experiment: Experiment, metric_name: str
    ) -> TorchModelBridge:
        """
        Retrieves the modelbridge used for the analysis. The model uses an OAK
        (Orthogonal Additive Kernel) with a sparsity-inducing prior,
        which decomposes the objective into an additive sum of components.

        The kernel comes with a sparsity-inducing prior, which attempts explain the
        data with as few components as possible. The smoothness of the components is
        regularized by a lengthscale prior to guard against excessicely short
        lengthscales being fit.
        """
        data = experiment.lookup_data().filter(metric_names=[metric_name])
        model_bridge = Models.BOTORCH_MODULAR(
            search_space=experiment.search_space,
            experiment=experiment,
            data=data,
            surrogate=Surrogate(
                covar_module_class=OrthogonalAdditiveKernel,
                covar_module_options={
                    # A fairly restrictive prior on the lengthscale avoids spurious
                    # fits, where a single component is fit to explain all
                    # variability.
                    # NOTE (hvarfner) Imposing a calibrated sparsity-inducing prior
                    # is probably a good add, but second-order components tend to
                    # break down even for weak priors.
                    "base_kernel": RBFKernel(
                        ard_num_dims=len(experiment.search_space.tunable_parameters),
                        lengthscale_prior=LogNormalPrior(2, 1),
                    ),
                    "dim": len(experiment.search_space.tunable_parameters),
                    "dtype": torch.float64,
                    "device": self.torch_device,
                    "second_order": self.fit_interactions,
                    "coeff_constraint": Positive(
                        transform=torch.exp, inv_transform=torch.log
                    ),
                },
                allow_batched_models=False,
            ),
        )

        return assert_is_instance(model_bridge, TorchModelBridge)


def _prepare_surface_plot(
    experiment: Experiment,
    model: TorchModelBridge,
    feature_name: str,
    metric_name: str,
) -> go.Figure:
    if "&" in feature_name:
        # Plot a contour plot for the second-order component.
        x_parameter_name, y_parameter_name = feature_name.split(" & ")
        df = _prepare_contour_data(
            experiment=experiment,
            model=model,
            x_parameter_name=x_parameter_name,
            y_parameter_name=y_parameter_name,
            metric_name=metric_name,
        )

        return _prepare_contour_plot(
            df=df,
            x_parameter_name=x_parameter_name,
            y_parameter_name=y_parameter_name,
            metric_name=metric_name,
            log_x=is_axis_log_scale(
                parameter=experiment.search_space.parameters[x_parameter_name]
            ),
            log_y=is_axis_log_scale(
                parameter=experiment.search_space.parameters[y_parameter_name]
            ),
        )

    # If the feature is a first-order component, plot a slice plot.
    df = _prepare_slice_data(
        experiment=experiment,
        model=model,
        parameter_name=feature_name,
        metric_name=metric_name,
    )

    return _prepare_slice_plot(
        df=df,
        parameter_name=feature_name,
        metric_name=metric_name,
        log_x=is_axis_log_scale(
            parameter=experiment.search_space.parameters[feature_name]
        ),
    )
