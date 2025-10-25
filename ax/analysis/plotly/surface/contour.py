# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import final

import pandas as pd
from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import METRIC_CONTINUOUS_COLOR_SCALE

from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.surface.utils import (
    get_features_for_slice_or_contour,
    get_parameter_values,
    is_axis_log_scale,
)
from ax.analysis.plotly.utils import select_metric, truncate_label
from ax.analysis.utils import (
    extract_relevant_adapter,
    relativize_data,
    validate_adapter_can_predict,
    validate_experiment,
)
from ax.core.experiment import Experiment
from ax.core.parameter import DerivedParameter
from ax.core.trial_status import STATUSES_EXPECTING_DATA
from ax.core.utils import get_target_trial_index
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go
from pyre_extensions import none_throws, override

CONTOUR_CARDGROUP_TITLE = "Contour Plots: Metric effects by parameter values"

CONTOUR_CARDGROUP_SUBTITLE = (
    "These plots show the relationship between a metric and two parameters. They "
    "show the predicted values of the metric (indicated by color) as a function of "
    "the two parameters on the x- and y-axes while keeping all other parameters "
    "fixed at their status_quo value (or mean value if status_quo is unavailable). "
)


@final
class ContourPlot(Analysis):
    """
    Plot a 2D surface of the surrogate model's predicted outcomes for a given pair of
    parameters, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the x parameter specified
        - PARAMETER_NAME: The value of the y parameter specified
        - METRIC_NAME: The predected mean of the metric specified
        - sampled: Whether the parameter values were sampled in at least one trial
    """

    def __init__(
        self,
        x_parameter_name: str,
        y_parameter_name: str,
        metric_name: str | None = None,
        display_sampled: bool = True,
        relativize: bool = False,
    ) -> None:
        """
        Args:
            x_parameter_name: The name of the parameter to plot on the x-axis.
            y_parameter_name: The name of the parameter to plot on the y-axis.
            metric_name: The name of the metric to plot
            display_sampled: If True, plot "x"s at x coordinates which have been
                sampled in at least one trial.
            relativize: If True, relativize the metric values to the status quo.
        """
        self.x_parameter_name = x_parameter_name
        self.y_parameter_name = y_parameter_name
        self.metric_name = metric_name
        self._display_sampled = display_sampled
        self.relativize = relativize

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        ContourPlot requires an Experiment with at least one trial with data as well as
        an Adapter which can predict out of sample points for the specified metric.
        """

        if (
            experiment_invalid_reason := validate_experiment(
                experiment=experiment,
                require_trials=True,
                require_data=True,
            )
        ) is not None:
            return experiment_invalid_reason

        experiment = none_throws(experiment)

        if (
            adapter_cannot_predict_reason := validate_adapter_can_predict(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                required_metric_names=[
                    self.metric_name or select_metric(experiment=experiment)
                ],
            )
        ) is not None:
            return adapter_cannot_predict_reason

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ContourPlot requires an Experiment")
        for name in (self.x_parameter_name, self.y_parameter_name):
            if isinstance(experiment.search_space.parameters[name], DerivedParameter):
                raise UserInputError(
                    f"ContourPlot does not support derived parameters: {name}"
                )

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        metric_name = self.metric_name or select_metric(experiment=experiment)

        df = _prepare_data(
            experiment=experiment,
            model=relevant_adapter,
            x_parameter_name=self.x_parameter_name,
            y_parameter_name=self.y_parameter_name,
            metric_name=metric_name,
            relativize=self.relativize,
        )

        fig = _prepare_plot(
            df=df,
            x_parameter_name=self.x_parameter_name,
            y_parameter_name=self.y_parameter_name,
            metric_name=metric_name,
            log_x=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.x_parameter_name]
            ),
            log_y=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.y_parameter_name]
            ),
            display_sampled=self._display_sampled,
            is_relative=self.relativize,
        )

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=(
                f"{self.x_parameter_name}, {self.y_parameter_name} vs. "
                f"{metric_name}"
            ),
            subtitle=(
                "The contour plot visualizes the predicted outcomes "
                f"for {metric_name} across a two-dimensional parameter space, "
                "with other parameters held fixed at their status_quo value "
                "(or mean value if status_quo is unavailable). This plot helps "
                "in identifying regions of optimal performance and understanding "
                "how changes in the selected parameters influence the predicted "
                "outcomes. Contour lines represent levels of constant predicted "
                "values, providing insights into the gradient and potential optima "
                "within the parameter space."
            ),
            df=df,
            fig=fig,
        )


def compute_contour_adhoc(
    x_parameter_name: str,
    y_parameter_name: str,
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    metric_name: str | None = None,
    display_sampled: bool = True,
    relativize: bool = False,
) -> PlotlyAnalysisCard:
    """
    Helper method to expose adhoc contour plotting. Only for advanced users in
    a notebook setting.

    Args:
        parameter_name: The name of the parameter to plot on the x-axis.
        experiment: The experiment to source data from.
        generation_strategy: Optional. The generation strategy to extract the adapter
            from.
        adapter: Optional. The adapter to use for predictions.
        metric_name: The name of the metric to plot on the y-axis. If not specified
            the objective will be used.
        display_sampled: If True, plot "x"s at x coordinates which have been sampled
            in at least one trial.
        relativize: If True, relativize the metric values to the status quo.
    """
    analysis = ContourPlot(
        x_parameter_name=x_parameter_name,
        y_parameter_name=y_parameter_name,
        metric_name=metric_name,
        display_sampled=display_sampled,
        relativize=relativize,
    )
    return analysis.compute(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )


def _prepare_data(
    experiment: Experiment,
    model: Adapter,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
    relativize: bool,
) -> pd.DataFrame:
    sampled = [
        {
            "x_parameter_name": arm.parameters[x_parameter_name],
            "y_parameter_name": arm.parameters[y_parameter_name],
            "arm_name": arm.name,
            "trial_index": trial.index,
        }
        for trial in experiment.trials.values()
        if trial.status in STATUSES_EXPECTING_DATA  # running, completed, early stopped
        for arm in trial.arms
        # Filter out arms which are not part of the search space (ex. when a parameter
        # is None).
        if experiment.search_space.check_membership(
            parameterization=arm.parameters,
            raise_error=False,
            check_all_parameters_present=False,
        )
    ]

    # Choose which parameter values to predict points for.
    unsampled_xs = get_parameter_values(
        parameter=experiment.search_space.parameters[x_parameter_name], density=10
    )
    unsampled_ys = get_parameter_values(
        parameter=experiment.search_space.parameters[y_parameter_name], density=10
    )

    xs = [*[sample["x_parameter_name"] for sample in sampled], *unsampled_xs]
    ys = [*[sample["y_parameter_name"] for sample in sampled], *unsampled_ys]

    # Construct observation features for each parameter value previously chosen by
    # fixing all other parameters to their status-quo value or mean.
    features = features = [
        get_features_for_slice_or_contour(
            parameters={
                x_parameter_name: x,
                y_parameter_name: y,
            },
            search_space=experiment.search_space,
        )
        for x in xs
        for y in ys
    ]

    predictions = model.predict(observation_features=features)

    df = none_throws(
        pd.DataFrame.from_records(
            [
                {
                    x_parameter_name: features[i].parameters[x_parameter_name],
                    y_parameter_name: features[i].parameters[y_parameter_name],
                    f"{metric_name}_mean": predictions[0][metric_name][i],
                    f"{metric_name}_sem": predictions[1][metric_name][metric_name][i]
                    ** 0.5,
                    "sampled": (
                        features[i].parameters[x_parameter_name],
                        features[i].parameters[y_parameter_name],
                    )
                    in [
                        (s["x_parameter_name"], s["y_parameter_name"]) for s in sampled
                    ],
                    "arm_name": sampled[i]["arm_name"]
                    if i < len(sampled)
                    else "unsampled",
                    "trial_index": sampled[i]["trial_index"]
                    if i < len(sampled)
                    else -1,
                }
                for i in range(len(features))
            ]
        ).drop_duplicates()
    )

    if relativize:
        target_trial_index = none_throws(get_target_trial_index(experiment=experiment))
        df = relativize_data(
            experiment=experiment,
            df=df,
            metric_names=[metric_name],
            is_raw_data=False,
            trial_index=None,
            trial_statuses=None,
            target_trial_index=target_trial_index,
        )

    return df


def _prepare_plot(
    df: pd.DataFrame,
    x_parameter_name: str,
    y_parameter_name: str,
    metric_name: str,
    log_x: bool,
    log_y: bool,
    display_sampled: bool,
    is_relative: bool,
) -> go.Figure:
    z_grid = df.pivot_table(
        index=y_parameter_name,
        columns=x_parameter_name,
        values=f"{metric_name}_mean",
        # aggfunc is required to gracefully handle duplicate values
        aggfunc="mean",
    )

    if is_relative:
        z_values = z_grid.values * 100
    else:
        z_values = z_grid.values

    fig = go.Figure(
        data=go.Contour(
            z=z_values,
            x=z_grid.columns.values,
            y=z_grid.index.values,
            colorscale=METRIC_CONTINUOUS_COLOR_SCALE,
            showscale=True,
            colorbar={
                "title": None,
                "ticksuffix": "%" if is_relative else "",
            },
            hoverinfo="skip",
        ),
        layout=go.Layout(
            xaxis_title=truncate_label(label=x_parameter_name),
            yaxis_title=truncate_label(label=y_parameter_name),
        ),
    )

    if display_sampled:
        x_sampled = df[df["sampled"]][x_parameter_name].tolist()
        y_sampled = df[df["sampled"]][y_parameter_name].tolist()
        samples = go.Scatter(
            x=x_sampled,
            y=y_sampled,
            mode="markers",
            marker={
                "symbol": "x",
                "color": "black",
            },
            name="Sampled",
            showlegend=False,
            hovertemplate="(%{x}, %{y})<extra>Sampled</extra>",
        )

        fig.add_trace(samples)

    # Set the x-axis scale to log if relevant
    if log_x:
        fig.update_xaxes(
            type="log",
            range=[
                math.log10(df[x_parameter_name].min()),
                math.log10(df[x_parameter_name].max()),
            ],
        )
    else:
        fig.update_xaxes(range=[df[x_parameter_name].min(), df[x_parameter_name].max()])

    if log_y:
        fig.update_yaxes(
            type="log",
            range=[
                math.log10(df[y_parameter_name].min()),
                math.log10(df[y_parameter_name].max()),
            ],
        )
    else:
        fig.update_yaxes(range=[df[y_parameter_name].min(), df[y_parameter_name].max()])

    return fig
