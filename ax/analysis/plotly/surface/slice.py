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
from ax.analysis.analysis_card import AnalysisCardBase
from ax.analysis.plotly.color_constants import AX_BLUE
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.surface.utils import (
    get_features_for_slice_or_contour,
    get_parameter_values,
    is_axis_log_scale,
)
from ax.analysis.plotly.utils import (
    get_scatter_point_color,
    select_metric,
    truncate_label,
    Z_SCORE_95_CI,
)
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

SLICE_CARDGROUP_TITLE = "Slice Plots: Metric effects by parameter value"

SLICE_CARDGROUP_SUBTITLE = (
    "These plots show the relationship between a metric and a parameter. They "
    "show the predicted values of the metric on the y-axis as a function of the "
    "parameter on the x-axis while keeping all other parameters fixed at their "
    "status_quo value (or mean value if status_quo is unavailable). "
)


@final
class SlicePlot(Analysis):
    """
    Plot a 1D "slice" of the surrogate model's predicted outcomes for a given
    parameter, where all other parameters are held fixed at their status-quo value or
    mean if no status quo is available.

    The DataFrame computed will contain the following columns:
        - PARAMETER_NAME: The value of the parameter specified
        - METRIC_NAME_mean: The predected mean of the metric specified
        - METRIC_NAME_sem: The predected sem of the metric specified
        - sampled: Whether the parameter value was sampled in at least one trial
    """

    def __init__(
        self,
        parameter_name: str,
        metric_name: str | None = None,
        display_sampled: bool = True,
        relativize: bool = False,
    ) -> None:
        """
        Args:
            parameter_name: The name of the parameter to plot on the x axis.
            metric_name: The name of the metric to plot on the y axis. If not
                specified the objective will be used.
            display_sampled: If True, plot "x"s at x coordinates which have been
                sampled in at least one trial.
            relativize: If True, relativize the metric values to the status quo.
        """
        self.parameter_name = parameter_name
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
        SlicePlot requires an Experiment with at least one trial with data as well as
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
            raise UserInputError("SlicePlot requires an Experiment")

        if isinstance(
            experiment.search_space.parameters[self.parameter_name], DerivedParameter
        ):
            raise UserInputError(
                f"SlicePlot does not support derived parameters: {self.parameter_name}"
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
            parameter_name=self.parameter_name,
            metric_name=metric_name,
            relativize=self.relativize,
        )

        fig = _prepare_plot(
            df=df,
            parameter_name=self.parameter_name,
            metric_name=metric_name,
            log_x=is_axis_log_scale(
                parameter=experiment.search_space.parameters[self.parameter_name]
            ),
            display_sampled=self._display_sampled,
            is_relative=self.relativize,
        )

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=f"{metric_name} vs. {self.parameter_name}",
            subtitle=(
                "The slice plot provides a one-dimensional view of predicted "
                f"outcomes for {metric_name} as a function of a single parameter, "
                "while keeping all other parameters fixed at their status_quo "
                "value (or mean value if status_quo is unavailable). "
                "This visualization helps in understanding the sensitivity and "
                "impact of changes in the selected parameter on the predicted "
                "metric outcomes."
            ),
            df=df,
            fig=fig,
        )


def compute_slice_adhoc(
    parameter_name: str,
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    metric_name: str | None = None,
    display_sampled: bool = True,
    relativize: bool = False,
) -> AnalysisCardBase:
    """
    Helper method to expose adhoc slice plotting. Only for advanced users in
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

    analysis = SlicePlot(
        parameter_name=parameter_name,
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
    parameter_name: str,
    metric_name: str,
    relativize: bool,
) -> pd.DataFrame:
    trials = experiment.extract_relevant_trials(trial_statuses=STATUSES_EXPECTING_DATA)
    sampled_xs = [
        {
            "parameter_value": arm.parameters[parameter_name],
            "arm_name": arm.name,
            "trial_index": trial.index,
        }
        for trial in trials
        for arm in trial.arms
        # Exclude parameter values which are not valid (ex. None when the parameter is
        # not known in a status quo arm).
        if experiment.search_space.parameters[parameter_name].validate(
            arm.parameters[parameter_name]
        )
    ]
    # Choose which parameter values to predict points for.
    unsampled_xs = get_parameter_values(
        parameter=experiment.search_space.parameters[parameter_name]
    )
    xs = [*[sample["parameter_value"] for sample in sampled_xs], *unsampled_xs]

    # Construct observation features for each parameter value previously chosen by
    # fixing all other parameters to their status-quo value or mean.
    features = [
        get_features_for_slice_or_contour(
            parameters={parameter_name: x},
            search_space=experiment.search_space,
        )
        for x in xs
    ]
    predictions = model.predict(observation_features=features)

    df = none_throws(
        pd.DataFrame.from_records(
            [
                {
                    parameter_name: xs[i],
                    f"{metric_name}_mean": predictions[0][metric_name][i],
                    f"{metric_name}_sem": predictions[1][metric_name][metric_name][i]
                    ** 0.5,  # Convert the variance to the SEM
                    "sampled": xs[i]
                    in [sample["parameter_value"] for sample in sampled_xs],
                    "arm_name": sampled_xs[i]["arm_name"]
                    if i < len(sampled_xs)
                    else "unsampled",
                    "trial_index": sampled_xs[i]["trial_index"]
                    if i < len(sampled_xs)
                    else -1,
                }
                for i in range(len(xs))
            ]
        ).drop_duplicates()
    ).sort_values(by=parameter_name)

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
    parameter_name: str,
    metric_name: str,
    log_x: bool,
    display_sampled: bool,
    is_relative: bool = False,
) -> go.Figure:
    x = df[parameter_name].tolist()
    y = df[f"{metric_name}_mean"].tolist()

    # Convert the SEMs to 95% confidence intervals
    y_upper = (
        df[f"{metric_name}_mean"] + Z_SCORE_95_CI * df[f"{metric_name}_sem"]
    ).tolist()
    y_lower = (
        df[f"{metric_name}_mean"] - Z_SCORE_95_CI * df[f"{metric_name}_sem"]
    ).tolist()

    # Draw a line at the mean and a shaded region between the upper and lower bounds
    line = go.Scatter(
        x=x,
        y=y,
        line={"color": AX_BLUE},
        mode="lines",
        name=metric_name,
        showlegend=False,
    )
    error_band = go.Scatter(
        # Concatenate x values in reverse order to create a closed polygon
        x=x + x[::-1],
        # Concatenate upper and lower bounds in reverse order
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor=get_scatter_point_color(hex_color=AX_BLUE, ci_transparency=True),
        line={"color": "rgba(255,255,255,0)"},  # Make "line" transparent
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(
        [line, error_band],
        layout=go.Layout(
            xaxis_title=truncate_label(label=parameter_name),
            yaxis_title=truncate_label(label=metric_name),
            yaxis={"tickformat": ".2%"} if is_relative else None,
        ),
    )

    if display_sampled:
        sampled = df[df["sampled"]]
        x_sampled = sampled[parameter_name].tolist()
        y_sampled = sampled[f"{metric_name}_mean"].tolist()

        samples = go.Scatter(
            x=x_sampled,
            y=y_sampled,
            mode="markers",
            marker={
                "symbol": "x",
                "color": "black",
            },
            name=f"Sampled {parameter_name}",
            showlegend=False,
        )

        fig.add_trace(samples)

    # Set the x-axis scale to log if relevant
    if log_x:
        fig.update_xaxes(
            type="log",
            range=[
                math.log10(df[parameter_name].min()),
                math.log10(df[parameter_name].max()),
            ],
        )
    else:
        fig.update_xaxes(range=[df[parameter_name].min(), df[parameter_name].max()])

    return fig
