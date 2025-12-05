# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any, final, Mapping, Sequence

import numpy as np
import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.registry import Generators
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import BOTORCH_COLOR_SCALE

from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.utils import (
    BEST_LINE_SETTINGS,
    get_arm_tooltip,
    get_trial_trace_name,
    LEGEND_POSITION,
    MARGIN_REDUCUTION,
    MULTIPLE_CANDIDATE_TRIALS_LEGEND,
    SINGLE_CANDIDATE_TRIAL_LEGEND,
    trial_index_to_color,
    truncate_label,
    Z_SCORE_95_CI,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    get_lower_is_better,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
    validate_experiment_has_trials,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from plotly import graph_objects as go
from pyre_extensions import none_throws, override

logger: Logger = get_logger(__name__)

SCATTER_CARDGROUP_TITLE = "Scatter Plot"
SCATTER_CARDGROUP_SUBTITLE = (
    "These plots display the effects of each arm on two metrics "
    "displayed on the x- and y-axes. They are useful for understanding the "
    "trade-off between the two metrics and for visualizing the Pareto frontier."
)


@final
class ScatterPlot(Analysis):
    """
    Plotly Scatter plot for any two metrics. Each arm is represented by a single point
    with 95% confidence intervals if the data is available. Effects may be either the
    raw observed effects, or the predicted effects using a model. The latter is often
    more trustworthy (and leads to better reproducibility) than using the raw data,
    especially when model fit is good and in high-noise settings.


    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - trial_status: The status of the trial
        - arm_name: The name of the arm
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - p_feasible: The probability that the arm is feasible (does not violate any
            constraints)
        - **METRIC_NAME_mean: The observed mean of the metric specified
        - **METRIC_NAME_sem: The observed sem of the metric specified
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        ScatterPlot requires an Experiment with at least one trial with data which
        and for at least one trial pass the trial index / trial status filtering. If
        using model predictions, a suitable adapter must also be provided.
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
            no_trials_reason := validate_experiment_has_trials(
                experiment=experiment,
                trial_indices=[self.trial_index]
                if self.trial_index is not None
                else None,
                trial_statuses=self.trial_statuses,
                required_metric_names=[self.x_metric_name, self.y_metric_name],
            )
        ) is not None:
            return no_trials_reason

        if self.use_model_predictions:
            if (
                adapter_cannot_predict_reason := validate_adapter_can_predict(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                    required_metric_names=[self.x_metric_name, self.y_metric_name],
                )
            ) is not None:
                return adapter_cannot_predict_reason

    def __init__(
        self,
        x_metric_name: str,
        y_metric_name: str,
        use_model_predictions: bool = True,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        labels: Mapping[str, str] | None = None,
        show_pareto_frontier: bool = False,
        title: str | None = None,
    ) -> None:
        """
        Args:
            x_metric_name: The name of the metric to plot on the x-axis.
            y_metric_name: The name of the metric to plot on the y-axis.
            use_model_predictions: Whether to use model predictions or observed data.
                If ``True``, the plot will show the predicted effects of each arm based
                on the model. If ``False``, the plot will show the observed effects of
                each arm. The latter is often less trustworthy than the former,
                especially when model fit is good and in high-noise settings.
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            trial_index: If present, only use arms from the trial with the given index.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
            show_pareto_frontier: Whether to draw a line representing the Pareto
                frontier for the two metrics on the plot.
            title: An optional title for the plot.
        """

        self.x_metric_name = x_metric_name
        self.y_metric_name = y_metric_name
        self.use_model_predictions = use_model_predictions
        self.relativize = relativize
        self.trial_index = trial_index
        # By default, include all trials except those that are abandoned or stale.
        if trial_statuses is not None:
            self.trial_statuses: list[TrialStatus] | None = [*trial_statuses]
        elif self.trial_index is not None:
            self.trial_statuses: list[TrialStatus] | None = None
        else:
            self.trial_statuses: list[TrialStatus] | None = [
                *{*TrialStatus} - {TrialStatus.ABANDONED, TrialStatus.STALE}
            ]
        self.additional_arms = additional_arms
        self.labels: dict[str, str] = {**labels} if labels is not None else {}
        self.show_pareto_frontier = show_pareto_frontier
        self.title = title

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError("ScatterPlot requires an Experiment")

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )

            if not relevant_adapter.can_predict:
                logger.warning(
                    f"Adapter {relevant_adapter} cannot make out of sample "
                    "predictions, falling back to EmpiricalBayesThompson."
                )

                data = (
                    experiment.lookup_data(trial_indices=[self.trial_index])
                    if self.trial_index is not None
                    else experiment.lookup_data()
                )
                relevant_adapter = Generators.EMPIRICAL_BAYES_THOMPSON(
                    experiment=experiment, data=data
                )
        else:
            relevant_adapter = None

        df = prepare_arm_data(
            experiment=experiment,
            metric_names=[self.x_metric_name, self.y_metric_name],
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=self.relativize,
        )

        # Retrieve the metric labels from the mapping provided by the user, defaulting
        # to the metric name if no label is provided, truncated.
        x_metric_label = self.labels.get(
            self.x_metric_name, truncate_label(label=self.x_metric_name)
        )
        y_metric_label = self.labels.get(
            self.y_metric_name, truncate_label(label=self.y_metric_name)
        )
        x_lower_is_better = get_lower_is_better(
            experiment=experiment, metric_name=self.x_metric_name
        )
        y_lower_is_better = get_lower_is_better(
            experiment=experiment, metric_name=self.y_metric_name
        )

        figure = _prepare_figure(
            df=df,
            x_metric_name=self.x_metric_name,
            y_metric_name=self.y_metric_name,
            x_metric_label=x_metric_label,
            y_metric_label=y_metric_label,
            is_relative=self.relativize,
            show_pareto_frontier=self.show_pareto_frontier,
            x_lower_is_better=x_lower_is_better
            if x_lower_is_better is not None
            else False,
            y_lower_is_better=y_lower_is_better
            if y_lower_is_better is not None
            else False,
        )
        if self.title is None:
            title = (
                f"{'Modeled' if self.use_model_predictions else 'Observed'} "
                f"{'Relativized ' if self.relativize else ''}Effects:"
                f" {x_metric_label} vs. {y_metric_label}"
            )
        else:
            title = self.title

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=title,
            subtitle=(
                "This plot displays the effects of each arm on the two selected "
                "metrics. It is useful for understanding the trade-off between "
                "the two metrics and for visualizing the Pareto frontier."
            ),
            df=df,
            fig=figure,
        )


def compute_scatter_adhoc(
    experiment: Experiment,
    x_metric_name: str,
    y_metric_name: str,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    use_model_predictions: bool = True,
    relativize: bool = False,
    trial_index: int | None = None,
    trial_statuses: Sequence[TrialStatus] | None = None,
    additional_arms: Sequence[Arm] | None = None,
    labels: Mapping[str, str] | None = None,
) -> PlotlyAnalysisCard:
    """
    Compute ScatterPlot cards for the given experiment and either Adapter or
    GenerationStrategy.

    Note that cards are not saved to the database when computed adhoc -- they are only
    saved when computed as part of a call to ``Client.compute_analyses`` or equivalent.

    Args:
        experiment: The experiment to extract data from.
        x_metric_name: The name of the metric to plot on the x-axis.
        y_metric_name: The name of the metric to plot on the y-axis.
        generation_strategy: The GenerationStrategy to use for predictions if
            use_model_predictions=True.
        adapter: The adapter to use for predictions if use_model_predictions=True.
        use_model_predictions: Whether to use model predictions or observed data.
            If ``True``, the plot will show the predicted effects of each arm based
            on the model. If ``False``, the plot will show the observed effects of
            each arm. The latter is often less trustworthy than the former,
            especially when model fit is good and in high-noise settings.
        relativize: Whether to relativize the effects of each arm against the status
            quo arm. If multiple status quo arms are present, relativize each arm
            against the status quo arm from the same trial.
        trial_index: If present, only use arms from the trial with the given index.
        additional_arms: If present, include these arms in the plot in addition to
            the arms in the experiment. These arms will be marked as belonging to a
            trial with index -1.
        labels: A mapping from metric names to labels to use in the plot. If a label
            is not provided for a metric, the metric name will be used.
    """
    analysis = ScatterPlot(
        x_metric_name=x_metric_name,
        y_metric_name=y_metric_name,
        use_model_predictions=use_model_predictions,
        relativize=relativize,
        trial_index=trial_index,
        trial_statuses=trial_statuses,
        additional_arms=additional_arms,
        labels=labels,
    )

    return analysis.compute(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )


def get_xy_trial_data(
    trial_df: pd.DataFrame,
    metric_name: str,
    trials_list: list[int],
    trial_index: int,
) -> tuple[pd.Series, dict[str, Any] | None]:
    """Get the mean and SEM for a particular metric and trial_index."""
    error = None
    mean_name = f"{metric_name}_mean"
    sem_name = f"{metric_name}_sem"
    xy_df = trial_df[~trial_df[mean_name].isna()]
    if not xy_df[sem_name].isna().all():
        error = {
            "type": "data",
            "array": xy_df[sem_name] * Z_SCORE_95_CI,
            "color": "silver",
            "thickness": 1,
        }
    else:
        error = None
    return xy_df[mean_name], error


def _prepare_figure(
    df: pd.DataFrame,
    x_metric_name: str,
    y_metric_name: str,
    x_metric_label: str,
    y_metric_label: str,
    is_relative: bool,
    show_pareto_frontier: bool,
    x_lower_is_better: bool,
    y_lower_is_better: bool,
) -> go.Figure:
    # Initialize the Scatters one at a time since we cannot specify multiple different
    # error bar colors from within one trace.
    candidate_trials = df[df["trial_status"] == TrialStatus.CANDIDATE.name][
        "trial_index"
    ].unique()

    trials = df["trial_index"].unique()

    trials_list = trials.tolist()
    trial_indices = trials_list.copy()
    trial_indices.extend(candidate_trials)

    scatters = []
    scatter_trial_indices = []  # Track trial indices for each scatter

    # Track trials that get included in the plot
    num_candidate_trials = 0
    num_non_candidate_trials = 0
    candidate_trial_marker = None

    for trial_index in trial_indices:
        trial_df = df[df["trial_index"] == trial_index]
        mean_x, error_x = get_xy_trial_data(
            trial_df=trial_df,
            metric_name=x_metric_name,
            trials_list=trials_list,
            trial_index=trial_index,
        )
        mean_y, error_y = get_xy_trial_data(
            trial_df=trial_df,
            metric_name=y_metric_name,
            trials_list=trials_list,
            trial_index=trial_index,
        )

        # Skip trials with no meaningful data for either metric
        if mean_x.empty and mean_y.empty:
            continue

        marker = {
            "color": trial_index_to_color(
                trial_df=trial_df,
                trials_list=trials_list,
                trial_index=trial_index,
                transparent=False,
            ),
        }

        if trial_df["trial_status"].iloc[0] == TrialStatus.CANDIDATE.name:
            num_candidate_trials += 1
            candidate_trial_marker = marker
        else:
            num_non_candidate_trials += 1

        text = trial_df.apply(
            lambda row: get_arm_tooltip(
                row=row, metric_names=[x_metric_name, y_metric_name]
            ),
            axis=1,
        )

        scatters.append(
            go.Scatter(
                x=mean_x,
                y=mean_y,
                error_x=error_x,
                error_y=error_y,
                mode="markers",
                marker=marker,
                name=get_trial_trace_name(trial_index=trial_index),
                showlegend=False,  # Will be set after determining use_colorscale
                hoverinfo="text",
                text=text,
                legendgroup="candidate_trials"
                if trial_df["trial_status"].iloc[0] == TrialStatus.CANDIDATE.name
                else None,
            )
        )
        scatter_trial_indices.append(trial_index)

    # Determine use_colorscale based on actual included trials
    use_colorscale = num_non_candidate_trials > 10

    # Update markers and legend settings based on use_colorscale
    for scatter, trial_index in zip(scatters, scatter_trial_indices):
        trial_df = df[df["trial_index"] == trial_index]

        if use_colorscale:
            # Add colorscale settings to marker
            scatter.marker.update(
                {
                    "colorscale": BOTORCH_COLOR_SCALE,
                    "showscale": True,
                    "cmin": min(scatter_trial_indices),
                    "cmax": max(scatter_trial_indices),
                    "colorbar": {
                        "title": "Trial Index",
                        "orientation": "h",
                        "x": 0.4,
                        "xanchor": "center",
                        "y": -0.30,
                        "yanchor": "top",
                    },
                }
            )
        else:
            # Show legend for all non-candidate trials when not using colorscale
            scatter.showlegend = (
                trial_df["trial_status"].iloc[0] != TrialStatus.CANDIDATE.name
            )

    # Determine legend position before creating figure
    legend_position = LEGEND_POSITION.copy()
    if use_colorscale:
        # Position legend to the right, align with colorscale
        legend_position.update(
            {
                "orientation": "v",
                "yanchor": "top",
                "y": -0.33,
                "xanchor": "left",
                "x": 0.9,
            }
        )

    figure = go.Figure(data=scatters)
    figure.update_layout(
        xaxis_title=x_metric_label,
        yaxis_title=y_metric_label,
        xaxis_tickformat=".2%" if is_relative else None,
        yaxis_tickformat=".2%" if is_relative else None,
        legend=legend_position,
        margin=MARGIN_REDUCUTION,
    )

    # Add candidate trial legend at the end
    if num_candidate_trials > 0:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=candidate_trial_marker,
                name=SINGLE_CANDIDATE_TRIAL_LEGEND
                if num_candidate_trials == 1
                else MULTIPLE_CANDIDATE_TRIALS_LEGEND,
                showlegend=True,
                hoverinfo="skip",
                legendgroup="candidate_trials",
            )
        )

    # Add horizontal and vertical lines for the status quo.
    if "status_quo" in df["arm_name"].values:
        x = df[df["arm_name"] == "status_quo"][f"{x_metric_name}_mean"].iloc[0]
        y = df[df["arm_name"] == "status_quo"][f"{y_metric_name}_mean"].iloc[0]
        if not np.isnan(x) or not np.isnan(y):
            figure.add_shape(
                type="line",
                yref="paper",
                x0=x,
                y0=0,
                x1=x,
                y1=1,
                line={"color": "gray", "dash": "dot"},
            )

            figure.add_shape(
                type="line",
                xref="paper",
                x0=0,
                y0=y,
                x1=1,
                y1=y,
                line={"color": "gray", "dash": "dot"},
            )

    if show_pareto_frontier:
        # If there are no arms which are not likely to violate constraints, return the
        # figure as is, without adding a Pareto frontier line.
        if len(df) == 0:
            return figure

        sorted_df = df.sort_values(
            by=f"{x_metric_name}_mean", ascending=x_lower_is_better
        )

        pareto_x = [sorted_df[f"{x_metric_name}_mean"].iloc[0]]
        pareto_y = [sorted_df[f"{y_metric_name}_mean"].iloc[0]]
        for i in range(1, len(sorted_df)):
            if not y_lower_is_better and sorted_df[f"{y_metric_name}_mean"].iloc[
                i
            ] > max(sorted_df[f"{y_metric_name}_mean"].iloc[:i]):
                pareto_x.append(sorted_df[f"{x_metric_name}_mean"].iloc[i])
                pareto_y.append(sorted_df[f"{y_metric_name}_mean"].iloc[i])
            elif y_lower_is_better and sorted_df[f"{y_metric_name}_mean"].iloc[i] < min(
                sorted_df[f"{y_metric_name}_mean"].iloc[:i]
            ):
                pareto_x.append(sorted_df[f"{x_metric_name}_mean"].iloc[i])
                pareto_y.append(sorted_df[f"{y_metric_name}_mean"].iloc[i])

        pareto_trace = go.Scatter(x=pareto_x, y=pareto_y, **BEST_LINE_SETTINGS)

        figure.add_trace(pareto_trace)

    return figure
