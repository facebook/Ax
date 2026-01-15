# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final, Mapping, Sequence

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import BOTORCH_COLOR_SCALE
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card
from ax.analysis.plotly.utils import (
    get_arm_tooltip,
    get_trial_statuses_with_fallback,
    get_trial_trace_name,
    LEGEND_BASE_OFFSET,
    LEGEND_POSITION,
    MARGIN_REDUCUTION,
    MULTIPLE_CANDIDATE_TRIALS_LEGEND,
    SINGLE_CANDIDATE_TRIAL_LEGEND,
    trial_index_to_color,
    truncate_label,
    X_TICKER_SCALING_FACTOR,
    Z_SCORE_95_CI,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
    validate_experiment_has_trials,
)
from ax.core.analysis_card import AnalysisCard, AnalysisCardGroup
from ax.core.arm import Arm
from ax.core.data import sort_by_trial_index_and_arm_name
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go
from pyre_extensions import none_throws, override


@final
class ArmEffectsPlot(Analysis):
    """
    Plot the effects of each arm in an experiment on a given metric. Effects may be
    either the raw observed effects, or the predicted effects using a model. The
    latter is often more trustworthy (and leads to better reproducibility) than using
    the raw data, especially when model fit is good and in high-noise settings.

    Each arm is represented by a point on the plot with 95% confidence intervals. The
    color of the point indicates the status of the arm's trial (e.g. RUNNING, SUCCEDED,
    FAILED). Arms which are likely to violate a constraint (i.e. according to either
    the raw or modeled effects, the probability all constraints are satisfied is < 5%)
    are marked with a red outline. Each arm also has a hover tooltip with additional
    information.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - trial_status: The status of the trial
        - arm_name: The name of the arm
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METRIC_NAME_mean: The observed mean of the metric specified
        - **METRIC_NAME_sem: The observed sem of the metric specified
    """

    def __init__(
        self,
        metric_name: str,
        use_model_predictions: bool = True,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            metric_name: The name of the metrics to include in the plot.
            use_model_predictions: Whether to use model predictions or observed data.
                If ``True``, the plot will show the predicted effects of each arm based
                on the model. If ``False``, the plot will show the observed effects of
                each arm. The latter is often less trustworthy than the former,
                especially when model fit is good and in high-noise settings.
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            trial_index: If present, only use arms from the trial with the given index.
            trial_statuses: If present, only use arms from trials with the given
                statuses. By default, exclude STALE, ABANDONED, and FAILED trials.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            label: A label to use in the plot in place of the metric name.
        """

        self.metric_name = metric_name
        self.use_model_predictions = use_model_predictions
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses: list[TrialStatus] | None = (
            get_trial_statuses_with_fallback(
                trial_statuses=trial_statuses, trial_index=trial_index
            )
        )
        self.additional_arms = additional_arms
        self.label = label

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        ArmEffectsPlot requires an Experiment with at least one trial with data which
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
                # If using model predictions we do not need to have an observation
                required_metric_names=(
                    None if self.use_model_predictions else [self.metric_name]
                ),
            )
        ) is not None:
            return no_trials_reason

        if self.use_model_predictions:
            if (
                adapter_cannot_predict_reason := validate_adapter_can_predict(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                    required_metric_names=[self.metric_name],
                )
            ) is not None:
                return adapter_cannot_predict_reason

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        experiment = none_throws(experiment)

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
        else:
            relevant_adapter = None

        df = prepare_arm_data(
            experiment=experiment,
            metric_names=[self.metric_name],
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=self.relativize,
        )

        metric_label = (
            self.label
            if self.label is not None
            else truncate_label(label=self.metric_name)
        )

        status_quo_str = (
            experiment.status_quo.name
            if experiment.status_quo is not None
            else "status quo"
        )

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=(
                ("Modeled " if self.use_model_predictions else "Observed ")
                + f"Arm Effects on {metric_label}"
                + (
                    f" for trial {self.trial_index}"
                    if self.trial_index is not None
                    else ""
                )
                + (
                    f' relative to "{status_quo_str}"'
                    if self.relativize and experiment.status_quo is not None
                    else ""
                )
            ),
            subtitle=_get_subtitle(
                metric_label=metric_label,
                use_model_predictions=self.use_model_predictions,
                trial_index=self.trial_index,
            ),
            df=df[
                [
                    "trial_index",
                    "trial_status",
                    "arm_name",
                    "status_reason",
                    "generation_node",
                    f"{self.metric_name}_mean",
                    f"{self.metric_name}_sem",
                ]
            ].copy(),
            fig=_prepare_figure(
                df=df,
                metric_name=self.metric_name,
                is_relative=self.relativize,
                status_quo_arm_name=experiment.status_quo.name
                if experiment.status_quo
                else None,
                metric_label=metric_label,
            ),
        )


def compute_arm_effects_adhoc(
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
    metric_names: Sequence[str] | None = None,
    use_model_predictions: bool = True,
    relativize: bool = False,
    trial_index: int | None = None,
    trial_statuses: Sequence[TrialStatus] | None = None,
    additional_arms: Sequence[Arm] | None = None,
    labels: Mapping[str, str] | None = None,
) -> AnalysisCardGroup:
    """
    Compute ArmEffectsPlot cards for the given experiment and either Adapter or
    GenerationStrategy.

    Note that cards are not saved to the database when computed adhoc -- they are only
    saved when computed as part of a call to ``Client.compute_analyses`` or equivalent.

    Args:
        experiment: The experiment to extract data from.
        metric_names: The names of the metrics to include in the plot. If not
            specified, all metrics in the experiment will be used.
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

    return AnalysisCardGroup(
        name="ArmEffectsPlot",
        title="Adhoc Arm Effects Plots",
        subtitle=None,
        children=[
            ArmEffectsPlot(
                metric_name=metric_name,
                use_model_predictions=use_model_predictions,
                relativize=relativize,
                trial_index=trial_index,
                trial_statuses=trial_statuses,
                additional_arms=additional_arms,
                label=labels.get(metric_name) if labels is not None else None,
            ).compute_or_error_card(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            for metric_name in (
                metric_names
                if metric_names is not None
                else [*experiment.metrics.keys()]
            )
        ],
    )


def _prepare_figure(
    df: pd.DataFrame,
    metric_name: str,
    is_relative: bool,
    status_quo_arm_name: str | None,
    metric_label: str,
) -> go.Figure:
    # Prepare separate scatter traces for each trial index. Each trace has (x, y)
    # points for the arms which we have a mean for in the provided dataframe.
    candidate_trials = df[df["trial_status"] == TrialStatus.CANDIDATE.name][
        "trial_index"
    ].unique()

    trials = df["trial_index"].unique()

    trial_indices = list(trials)
    trial_indices.extend(candidate_trials)

    trials_list = trials.tolist()

    scatters = []
    scatter_trial_indices = []  # Track trial indices for each scatter
    # Sort the dataframe by trial index and arm name.
    # Non-default arm names (not digit + underscore + digit) are sorted to the front.
    # default arm names (e.g. 0_0, 1_8) are sorted that '0_1' < '0_5' < '0_10'
    df = sort_by_trial_index_and_arm_name(df=df)
    # Add a column combining trial_index and arm_name to be used as x mark
    df["x_key_order"] = df["trial_index"].astype(str) + ":" + df["arm_name"]
    arm_order = []
    arm_label = []

    # Track trials that get included in the plot
    num_candidate_trials = 0
    num_non_candidate_trials = 0
    candidate_trial_marker = None

    for trial_index in trial_indices:
        trial_df = df[df["trial_index"] == trial_index]
        xy_df = trial_df[~trial_df[f"{metric_name}_mean"].isna()]
        # Skip trials with no valid data points as they will not end up in the plot
        if xy_df.empty:
            continue
        if is_relative and status_quo_arm_name is not None:
            # Exclude status quo arms from relativized plots, since arms are relative
            # with respect to the status quo.
            xy_df = xy_df[xy_df["arm_name"] != status_quo_arm_name]

        arm_order = arm_order + xy_df["x_key_order"].to_list()
        arm_label = arm_label + xy_df["arm_name"].to_list()
        if not trial_df[f"{metric_name}_sem"].isna().all():
            error_y = {
                "type": "data",
                "array": Z_SCORE_95_CI * xy_df[f"{metric_name}_sem"],
                "color": trial_index_to_color(
                    trial_df=trial_df,
                    trials_list=trials_list,
                    trial_index=trial_index,
                    transparent=True,
                ),
            }
        else:
            error_y = None

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

        text = xy_df.apply(
            lambda row: get_arm_tooltip(row=row, metric_names=[metric_name]), axis=1
        )

        scatters.append(
            go.Scatter(
                x=xy_df["x_key_order"],
                y=xy_df[f"{metric_name}_mean"],
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

    # get the max length of x-ticker (arm name) to set the xaxis label and
    # legend position
    # This assumes the x-tickers are rotated 90 degrees (vertical) so legend
    # will be always below the x-label ('Arm Name').
    max_label_len = max(len(label) for label in arm_label)
    legend_y = LEGEND_BASE_OFFSET - (max_label_len / X_TICKER_SCALING_FACTOR)

    legend_position = LEGEND_POSITION.copy()
    if use_colorscale:
        # Position candidate legend to the right of the colorscale
        legend_position.update(
            {
                "orientation": "v",
                "yanchor": "top",
                "y": -0.33,
                "xanchor": "left",
                "x": 0.9,
            }
        )
    else:
        legend_position["y"] = legend_y

    figure = go.Figure(data=scatters)
    figure.update_layout(
        xaxis_title="Arm Name",
        yaxis_title=metric_label,
        yaxis_tickformat=".2%" if is_relative else None,
        legend=legend_position,
        margin=MARGIN_REDUCUTION,
        xaxis={"tickvals": arm_order, "ticktext": arm_label},
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

    # Add a horizontal line for the status quo.
    if status_quo_arm_name in df["arm_name"].values:
        # In relativized plots the status quo is always 0% on the y-axis.
        if is_relative:
            y = 0
        else:
            y = df[df["arm_name"] == status_quo_arm_name][f"{metric_name}_mean"].iloc[0]

        figure.add_shape(
            type="line",
            xref="paper",
            x0=0,
            y0=y,
            x1=1,
            y1=y,
            line={"color": "gray", "dash": "dot"},
        )

    return figure


def _get_subtitle(
    metric_label: str,
    use_model_predictions: bool,
    trial_index: int | None = None,
) -> str:
    first_clause = (
        f"{'Modeled' if use_model_predictions else 'Observed'} effects on "
        f"{metric_label}"
    )
    trial_clause = f" for Trial {trial_index}." if trial_index is not None else ""
    first_sentence = f"{first_clause}{trial_clause}."

    if use_model_predictions:
        return (
            f"{first_sentence} This plot visualizes predictions of the "
            "true metric changes for each arm based on Ax's model. This is the "
            "expected delta you would expect if you (re-)ran that arm. This plot helps "
            "in anticipating the outcomes and performance of arms based on the model's "
            "predictions. Note, flat predictions across arms indicate that the model "
            "predicts that there is no effect, meaning if you were to re-run the "
            "experiment, the delta you would see would be small and fall within the "
            "confidence interval indicated in the plot."
        )

    return (
        f"{first_sentence} This plot visualizes the effects from previously-run arms "
        "on a specific metric, providing insights into their performance. This plot "
        "allows one to compare and contrast the effectiveness of different arms, "
        "highlighting which configurations have yielded the most favorable outcomes. "
    )
