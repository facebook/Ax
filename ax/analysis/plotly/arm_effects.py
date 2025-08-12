# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Mapping, Sequence

import numpy as np

import pandas as pd
from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardBase
from ax.analysis.plotly.color_constants import CONSTRAINT_VIOLATION_RED
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card
from ax.analysis.plotly.utils import (
    BEST_LINE_SETTINGS,
    get_arm_tooltip,
    get_trial_trace_name,
    LEGEND_BASE_OFFSET,
    LEGEND_POSITION,
    MARGIN_REDUCUTION,
    trial_index_to_color,
    truncate_label,
    X_TICKER_SCALING_FACTOR,
    Z_SCORE_95_CI,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    get_lower_is_better,
    POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD,
    prepare_arm_data,
    update_metric_names_if_using_p_feasible,
)
from ax.core.arm import Arm
from ax.core.base_trial import sort_by_trial_index_and_arm_name
from ax.core.experiment import Experiment
from ax.core.trial_status import FAILED_ABANDONED_CANDIDATE_STATUSES, TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from plotly import graph_objects as go
from pyre_extensions import override

CARDGROUP_TITLE = "Metric Effects: Values of key metrics for all arms in the experiment"

PREDICTED_EFFECTS_CARDGROUP_SUBTITLE = (
    "These plots visualize predictions of the 'true' metric changes for each arm, "
    "based on Ax's model. Since Ax applies Empirical Bayes shrinkage to adjust for "
    "noise and also accounts for non-stationarity in the data, predicted metric "
    "effects will not match raw observed data perfectly, but will be more "
    "representative of the reproducible effects that will manifest in a long-term "
    "validation experiment. <br><br>"
    "NOTE: Flat predictions across arms indicate that the model predicts that "
    "none of the arms had a sufficient effect on the metric, meaning that if you "
    "re-ran the experiment, the delta you would see would be small and fall "
    "within the confidence interval indicated in the plot. In other words, this "
    "indicates that according to the model, the raw observed effects on this metric "
    "are primarily noise."
)

RAW_EFFECTS_CARDGROUP_SUBTITLE = (
    "These plots visualize the raw data on the effects we observed from "
    "previously-run arms on a specific metric, providing insights into "
    "their performance. These plots allow one to compare and contrast the "
    "effectiveness of different arms, highlighting which configurations have yielded "
    "the most favorable outcomes."
)


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
        - p_feasible: The probability that the arm is feasible (does not violate any
            constraints)
        - **METRIC_NAME_mean: The observed mean of the metric specified
        - **METRIC_NAME_sem: The observed sem of the metric specified
    """

    def __init__(
        self,
        metric_names: Sequence[str] | None = None,
        use_model_predictions: bool = True,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        labels: Mapping[str, str] | None = None,
        show_cumulative_best: bool = False,
    ) -> None:
        """
        Args:
            metric_names: The names of the metrics to include in the plot. If not
                specified, all metrics in the experiment will be used.
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
            show_cumulative_best: Whether to draw a line through the best point seen so
                far during the optimization.
        """

        self.metric_names = metric_names
        self.use_model_predictions = use_model_predictions
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms
        self.labels: Mapping[str, str] = labels or {}
        self.show_cumulative_best = show_cumulative_best

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        if experiment is None:
            raise UserInputError("ArmEffectsPlot requires an Experiment.")

        metric_names = self.metric_names or [*experiment.metrics.keys()]

        if self.use_model_predictions:
            relevant_adapter = extract_relevant_adapter(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
        else:
            relevant_adapter = None
        metric_names_to_plot = metric_names
        # if using p_feasible, we need to get the data for the metrics involved
        # in constraints even though we don't plot them
        metric_names = update_metric_names_if_using_p_feasible(
            metric_names=metric_names_to_plot, experiment=experiment
        )
        df = prepare_arm_data(
            experiment=experiment,
            metric_names=metric_names,
            use_model_predictions=self.use_model_predictions,
            adapter=relevant_adapter,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            relativize=self.relativize,
        )

        # Retrieve the metric labels from the mapping provided by the user, defaulting
        # to the metric name if no label is provided, truncated.
        metric_labels = {
            metric_name: self.labels.get(metric_name, truncate_label(label=metric_name))
            for metric_name in metric_names
        }

        cards = [
            create_plotly_analysis_card(
                name=self.__class__.__name__,
                title=(
                    f"{'Modeled' if self.use_model_predictions else 'Observed'} "
                    f"{'Relativized ' if self.relativize else ''}Arm "
                    f"Effects on {metric_labels[metric_name]}"
                    + (
                        f" for trial {self.trial_index}"
                        if self.trial_index is not None
                        else ""
                    )
                ),
                subtitle=_get_subtitle(
                    metric_label=metric_labels[metric_name],
                    use_model_predictions=self.use_model_predictions,
                    trial_index=self.trial_index,
                ),
                df=df[
                    [
                        "trial_index",
                        "trial_status",
                        "arm_name",
                        "generation_node",
                        "p_feasible_mean",
                        "p_feasible_sem",
                    ]
                    + (
                        [
                            f"{metric_name}_mean",
                            f"{metric_name}_sem",
                        ]
                        if metric_name != "p_feasible"
                        else []
                    )
                ].copy(),
                fig=_prepare_figure(
                    df=df,
                    metric_name=metric_name,
                    is_relative=self.relativize,
                    status_quo_arm_name=experiment.status_quo.name
                    if experiment.status_quo
                    else None,
                    metric_label=metric_labels[metric_name],
                    show_cumulative_best=self.show_cumulative_best,
                    lower_is_better=get_lower_is_better(
                        experiment=experiment, metric_name=metric_name
                    )
                    or False,
                ),
            )
            for metric_name in metric_names_to_plot
        ]

        return self._create_analysis_card_group_or_card(
            title=CARDGROUP_TITLE,
            subtitle=PREDICTED_EFFECTS_CARDGROUP_SUBTITLE
            if self.use_model_predictions
            else RAW_EFFECTS_CARDGROUP_SUBTITLE,
            children=cards,
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
    show_cumulative_best: bool = False,
) -> AnalysisCardBase:
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
        show_cumulative_best: Whether to draw a line through the best point seen so
            far during the optimization.
    """

    analysis = ArmEffectsPlot(
        metric_names=metric_names,
        use_model_predictions=use_model_predictions,
        relativize=relativize,
        trial_index=trial_index,
        trial_statuses=trial_statuses,
        additional_arms=additional_arms,
        labels=labels,
        show_cumulative_best=show_cumulative_best,
    )

    return analysis.compute(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )


def _prepare_figure(
    df: pd.DataFrame,
    metric_name: str,
    is_relative: bool,
    status_quo_arm_name: str | None,
    metric_label: str,
    show_cumulative_best: bool,
    lower_is_better: bool,
) -> go.Figure:
    # Prepare separate scatter traces for each trial index. Each trace has (x, y)
    # points for the arms which we have a mean for in the provided dataframe.
    candidate_trial = df[df["trial_status"] == TrialStatus.CANDIDATE.name][
        "trial_index"
    ].max()
    # Filter out undesired trials like FAILED and ABANDONED trials from plot.
    trials = df[
        ~df["trial_status"].isin(
            [ts.name for ts in FAILED_ABANDONED_CANDIDATE_STATUSES]
        )
    ]["trial_index"].unique()

    # Check if candidate_trial is NaN and handle it
    trial_indices = list(trials)
    if not np.isnan(candidate_trial):
        trial_indices.append(candidate_trial)

    trials_list = trials.tolist()
    scatters = []
    # Sort the dataframe by trial index and arm name.
    # Non-default arm names (not digit + underscore + digit) are sorted to the front.
    # default arm names (e.g. 0_0, 1_8) are sorted that '0_1' < '0_5' < '0_10'
    df = sort_by_trial_index_and_arm_name(df=df)
    # Add a column combining trial_index and arm_name to be used as x mark
    df["x_key_order"] = df["trial_index"].astype(str) + ":" + df["arm_name"]
    arm_order = []
    arm_label = []
    for trial_index in trial_indices:
        trial_df = df[df["trial_index"] == trial_index]
        xy_df = trial_df[~trial_df[f"{metric_name}_mean"].isna()]
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
            "line": {
                "width": trial_df.apply(
                    lambda row: 2
                    if row["p_feasible_mean"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD
                    else 0,
                    axis=1,
                ),
                "color": CONSTRAINT_VIOLATION_RED,
            },
        }
        text = trial_df.apply(
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
                hoverinfo="text",
                text=text,
            )
        )

    # get the max length of x-ticker (arm name) to set the xaxis label and
    # legend position
    # This assumes the x-tickers are rotated 90 degrees (vertical) so legend
    # will be always below the x-label ('Arm Name').
    max_label_len = max(len(label) for label in arm_label)
    legend_y = LEGEND_BASE_OFFSET - (max_label_len / X_TICKER_SCALING_FACTOR)

    legend_position = LEGEND_POSITION.copy()
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

    # Add a red circle with no fill if any arms are marked as possibly infeasible.
    if (df["p_feasible_mean"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD).any():
        legend_trace = go.Scatter(
            # None here allows us to place a legend item without corresponding points
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": "rgba(0, 0, 0, 0)",
                "line": {"width": 2, "color": "red"},
            },
            name="Possible Constraint Violation",
        )

        figure.add_trace(legend_trace)

    if show_cumulative_best:
        # Sort and filter arms which are eligable to the best point (i.e. that are not
        # likely to violate constraints).
        zipped = [
            *zip(
                *[
                    (
                        arm_name,
                        df[df["arm_name"] == arm_name][f"{metric_name}_mean"].iloc[0],
                    )
                    for arm_name in arm_label
                    if df[df["arm_name"] == arm_name]["p_feasible_mean"].iloc[0]
                    >= POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD
                ]
            )
        ]

        # If there are no arms which are not likely to violate constraints, return the
        # figure as is, without adding a best point line.
        if len(zipped) != 2:
            return figure

        sorted_arms, sorted_y = zipped

        best_y = (
            np.minimum.accumulate(sorted_y)
            if lower_is_better
            else np.maximum.accumulate(sorted_y)
        )

        # Add a line for the best point seen so far.
        figure.add_trace(go.Scatter(x=sorted_arms, y=best_y, **BEST_LINE_SETTINGS))

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
