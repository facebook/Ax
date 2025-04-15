# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping, Sequence

import pandas as pd

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import get_arm_tooltip, trial_status_to_plotly_color
from ax.analysis.utils import (
    extract_relevant_adapter,
    POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD,
    prepare_arm_data,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from plotly import graph_objects as go
from pyre_extensions import override


class ArmEffectsPlot(PlotlyAnalysis):
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
        """

        self.metric_names = metric_names
        self.use_model_predictions = use_model_predictions
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self.additional_arms = additional_arms
        self.labels = labels or {}

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
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
        # to the metric name if no label is provided.
        metric_labels = {
            metric_name: self.labels.get(metric_name, metric_name)
            for metric_name in metric_names
        }

        return [
            self._create_plotly_analysis_card(
                title=(
                    f"{'Modeled' if self.use_model_predictions else 'Observed'} Arm "
                    f"Effects on {metric_labels[metric_name]}"
                ),
                subtitle=_get_subtitle(
                    metric_label=metric_labels[metric_name],
                    use_model_predictions=self.use_model_predictions,
                    trial_index=self.trial_index,
                ),
                level=AnalysisCardLevel.MID,
                df=df[
                    [
                        "trial_index",
                        "trial_status",
                        "arm_name",
                        "generation_node",
                        "p_feasible",
                        f"{metric_name}_mean",
                        f"{metric_name}_sem",
                    ]
                ].copy(),
                fig=_prepare_figure(
                    df=df,
                    metric_name=metric_name,
                    is_relative=self.relativize,
                    metric_label=metric_labels[metric_name],
                    status_quo_arm_name=experiment.status_quo.name
                    if experiment.status_quo
                    else None,
                ),
                category=AnalysisCardCategory.INSIGHT,
            )
            for metric_name in metric_names
        ]


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
) -> list[PlotlyAnalysisCard]:
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

    analysis = ArmEffectsPlot(
        metric_names=metric_names,
        use_model_predictions=use_model_predictions,
        relativize=relativize,
        trial_index=trial_index,
        trial_statuses=trial_statuses,
        additional_arms=additional_arms,
        labels=labels,
    )

    return [
        *analysis.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
    ]


def _prepare_figure(
    df: pd.DataFrame,
    metric_name: str,
    is_relative: bool,
    metric_label: str,
    status_quo_arm_name: str | None,
) -> go.Figure:
    scatters = [
        go.Scatter(
            x=df[
                (df["trial_status"] == trial_status) & ~df[f"{metric_name}_mean"].isna()
            ]["arm_name"],
            y=df[df["trial_status"] == trial_status][f"{metric_name}_mean"],
            error_y=(
                {
                    "type": "data",
                    "array": df[df["trial_status"] == trial_status][
                        f"{metric_name}_sem"
                    ]
                    * 1.96,
                    "color": trial_status_to_plotly_color(
                        trial_status=trial_status, ci_transparency=True
                    ),
                }
                if not df[f"{metric_name}_sem"].isna().all()
                else None
            ),
            mode="markers",
            marker={
                "color": trial_status_to_plotly_color(
                    trial_status=trial_status, ci_transparency=False
                ),
                "line": {
                    "width": df[df["trial_status"] == trial_status].apply(
                        lambda row: 2
                        if row["p_feasible"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD
                        else 0,
                        axis=1,
                    ),
                    "color": "red",
                },
            },
            # Apply user-friendly name for UNKNOWN_GENERATION_NODE
            name=trial_status,
            hoverinfo="text",
            text=df[df["trial_status"] == trial_status].apply(
                lambda row: get_arm_tooltip(row=row, metric_names=[metric_name]), axis=1
            ),
        )
        for trial_status in df["trial_status"].unique()
    ]

    figure = go.Figure(data=scatters)
    figure.update_layout(
        xaxis_title="Arm Name",
        yaxis_title=metric_label,
        yaxis_tickformat=".2%" if is_relative else None,
    )

    # Order arms by trial index, then by arm name. Always put additional arms last.
    arm_order = df.sort_values(by=["trial_index", "arm_name"])["arm_name"].tolist()

    additional_arm_names = df[df["trial_index"] == -1]["arm_name"].tolist()

    arm_order = [
        *[arm_name for arm_name in arm_order if arm_name == status_quo_arm_name],
        *[
            arm_name
            for arm_name in arm_order
            if arm_name not in additional_arm_names and arm_name != status_quo_arm_name
        ],
        *[
            arm_name
            for arm_name in arm_order
            if arm_name in additional_arm_names and arm_name != status_quo_arm_name
        ],
    ]
    figure.update_xaxes(categoryorder="array", categoryarray=arm_order)

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
    if (df["p_feasible"] < POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD).any():
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
