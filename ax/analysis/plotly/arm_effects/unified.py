# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import pandas as pd

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import get_arm_tooltip, trial_status_to_plotly_color
from ax.analysis.utils import extract_relevant_adapter, prepare_arm_data
from ax.core.arm import Arm
from ax.core.experiment import Experiment
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
    FAILED). Each arm also has a hover tooltip with additional information.

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
        metric_names: Sequence[str] | None = None,
        use_model_predictions: bool = True,
        trial_index: int | None = None,
        additional_arms: Sequence[Arm] | None = None,
    ) -> None:
        self.metric_names = metric_names
        self.use_model_predictions = use_model_predictions
        self.trial_index = trial_index
        self.additional_arms = additional_arms

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
            additional_arms=self.additional_arms,
        )

        return [
            self._create_plotly_analysis_card(
                title=(
                    f"{'Modeled' if self.use_model_predictions else 'Observed'} Arm "
                    f"Effects on {metric_name}"
                ),
                subtitle=_get_subtitle(
                    metric_name=metric_name,
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
                        f"{metric_name}_mean",
                        f"{metric_name}_sem",
                    ]
                ].copy(),
                fig=_prepare_figure(df=df, metric_name=metric_name),
                category=AnalysisCardCategory.INSIGHT,
            )
            for metric_name in metric_names
        ]


def _prepare_figure(
    df: pd.DataFrame,
    metric_name: str,
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
                )
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
    figure.update_layout(xaxis_title="Arm Name", yaxis_title=metric_name)

    # Order arms by trial index, then by arm name. Always put additional arms last.
    arm_order = df.sort_values(by=["trial_index", "arm_name"])["arm_name"].tolist()

    additional_arm_names = df[df["trial_index"] == -1]["arm_name"].tolist()
    arm_order = [
        *[arm_name for arm_name in arm_order if arm_name == "status_quo"],
        *[
            arm_name
            for arm_name in arm_order
            if arm_name not in additional_arm_names and arm_name != "status_quo"
        ],
        *[
            arm_name
            for arm_name in arm_order
            if arm_name in additional_arm_names and arm_name != "status_quo"
        ],
    ]

    figure.update_xaxes(categoryorder="array", categoryarray=arm_order)

    # Add a horizontal line for the status quo.
    if "status_quo" in df["arm_name"].values:
        y = df[df["arm_name"] == "status_quo"][f"{metric_name}_mean"].iloc[0]

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
    metric_name: str,
    use_model_predictions: bool,
    trial_index: int | None = None,
) -> str:
    first_clause = (
        f"{'Modeled' if use_model_predictions else 'Observed'} effects on {metric_name}"
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
