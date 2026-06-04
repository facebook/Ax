# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger
from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.color_constants import LIGHT_AX_BLUE
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card
from ax.analysis.plotly.utils import truncate_label
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_adapter_can_predict,
    validate_experiment,
    validate_experiment_has_trials,
)
from ax.core.analysis_card import AnalysisCard
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from plotly import graph_objects as go
from pyre_extensions import none_throws, override

logger: Logger = get_logger(__name__)

Z_SCORE_95_CI: float = 1.96


@final
class UtilityRankingPlot(Analysis):
    """Ranked bar chart of model-predicted latent utility per arm for preference
    metrics (e.g., pairwise_pref_query backed by PairwiseGP).

    Unlike ArmEffectsPlot, this analysis does NOT show "observed" effects because
    the raw observations for preference metrics are binary comparison outcomes
    (0/1), which are meaningless as per-arm summaries.  Instead it shows the
    model's predicted latent utility for each arm, ranked from most preferred
    (top) to least preferred (bottom).

    The DataFrame produced contains one row per arm with columns:
        - trial_index, arm_name, trial_status, generation_node
        - {metric_name}_mean, {metric_name}_sem  (latent utility predictions)
    """

    def __init__(
        self,
        metric_name: str,
        trial_statuses: Sequence[TrialStatus] | None = None,
        show_ci: bool = False,
    ) -> None:
        """
        Args:
            metric_name: The preference metric to rank arms by.
            trial_statuses: If present, only include arms from trials with
                these statuses.
            show_ci: Whether to show 95% CI error bars. Defaults to False
                because PairwiseGP's latent utility posterior variance is
                typically very large (the utility scale is arbitrary).
        """
        self.metric_name = metric_name
        self.trial_statuses = trial_statuses
        self.show_ci = show_ci

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        if (
            experiment_invalid := validate_experiment(
                experiment=experiment,
                require_trials=True,
                require_data=True,
            )
        ) is not None:
            return experiment_invalid

        experiment = none_throws(experiment)

        if (
            no_trials := validate_experiment_has_trials(
                experiment=experiment,
                trial_indices=None,
                trial_statuses=list(self.trial_statuses)
                if self.trial_statuses is not None
                else None,
                required_metric_names=[self.metric_name],
            )
        ) is not None:
            return no_trials

        if (
            adapter_err := validate_adapter_can_predict(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                required_metric_names=[self.metric_name],
            )
        ) is not None:
            return adapter_err

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        experiment = none_throws(experiment)

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        df = prepare_arm_data(
            experiment=experiment,
            metric_names=[self.metric_name],
            use_model_predictions=True,
            adapter=relevant_adapter,
            trial_statuses=list(self.trial_statuses)
            if self.trial_statuses is not None
            else None,
        )

        # Deduplicate by arm_name. In LILO experiments, the same arm appears
        # in multiple labeling trials. Model predictions are identical for a
        # given arm regardless of trial, so keep only the first occurrence.
        df = df.drop_duplicates(subset=["arm_name"], keep="first")

        mean_col = f"{self.metric_name}_mean"

        # Offset by status quo utility when available, so the plot shows
        # relative preference (how much better/worse than status quo).
        sq_name = (
            experiment.status_quo.name if experiment.status_quo is not None else None
        )
        offset_by_sq = False
        if sq_name is not None and sq_name in df["arm_name"].values:
            sq_utility = df.loc[df["arm_name"] == sq_name, mean_col].iloc[0]
            # Guard against NaN SQ utility (e.g., when the SQ arm has
            # parameters outside the model space and is unpredictable).
            # Subtracting NaN would poison all arm predictions.
            if pd.notna(sq_utility):
                df[mean_col] = df[mean_col] - sq_utility
                offset_by_sq = True
            else:
                logger.warning(
                    f"Status quo arm '{sq_name}' has no model prediction "
                    f"(NaN utility) for metric '{self.metric_name}'. This "
                    "typically happens when the status quo arm has parameters "
                    "outside the model's search space (e.g., None values for "
                    "conditional parameters). Showing absolute utility values "
                    "instead of status-quo-relative values."
                )

        metric_label = truncate_label(label=self.metric_name)

        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title=f"Utility Ranking for {metric_label}",
            subtitle=(
                "Ranked model-predicted latent utility per arm for the preference "
                f"metric '{metric_label}'. Arms are sorted from most preferred "
                "(top) to least preferred (bottom)."
                + (
                    " Values are relative to the status quo arm."
                    if offset_by_sq
                    else ""
                )
            ),
            df=df[
                [
                    "trial_index",
                    "trial_status",
                    "arm_name",
                    "generation_node",
                    mean_col,
                    f"{self.metric_name}_sem",
                ]
            ].copy(),
            fig=_prepare_figure(
                df=df,
                metric_name=self.metric_name,
                show_ci=self.show_ci,
                offset_by_sq=offset_by_sq,
            ),
        )


def _prepare_figure(
    df: pd.DataFrame,
    metric_name: str,
    show_ci: bool = False,
    offset_by_sq: bool = False,
) -> go.Figure:
    """Build a horizontal dot plot of latent utility, ranked descending.

    Uses go.Scatter (dot plot with optional error bars) to match the visual
    style of ArmEffectsPlot, which also uses scatter for per-arm effects.
    Candidate trials are shown in a lighter color to distinguish them from
    completed trials (consistent with ArmEffectsPlot's convention).
    """
    mean_col = f"{metric_name}_mean"
    sem_col = f"{metric_name}_sem"

    # Drop rows with missing predictions.
    plot_df = df.dropna(subset=[mean_col]).copy()

    # Sort ascending so highest utility is at the top.
    plot_df = plot_df.sort_values(by=mean_col, ascending=True)

    is_candidate = plot_df["trial_status"] == TrialStatus.CANDIDATE.name
    completed_df = plot_df[~is_candidate]
    candidate_df = plot_df[is_candidate]

    fig = go.Figure()

    # Add completed arms trace.
    for subset_df, color, label in [
        (completed_df, "#1f77b4", "Completed"),
        (candidate_df, LIGHT_AX_BLUE, "Candidate"),
    ]:
        if subset_df.empty:
            continue

        error_x = None
        if show_ci:
            sem_series: pd.Series = subset_df[sem_col]
            if not sem_series.isna().all():
                error_x = {
                    "type": "data",
                    "array": (Z_SCORE_95_CI * sem_series).values.tolist(),
                    "visible": True,
                }

        hover_text = subset_df.apply(
            lambda row: (
                f"Arm: {row['arm_name']}<br>"
                f"Trial: {row['trial_index']}"
                f" ({row['trial_status']})<br>"
                f"Utility: {row[mean_col]:.4f}"
                + (
                    f" ± {Z_SCORE_95_CI * row[sem_col]:.4f}"
                    if show_ci and pd.notna(row[sem_col])
                    else ""
                )
            ),
            axis=1,
        )

        fig.add_trace(
            go.Scatter(
                x=subset_df[mean_col].tolist(),
                y=subset_df["arm_name"].tolist(),
                mode="markers",
                marker={"size": 8, "color": color},
                error_x=error_x,
                hoverinfo="text",
                text=hover_text,
                name=label,
                legendgroup=label,
            )
        )

    # Add a vertical reference line at 0 when offset by SQ.
    if offset_by_sq:
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

    x_title = (
        "Predicted Utility (relative to status quo)"
        if offset_by_sq
        else "Predicted Latent Utility"
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Arm",
        yaxis={"type": "category"},
        margin={"l": 120, "r": 40, "t": 40, "b": 60},
    )

    return fig
