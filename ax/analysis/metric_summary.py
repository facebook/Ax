# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Sequence

from ax.analysis.analysis import (
    Analysis,
    AnalysisCard,
    AnalysisCardCategory,
    AnalysisCardLevel,
)
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from pyre_extensions import override


class MetricSummary(Analysis):
    """
    Creates a dataframe with information about each metric in the
    experiment. The resulting dataframe has one row per metric, and the
    following columns:
        - Name: the name of the metric.
        - Type: the metric subclass (e.g., Metric, BraninMetric).
        - Goal: the goal for this for this metric, based on the optimization
            config (minimize, maximize, constraint or track).
        - Bound: the bound of this metric (e.g., "<=10.0") if it is being used
            as part of an ObjectiveThreshold or OutcomeConstraint.
        - Lower is Better: whether the user prefers this metric to be lower,
            if provided.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[AnalysisCard]:
        if experiment is None:
            raise UserInputError(
                "`MetricSummary` analysis requires an `Experiment` input"
            )
        return [
            self._create_analysis_card(
                title=f"MetricSummary for `{experiment.name}`",
                subtitle="High-level summary of the `Metric`-s in this `Experiment`",
                level=AnalysisCardLevel.MID,
                df=experiment.metric_config_summary_df,
                category=AnalysisCardCategory.INFO,
            )
        ]
