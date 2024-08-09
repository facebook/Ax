# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.old.base_analysis import BaseAnalysis
from ax.analysis.old.base_plotly_visualization import BasePlotlyVisualization

from ax.core.experiment import Experiment

from ax.utils.common.timeutils import current_timestamp_in_millis


class AnalysisReport:
    """
    A class corresponding to a set of analysis ran on the same
    set of data from an experiment.
    """

    analyses: list[BaseAnalysis] = []
    experiment: Experiment

    time_started: Optional[int] = None
    time_completed: Optional[int] = None

    def __init__(
        self,
        experiment: Experiment,
        analyses: list[BaseAnalysis],
        time_started: Optional[int] = None,
        time_completed: Optional[int] = None,
    ) -> None:
        """
        This class is a collection of AnalysisReport.

        Args:
            experiment: Experiment which the analyses are generated from
            time_started: time the completed report was started
            time_completed: time the completed report was started

        """
        self.experiment = experiment
        self.analyses = analyses
        self.time_started = time_started
        self.time_completed = time_completed

    @property
    def report_completed(self) -> bool:
        """
        Returns:
            True if the report is completed, False otherwise.
        """
        return self.time_completed is not None

    def run_analysis_report(
        self,
    ) -> list[
        tuple[
            BaseAnalysis,
            pd.DataFrame,
            Optional[go.Figure],
        ]
    ]:
        """
        Runs all analyses in the report and produces the result.

        Returns:
            analysis_report_result: list of tuples (analysis, df, Optional[fig])
        """
        if not self.report_completed:
            self.time_started = current_timestamp_in_millis()

        analysis_report_result = []
        for analysis in self.analyses:
            analysis_report_result.append(
                (
                    analysis,
                    analysis.get_df(),
                    (
                        None
                        if not isinstance(analysis, BasePlotlyVisualization)
                        else analysis.get_fig()
                    ),
                )
            )

        if not self.report_completed:
            self.time_completed = current_timestamp_in_millis()

        return analysis_report_result
