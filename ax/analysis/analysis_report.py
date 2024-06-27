# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.base_analysis import BaseAnalysis
from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.core.experiment import Experiment

from ax.utils.common.timeutils import current_timestamp_in_millis


class AnalysisReport:
    """
    A class corresponding to a set of analysis ran on the same
    set of data from an experiment.
    """

    analyses: List[BaseAnalysis] = []
    experiment: Experiment

    report_completed: bool = False
    time_started: Optional[int] = None
    time_completed: Optional[int] = None

    def __init__(
        self,
        experiment: Experiment,
        analyses: List[BaseAnalysis],
        report_completed: bool = False,
        time_started: Optional[int] = None,
        time_completed: Optional[int] = None,
    ) -> None:
        """
        This class is a collection of AnalysisReport.
        This class takes two states, indicated by "report_completed"
        If report_completed is True, then the report has been run and
            the time_started and time_completed are set.
            "run_analysis_report" will not update the time_started
            and time_completed fields, but will stil run the individual
            analyses and return the outputs.
        If report_completed is False, then the report has
            not been run and the time_started and time_completed are None.
            "run_analysis_report" will set the time_started and time_completed.

        When loading an analysis report from the database, the report has already ran,
            so will be loaded back with "report_completed" = True and
            time_started and time_completed set.

        Args:
            experiment: Experiment which the analyses are generated from
            report_completed: Whether the report was loaded from the database
            time_started: time the completed report was started

        """
        self.experiment = experiment
        self.analyses = analyses
        if report_completed:
            self.time_started = time_started
            self.time_completed = time_completed
            self.report_completed = True

    def run_analysis_report(
        self,
    ) -> List[
        Tuple[
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
            self.report_completed = True

        return analysis_report_result
