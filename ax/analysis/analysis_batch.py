# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.base_analysis import BaseAnalysis
from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.core.experiment import Experiment

from ax.utils.common.timeutils import current_timestamp_in_millis


class AnalysisBatch:
    """
    A class corresponding to a set of analysis ran on the same
    set of data from an experiment.
    """

    analyses: List[Union[BaseAnalysis, BasePlotlyVisualization]] = []
    experiment: Experiment
    time_started: int

    analysis_output: Optional[
        List[
            Tuple[
                Union[BaseAnalysis, BasePlotlyVisualization],
                pd.DataFrame,
                Optional[go.Figure],
            ]
        ]
    ] = None

    time_started: Optional[int] = None
    time_completed: Optional[int] = None
    db_id: Optional[int] = None

    def __init__(self, experiment: Experiment) -> None:
        """
        Args:
            experiment: Experiment which the analyses are generated from
        """
        self.experiment = experiment

    def add_analysis(self, analysis: BaseAnalysis) -> None:
        if self.analysis_output is not None:
            raise ValueError(
                "Analysis batch has already been run. Cannot add more analyses."
            )

        self.analyses.append(analysis)

    def run_analysis_batch(
        self,
    ) -> List[
        Tuple[
            Union[BaseAnalysis, BasePlotlyVisualization],
            pd.DataFrame,
            Optional[go.Figure],
        ]
    ]:
        """
        Runs all analyses in the batch and returns a list of tuples
        of the form (analysis, df, fig)
        """
        if self.analysis_output is not None:
            return self.analysis_output

        self.time_started = current_timestamp_in_millis()

        output = []
        for analysis in self.analyses:
            output.append(
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
        self.analysis_output = output

        self.time_completed = current_timestamp_in_millis()

        return output
