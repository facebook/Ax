# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd

import plotly.graph_objects as go

from ax.analysis.old.base_analysis import BaseAnalysis
from ax.core.experiment import Experiment


class BasePlotlyVisualization(BaseAnalysis):
    """
    Abstract PlotlyVisualization class for ax.
    This is an interface that defines the method to be implemented by all ax plots.
    Computes an output dataframe for each analysis
    """

    def __init__(
        self,
        experiment: Experiment,
        df_input: Optional[pd.DataFrame] = None,
        fig_input: Optional[go.Figure] = None,
    ) -> None:
        """
        Initialize the analysis with the experiment object.
        For scenarios where an analysis output is already available,
        we can pass the dataframe as an input.
        """
        self._fig = fig_input
        super().__init__(experiment=experiment, df_input=df_input)

    @property
    def fig(self) -> go.Figure:
        """
        Return the output of the analysis of this class.
        """
        if self._fig is None:
            self._fig = self.get_fig()
        return self._fig

    def get_fig(self) -> go.Figure:
        """
        Return the plotly figure of the analysis of this class.
        Subclasses should overwrite this.
        """
        raise NotImplementedError("get_fig must be implemented by subclass")
