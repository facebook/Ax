# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import abstractmethod

import plotly.graph_objects as go

from ax.analysis.base_analysis import BaseAnalysis


class BasePlotlyVisualization(BaseAnalysis):
    """
    Abstract PlotlyVisualization class for ax.
    This is an interface that defines the method to be implemented by all ax plots.
    Computes an output dataframe for each analysis
    """

    @abstractmethod
    def get_fig(self) -> go.Figure:
        """
        Return the plotly figure of the analysis of this class.
        """
