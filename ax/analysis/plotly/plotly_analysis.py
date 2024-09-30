# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

from ax.analysis.analysis import Analysis, AnalysisCard
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from IPython.display import display
from plotly import graph_objects as go, io as pio


class PlotlyAnalysisCard(AnalysisCard):
    blob_annotation = "plotly"

    def get_figure(self) -> go.Figure:
        return pio.from_json(self.blob)

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is printed in an
        IPython environment (ex. Jupyter). Here we want to display the Plotly figure.
        """
        display(self.get_figure())


class PlotlyAnalysis(Analysis):
    """
    An Analysis that computes a Plotly figure.
    """

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategyInterface] = None,
    ) -> PlotlyAnalysisCard: ...
