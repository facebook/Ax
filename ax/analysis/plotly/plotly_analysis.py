# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from plotly import graph_objects as go


class PlotlyAnalysisCard(AnalysisCard):
    name: str

    title: str
    subtitle: str
    level: AnalysisCardLevel

    df: pd.DataFrame
    blob: go.Figure
    blob_annotation = "plotly"

    def get_figure(self) -> go.Figure:
        return self.blob


class PlotlyAnalysis(Analysis):
    """
    An Analysis that computes a Plotly figure.
    """

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: Optional[GenerationStrategy] = None,
    ) -> PlotlyAnalysisCard: ...
