# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Sequence

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from IPython.display import display
from plotly import graph_objects as go, io as pio
from pyre_extensions import override


class PlotlyAnalysisCard(AnalysisCard):
    blob_annotation = "plotly"

    def get_figure(self) -> go.Figure:
        return pio.from_json(self.blob)

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is printed in an
        IPython environment (ex. Jupyter). Here we want to display the Plotly figure.
        """
        self._display_header()
        display(self.get_figure())


class PlotlyAnalysis(Analysis):
    """
    An Analysis that computes a Plotly figure.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]: ...

    def _create_plotly_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        fig: go.Figure,
        category: int,
    ) -> PlotlyAnalysisCard:
        """
        Make a PlotlyAnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return PlotlyAnalysisCard(
            name=self.name,
            attributes=self.attributes,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=pio.to_json(fig),
            category=category,
        )
