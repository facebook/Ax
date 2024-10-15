# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from IPython.display import display, Markdown


class MarkdownAnalysisCard(AnalysisCard):
    blob_annotation = "markdown"

    def get_markdown(self) -> str:
        return self.blob

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is printed in an
        IPython environment (ex. Jupyter). Here we want to render the Markdown.
        """
        display(Markdown(self.blob))


class MarkdownAnalysis(Analysis):
    """
    An Analysis that computes a paragraph of Markdown formatted text.
    """

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> MarkdownAnalysisCard: ...

    def _create_markdown_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        message: str,
    ) -> MarkdownAnalysisCard:
        """
        Make a MarkdownAnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return MarkdownAnalysisCard(
            name=self.__class__.__name__,
            attributes=self.__dict__,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=message,
        )
