# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import traceback

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel, AnalysisE
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
        display(Markdown(f"## {self.title}\n\n### {self.subtitle}\n\n{self.blob}"))


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
            name=self.name,
            attributes=self.attributes,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=message,
        )


def markdown_analysis_card_from_analysis_e(
    analysis_e: AnalysisE,
) -> MarkdownAnalysisCard:
    return MarkdownAnalysisCard(
        name=analysis_e.analysis.name,
        title=f"{analysis_e.analysis.name} Error",
        subtitle=f"An error occurred while computing {analysis_e.analysis}",
        attributes=analysis_e.analysis.attributes,
        blob="".join(
            traceback.format_exception(
                type(analysis_e.exception),
                analysis_e.exception,
                analysis_e.exception.__traceback__,
            )
        ),
        df=pd.DataFrame(),
        level=AnalysisCardLevel.DEBUG,
    )
