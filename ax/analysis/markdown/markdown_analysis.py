# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import traceback
from typing import Sequence

import pandas as pd
from ax.analysis.analysis import (
    Analysis,
    AnalysisCard,
    AnalysisCardCategory,
    AnalysisCardLevel,
    AnalysisE,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from IPython.display import display, Markdown
from pyre_extensions import override


class MarkdownAnalysisCard(AnalysisCard):
    blob_annotation = "markdown"

    def get_markdown(self) -> str:
        return self.blob

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is printed in an
        IPython environment (ex. Jupyter). Here we want to render the Markdown.
        """
        self._display_header()
        display(Markdown(self.blob))


class MarkdownAnalysis(Analysis):
    """
    An Analysis that computes a paragraph of Markdown formatted text.
    """

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[MarkdownAnalysisCard]: ...

    def _create_markdown_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        message: str,
        category: int,
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
            category=category,
        )


def markdown_analysis_card_from_analysis_e(
    analysis_e: AnalysisE,
) -> list[MarkdownAnalysisCard]:
    return [
        MarkdownAnalysisCard(
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
            category=AnalysisCardCategory.ERROR,
        )
    ]
