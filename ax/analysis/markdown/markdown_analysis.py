# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import markdown

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardBase
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from IPython.display import Markdown
from pyre_extensions import override


class MarkdownAnalysisCard(AnalysisCard):
    def get_markdown(self) -> str:
        return self.blob

    def _body_html(self) -> str:
        return f"<div class='content'>{markdown.markdown(self.get_markdown())}<div>"

    def _body_papermill(self) -> Markdown:
        return Markdown(self.get_markdown())


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
    ) -> AnalysisCardBase: ...

    def _create_markdown_analysis_card(
        self,
        title: str,
        subtitle: str,
        df: pd.DataFrame,
        message: str,
    ) -> MarkdownAnalysisCard:
        """
        Make a MarkdownAnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return MarkdownAnalysisCard(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            blob=message,
        )
