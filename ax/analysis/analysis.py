# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import Any, Protocol, Sequence

import pandas as pd
from ax.adapter.base import Adapter
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, ExceptionE, Ok, Result
from ax.utils.tutorials.environment import is_running_in_papermill
from IPython.display import display, HTML, Markdown

logger: Logger = get_logger(__name__)

# Simple HTML template for rendering a card with a title, subtitle, and body with
# scrollable overflow.
html_card_template = """
<style>
.card {{
    overflow: auto;
}}
</style>
<div class="card">
    <div class="card-header">
        <b>
        {title_str}
        </b>
        <p>
        {subtitle_str}
        </p>
    </div>
    <div class="card-body">
        {body_html}
    </div>
</div>
"""

# HTML template for putting cards into a 2 x N CSS grid.
html_grid_template = """
<style>
    .grid-container {{
        display: grid;
        grid-template-columns: repeat(2, 2fr);
        gap: 10px;
    }}
</style>
<div class="grid-container">
    {card_divs}
</div>
"""


class AnalysisCardBase(SortableBase, ABC):
    """
    Abstract base class for "cards", the result of a call to Analyis.compute(...).
    Cards may either be a single card (AnalysisCard and its subclasses) or an ordered
    collection of cards (AnalysisCardGroup) -- together these three classes form a
    tree structure which can hold arbitrarily nested collections of cards.

    When rendering in an IPython environment (ex. Jupyter), use AnalyisCardBase.flatten
    to produce an ordered list of cards to render.

    Args:
        name: The class name of the Analysis that produced this card (ex. "Summary",
        "ArmEffects", etc.).
    """

    name: str
    # Timestamp is especially useful when querying the database for the most recently
    # produced artifacts.
    _timestamp: datetime

    def __init__(self, name: str, timestamp: datetime | None = None) -> None:
        """
        Args:
            name: The class name of the Analysis that produced this card (ex.
            "Summary", "ArmEffects", etc.).
        """
        self.name = name
        self._timestamp = timestamp if timestamp is not None else datetime.now()

    @abstractmethod
    def flatten(self) -> list[AnalysisCard]:
        """
        Returns a list of AnalysisCards contained in this card in order. This is useful
        when processing a collection of cards where order is necessary but grouping can
        safely be ignored (ex. when rendering a collection of cards in an IPython).
        """
        pass


class AnalysisCardGroup(AnalysisCardBase):
    """
    An ordered collection of AnalysisCards. This is useful for grouping related
    analyses together.

    This is analogous to a "branch node" in a tree structure.

    Args:
        name: The name of the Analysis that produced this card.
    """

    children: list[AnalysisCardBase]

    def __init__(
        self,
        name: str,
        children: Sequence[AnalysisCardBase],
        timestamp: datetime | None = None,
    ) -> None:
        super().__init__(name=name, timestamp=timestamp)
        self.children = [*children]

    @property
    def _unique_id(self) -> str:
        return self.name

    def flatten(self) -> list[AnalysisCard]:
        return [child for child in self.children for child in child.flatten()]

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is rendered in an
        IPython environment (ex. Jupyter). This method should not be implemented by
        subclasses; instead they should implement the representation-specific helpers
        such as _body_html_ and _body_papermill_.
        """

        if is_running_in_papermill():
            for card in self.flatten():
                display(Markdown(f"**{card.title}**\n\n{card.subtitle}"))
                display(card._body_papermill())
                return

        display(
            HTML(
                html_grid_template.format(
                    card_divs="".join([card._repr_html_() for card in self.flatten()])
                )
            )
        )


class AnalysisCard(AnalysisCardBase):
    """
    The ultimate result of a call to Analysis.compute(...). This holds the raw data
    produced by the compute function as a dataframe, and some arbitrary blob of data
    which will be rendered in the card in a notebook or a UI front-end (e.g. a Plotly
    figure, Markdown formatted text etc.)

    Subclasses of AnalysisCard define the structure of the blob (ex. a Plotly Figure)
    and implement methods for rendering the card in a useful way.

    This is analogous to a "leaf node" in a tree structure.
    """

    title: str
    subtitle: str

    df: pd.DataFrame  # Raw data produced by the Analysis

    # Blob is the data processed for end-user consumption, encoded as a string,
    # typically JSON. Subclasses of Analysis can define their own methods for consuming
    # the blob and presenting it to the user (ex. PlotlyAnalysisCard.get_figure()
    # decodes the blob into a go.Figure object).
    blob: str

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str,
        df: pd.DataFrame,
        blob: str,
        timestamp: datetime | None = None,
    ) -> None:
        super().__init__(name=name, timestamp=timestamp)

        self.title = title
        self.subtitle = subtitle
        self.df = df
        self.blob = blob

    @property
    def _unique_id(self) -> str:
        return self.title

    def flatten(self) -> list[AnalysisCard]:
        return [self]

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is rendered in an
        IPython environment (ex. Jupyter). This method should not be implemented by
        subclasses; instead they should implement the representation-specific helpers
        such as _body_html_ and _body_papermill_.
        """

        if is_running_in_papermill():
            display(Markdown(f"**{self.title}**\n\n{self.subtitle}"))
            display(self._body_papermill())
            return

        display(HTML(self._repr_html_()))

    def _repr_html_(self) -> str:
        """
        IPython HTML representation hook. This is called when the AnalysisCard is
        rendered in an IPython environment (ex. Jupyter). This method should be
        implemented by subclasses of Analysis to display the AnalysisCard in a useful
        way.
        """

        return html_card_template.format(
            title_str=self.title,
            subtitle_str=self.subtitle,
            body_html=self._body_html(),
        )

    def _body_html(self) -> str:
        """
        Return the HTML body of the AnalysisCard (the dataframe, plot, etc.). This is
        used by the AnalysisCard._repr_html_ method to render the AnalysisCard in an
        IPython environment (ex. Jupyter).

        This, not _repr_html_, should be implemented by subclasses of AnalysisCard in
        most cases.

        By default, this method displays the raw data in a pandas DataFrame.
        """

        return f"<div class='content'>{self.df.to_html()}</div>"

    def _body_papermill(self) -> Any:  # pyre-ignore[3]
        """
        Return the body of the AnalysisCard in a simplified format for when html is
        undesirable (ex. when rendering the Ax website).

        By default, this method displays the raw data in a pandas DataFrame.
        """

        return self.df


class ErrorAnalysisCard(AnalysisCard):
    # TODO: Implement improved rendering which shows the traceback.
    # def _ipython_display_(self) -> None: ...
    pass


def error_card_from_exception(
    analysis_name: str, exception: Exception
) -> ErrorAnalysisCard:
    return ErrorAnalysisCard(
        name=analysis_name,
        title=f"{analysis_name} Error",
        subtitle=(
            f"An {exception.__class__.__name__} occurred while computing "
            f"{analysis_name}."
        ),
        df=pd.DataFrame(),
        blob="".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ),
    )


class AnalysisE(ExceptionE):
    analysis: Analysis

    def __init__(
        self,
        message: str,
        exception: Exception,
        analysis: Analysis,
    ) -> None:
        super().__init__(message, exception)
        self.analysis = analysis

    def error_card(self) -> ErrorAnalysisCard:
        return error_card_from_exception(
            analysis_name=self.analysis.__class__.__name__,
            exception=self.exception,
        )


class Analysis(Protocol):
    """
    An Analysis is a class that given either and Experiment, a GenerationStrategy, or
    both can compute some data intended for end-user consumption. The data is returned
    to the user in the form of an AnalysisCard which contains the raw data, a blob (the
    data processed for end-user consumption), and miscellaneous metadata that can be
    useful for rendering the card or a collection of cards.

    The AnalysisCard is a thin wrapper around the raw data and the processed blob;
    Analyses impose structure on their blob should subclass Analysis. See
    PlotlyAnalysis for an example which produces cards where the blob is always a
    Plotly Figure object.

    A good pattern to follow when implementing your own Analyses is to configure
    "settings" (like which parameter or metrics to operate on, or whether to use
    observed or modeled effects) in your Analyses' __init__ methods, then to consume
    these settings in the compute method.
    """

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        # Note: when implementing compute always prefer experiment.lookup_data() to
        # experiment.fetch_data() to avoid unintential data fetching within the report
        # generation.
        ...

    def compute_result(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> Result[AnalysisCardBase, AnalysisE]:
        """
        Utility method to compute an AnalysisCard as a Result. This can be useful for
        computing many Analyses at once and handling Exceptions later.
        """

        try:
            card = self.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            return Ok(value=card)

        except Exception as e:
            logger.error(f"Failed to compute {self.__class__.__name__}")
            logger.error(traceback.format_exc())

            return Err(
                value=AnalysisE(
                    message=f"Failed to compute {self.__class__.__name__}",
                    exception=e,
                    analysis=self,
                )
            )

    def _create_analysis_card(
        self,
        title: str,
        subtitle: str,
        df: pd.DataFrame,
    ) -> AnalysisCard:
        """
        Make an AnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCard(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            blob=df.to_json(),
        )

    def _create_analysis_card_group(
        self,
        children: Sequence[AnalysisCardBase],
    ) -> AnalysisCardGroup:
        """
        Make an AnalysisCardGroup from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCardGroup(
            name=self.__class__.__name__,
            children=children,
        )

    def _create_analysis_card_group_or_card(
        self,
        children: Sequence[AnalysisCardBase],
    ) -> AnalysisCardBase:
        """
        Make an AnalysisCardGroup from this Analysis using provided fields and
        details about the Analysis class. If there is only one child, return the child
        directly instead of wrapping it in a group.
        """
        if len(children) == 1:
            return children[0]

        return self._create_analysis_card_group(children=children)
