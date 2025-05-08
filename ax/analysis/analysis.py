# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import traceback
from collections.abc import Iterable
from enum import Enum, IntEnum
from logging import Logger
from typing import Any, Protocol, Sequence

import pandas as pd
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, ExceptionE, Ok, Result
from ax.utils.tutorials.environment import is_running_in_papermill
from IPython import get_ipython
from IPython.display import display, DisplayObject, HTML, Markdown
from plotly import graph_objects as go

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


class AnalysisCardLevel(IntEnum):
    DEBUG = 0
    LOW = 10
    MID = 20
    HIGH = 30
    CRITICAL = 40


class AnalysisCardCategory(IntEnum):
    ERROR = 0
    ACTIONABLE = 1
    INSIGHT = 2
    DIAGNOSTIC = 3  # Equivalent to "health check" in online setting
    INFO = 4


class AnalysisBlobAnnotation(Enum):
    DATAFRAME = "dataframe"
    PLOTLY = "plotly"
    MARKDOWN = "markdown"
    HEALTHCHECK = "healthcheck"
    ERROR = "error"


class AnalysisCard(Base):
    # Name of the analysis computed, usually the class name of the Analysis which
    # produced the card. Useful for grouping by when querying a large collection of
    # cards.
    name: str
    # Arguments passed to the Analysis which produced the card, or their eventual
    # values if they were inferred.
    attributes: dict[str, Any]

    title: str
    subtitle: str

    # Level of the card with respect to its importance. Higher levels are more
    # important, and will be displayed first.
    level: int

    df: pd.DataFrame  # Raw data produced by the Analysis

    # Blob is the data processed for end-user consumption, encoded as a string,
    # typically JSON. Subclasses of Analysis can define their own methods for consuming
    # the blob and presenting it to the user (ex. PlotlyAnalysisCard.get_figure()
    # decodes the blob into a go.Figure object).
    blob: str
    # Type of the card (ex: "insight", "diagnostic"), useful for
    # grouping the cards to display only one category in notebook environments.
    category: int
    # How to interpret the blob (ex. "dataframe", "plotly", "markdown")
    blob_annotation: AnalysisBlobAnnotation = AnalysisBlobAnnotation.DATAFRAME

    # Singleton for tracking whether this is the first time the AnalysisCard is being
    # initialized. This is used to control whether the custom IPython Formatter
    # needs to be registered.
    _first_initialization: bool = True

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        blob: str,
        category: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.subtitle = subtitle
        self.level = level
        self.df = df
        self.blob = blob
        self.attributes = {} if attributes is None else attributes
        self.category = category

        if AnalysisCard._first_initialization:
            AnalysisCard._first_initialization = False

            # Register a custom IPython Formatter for lists of AnalysisCard objects.
            # This allows the result of Analysis.compute(...) to be displayed in a
            # useful way in IPython environments (ex. Jupyter).
            ip = get_ipython()
            if ip is not None:
                html_formatter = ip.display_formatter.formatters["text/html"]
                html_formatter.for_type(list, _analysis_card_list_html_formatter)

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

    def _body_papermill(self) -> DisplayObject | go.Figure | pd.DataFrame:
        """
        Return the body of the AnalysisCard in a simplified format for when html is
        undesirable.

        By default, this method displays the raw data in a pandas DataFrame.
        """

        return self.df


class ErrorAnalysisCard(AnalysisCard):
    blob_annotation: AnalysisBlobAnnotation = AnalysisBlobAnnotation.ERROR


def display_cards(
    cards: Iterable[AnalysisCard], minimum_level: int = AnalysisCardLevel.LOW
) -> None:
    """
    Display a collection of AnalysisCards in IPython environments (ex. Jupyter).

    Cards get grouped by name then sorted by level, descending. Cards with level less
    than minimum_level are filtered out.

    Args:
        cards: Collection of AnalysisCards to display.
        minimum_level: Minimum level of cards to display.
    """
    # If we are running in papermill, display the cards one by one. Otherwise, generate
    # and display the full HTML
    if is_running_in_papermill():
        for card in _group_and_sort_cards(cards=cards, minimum_level=minimum_level):
            display(card)
    else:
        display(
            HTML(
                _generate_cards_html(
                    cards=_group_and_sort_cards(
                        cards=cards, minimum_level=minimum_level
                    )
                )
            )
        )


def _group_and_sort_cards(
    cards: Iterable[AnalysisCard], minimum_level: int = AnalysisCardLevel.LOW
) -> list[AnalysisCard]:
    """
    Group like cards together, filter out cards with level less than minimum_level,
    and sort by level.

    Args:
        cards: Collection of AnalysisCards to display.
        minimum_level: Minimum level of cards to display.
    """
    # Group cards by name, filter out cards with level less than minimum_level, and
    # sort the resulting groups by level, descending.
    card_groups = [
        sorted(
            [
                card
                for card in cards
                if card.name == name and card.level >= minimum_level
            ],
            key=lambda card: card.level,
            reverse=True,
        )
        for name in {card.name for card in cards}
    ]

    # Sort the groups by maximum level, descending, then flatten the groups into a
    # single list.
    return [
        card
        for group in sorted(
            card_groups,
            key=lambda group: max([card.level for card in group])
            if len(group) > 0
            else 0,
            reverse=True,
        )
        for card in group
    ]


def _generate_cards_html(cards: Iterable[AnalysisCard]) -> str:
    """
    Generate HTML for a collection of AnalysisCards.

    Args:
        cards: Collection of AnalysisCards to display.
        minimum_level: Minimum level of cards to display.
    """

    return html_grid_template.format(
        card_divs="".join([card._repr_html_() for card in cards])
    )


# pyre-ignore[2]: IPython formatter for can truly take in any object.
def _analysis_card_list_html_formatter(obj: Any) -> str | None:
    """
    IPython HTML formatter for lists of AnalysisCards. This is used to conveniently
    display the return values from Analysis.compute(...) or
    Client.compute_analyses(...).

    Will either return the HTML representation of the list of AnalysisCards, or None
    if the default IPython formatter should be used instead.
    """

    if not isinstance(obj, list):
        return None

    # Do not use the custom formatter if we are running in papermill.
    if is_running_in_papermill():
        return None

    # Intentionally using generator expression to avoid materializing the list.
    if not all(isinstance(card, AnalysisCard) for card in obj):
        return None

    return _generate_cards_html(obj)


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

    def error_card(self) -> list[AnalysisCard]:
        return [
            ErrorAnalysisCard(
                name=self.analysis.name,
                title=f"{self.analysis.name} Error",
                subtitle=f"An error occurred while computing {self.analysis}",
                attributes=self.analysis.attributes,
                blob="".join(
                    traceback.format_exception(
                        type(self.exception),
                        self.exception,
                        self.exception.__traceback__,
                    )
                ),
                df=pd.DataFrame(),
                level=AnalysisCardLevel.DEBUG,
                category=AnalysisCardCategory.ERROR,
            )
        ]


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

    """
    The exception class to use when computing this Analysis. This is used to
    construct the AnalysisE when an exception is thrown during compute.
    """
    exception_class: type[AnalysisE] = AnalysisE

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[AnalysisCard]:
        # Note: when implementing compute always prefer experiment.lookup_data() to
        # experiment.fetch_data() to avoid unintential data fetching within the report
        # generation.
        ...

    def compute_result(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> Result[Sequence[AnalysisCard], AnalysisE]:
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
            logger.error(f"Failed to compute {self.__class__.__name__}: {e}")

            return Err(
                value=self.exception_class(
                    message=f"Failed to compute {self.__class__.__name__}",
                    exception=e,
                    analysis=self,
                )
            )

    def _create_analysis_card(
        self,
        title: str,
        subtitle: str,
        level: int,
        df: pd.DataFrame,
        category: int,
    ) -> AnalysisCard:
        """
        Make an AnalysisCard from this Analysis using provided fields and
        details about the Analysis class.
        """
        return AnalysisCard(
            name=self.name,
            attributes=self.attributes,
            title=title,
            subtitle=subtitle,
            level=level,
            df=df,
            blob=df.to_json(),
            category=category,
        )

    @property
    def name(self) -> str:
        """The name the AnalysisCard will be given in compute."""
        return self.__class__.__name__

    @property
    def attributes(self) -> dict[str, Any]:
        """The attributes the AnalysisCard will be given in compute."""
        return self.__dict__

    def __repr__(self) -> str:
        try:
            return (
                f"{self.__class__.__name__}(name={self.name}, "
                f"attributes={json.dumps(self.attributes)})"
            )
        # in case there is logic in name or attributes that throws a json error
        except Exception:
            return self.__class__.__name__
