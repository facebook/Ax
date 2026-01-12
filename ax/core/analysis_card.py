# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
from ax.utils.common.base import SortableBase
from ax.utils.tutorials.environment import is_running_in_papermill
from IPython.display import display, HTML, Markdown
from plotly.offline import get_plotlyjs

# Simple HTML template for rendering a card with a title, subtitle, and body with
# scrollable overflow.
html_card_template = """
<style>
.card {{
    overflow: auto;
    border-width: thin;
    border-color: lightgray;
    border-style: solid;
    border-radius: 0.5em;
    padding: 10px;
}}
.card-header:hover {{
    cursor: pointer;
}}
</style>
<div class="card">
    <div class="card-header">
        <details>
            <summary><b>{title_str}</b></summary>
            <p>{subtitle_str}</p>
        </details>
    </div>
    <div class="card-body">
        {body_html}
    </div>
</div>
"""

# Simple HTML template for rendering a *group* card with a title, subtitle, and
# body with scrollable overflow.
html_group_card_template = """
<style>
.group-card {{
    overflow: auto;
    margin-top: 25px;
}}
.group-header {{
    font-size: 1.5em;
}}
.group-subtitle {{
    font-size: 1em;
}}
</style>
<div class="group-card">
    <div>
        <b class="group-header">
        {title_str}
        </b>
        <p class="group-subtitle">
        {subtitle_str}
        </p>
    </div>
    <div class="group-body">
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
    """

    name: str

    title: str
    subtitle: str

    _timestamp: datetime

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Args:
            name: The class name of the Analysis that produced this card (ex. "Summary",
                "ArmEffects", etc.).
            title: Human-readable title which describes the card's contents. This
                appears in all user facing interfaces as the title of the card.
            subtitle: Human-readable subtitle which provides additional information or
                context to improve the usability of the analysis for users.
            timestamp: The time at which the Analysis was computed. This can be
                especially useful when querying the database for the most recently
                produced artifacts.
        """
        self.name = name
        self.title = title
        self.subtitle = subtitle
        self._timestamp = timestamp if timestamp is not None else datetime.now()

    @abstractmethod
    def flatten(self) -> list[AnalysisCard]:
        """
        Returns a list of AnalysisCards contained in this card in order. This is useful
        when processing a collection of cards where order is necessary but grouping can
        safely be ignored (ex. when rendering a collection of cards in an IPython).
        """
        pass

    @abstractmethod
    def hierarchy_str(self, level: int = 0) -> str:
        """
        Returns a string representation of the card's nested hierarchy structure. This
        is useful for debugging and logging.

        Example:
            Root
                Child 1
                    Grandchild 1
                    Grandchild 2
                Child 2
                    Grandchild 3
        """
        pass

    @property
    def _unique_id(self) -> str:
        return str(hash(str(self.__dict__)))

    def _ipython_display_(self) -> None:
        """
        IPython display hook. This is called when the AnalysisCard is rendered in an
        IPython environment (ex. Jupyter). This method should not be implemented by
        subclasses; instead they should implement the representation-specific helpers
        such as _body_html_ and _body_papermill_.
        """

        # If in papermill used simplified rendering. This is used for the Ax website.
        if is_running_in_papermill():
            for card in self.flatten():
                display(Markdown(f"**{card.title}**\n\n{card.subtitle}"))
                display(card._body_papermill())

            return

        display(HTML(self._repr_html_()))

    @abstractmethod
    def _body_html(self, depth: int) -> str:
        """
        Return the HTML body of the card (the dataframe, plot, grid, etc.). This is
        used by the AnalysisCardBase._repr_html_ method to render the card in an
        IPython environment (ex. Jupyter).

        This, not _repr_html_ or _to_html, should be implemented by subclasses of
        AnalysisCardBase in most cases in order to keep treatment of titles and
        subtitles consistent.

        Since this method can sometimes be called recursively a "depth" parameter can
        be passed in as well.
        """
        pass

    def _repr_html_(self) -> str:
        """
        IPython HTML representation hook. This is called when the AnalysisCard is
        rendered in an IPython environment (ex. Jupyter). This method should be
        implemented by subclasses of Analysis to display the AnalysisCard in a useful
        way.
        """
        # require.js is not compatible with ES6 import used by plotly.js so we must
        # null out `define` here. This does not affect rendering.
        plotlyjs_script = f"<script>define = null;{get_plotlyjs()}</script>"

        return plotlyjs_script + self._to_html(depth=0)

    def _to_html(self, depth: int) -> str:
        return html_card_template.format(
            title_str=self.title,
            subtitle_str=self.subtitle,
            body_html=self._body_html(depth=depth),
        )


class AnalysisCardGroup(AnalysisCardBase):
    """
    An ordered collection of AnalysisCards. This is useful for grouping related
    analyses together.

    This is analogous to a "branch node" in a tree structure.
    """

    children: list[AnalysisCardBase]

    def __init__(
        self,
        name: str,
        title: str,
        subtitle: str | None,
        children: Sequence[AnalysisCardBase],
        timestamp: datetime | None = None,
    ) -> None:
        """
        Args:
            name: The class name of the Analysis that produced this card (ex. "Summary",
                "ArmEffects", etc.).
            title: Human-readable title which describes the card's contents. This
                appears in all user facing interfaces as the title of the card.
            subtitle: Human-readable subtitle which provides additional information or
                context to improve the usability of the analysis for users.
            timestamp: The time at which the Analysis was computed. This can be
                especially useful when querying the database for the most recently
                produced artifacts.
        """
        super().__init__(
            name=name,
            title=title,
            subtitle=subtitle if subtitle is not None else "",
            timestamp=timestamp,
        )

        self.children = [
            child
            for child in children
            # Filter out empty analysis card groups
            if not (isinstance(child, AnalysisCardGroup) and len(child.children) == 0)
        ]

    @property
    def _unique_id(self) -> str:
        return self.name

    def flatten(self) -> list[AnalysisCard]:
        return [child for child in self.children for child in child.flatten()]

    def hierarchy_str(self, level: int = 0) -> str:
        return f"{'    ' * level}{self.name}\n" + "\n".join(
            child.hierarchy_str(level=level + 1) for child in self.children
        )

    def _body_html(self, depth: int) -> str:
        """
        When rendering an AnalysisCardGroup as HTML use the following rules when
        constructing the card's body:

        * Render children in order
        * Render AnalysisCards (leaves) in a 2xN grid when adjacent to each other
        * Do not render AnalysisCardGroups in a grid
        * Do not render subtitles below depth == 2 (this is handled in
            AnalysisCardBase._to_html, not this method).
        """

        res = []
        leaf_cards = []
        for child in self.children:
            # Accumulate adjacent AnalysisCard leaves so they can be inserted into an
            # HTML grid later.
            if isinstance(child, AnalysisCard):
                leaf_cards.append(child)
                continue

            # If there are leaves accumulated, collect them into a grid and empty
            # the accumulator before appending the current child AnalysisCardGroup
            # underneath.
            if len(leaf_cards) > 0:
                leaves_grid = html_grid_template.format(
                    card_divs="".join(
                        [card._to_html(depth=depth + 1) for card in leaf_cards]
                    )
                )
                res.append(leaves_grid)
                leaf_cards = []

            res.append(child._to_html(depth=depth + 1))

        # Collect the accumulated leaves a final time to append to the result.
        if len(leaf_cards) > 0:
            leaves_grid = html_grid_template.format(
                card_divs="".join(
                    [card._to_html(depth=depth + 1) for card in leaf_cards]
                )
            )
            res.append(leaves_grid)

        return "\n".join(res)

    def _to_html(self, depth: int) -> str:
        return html_group_card_template.format(
            title_str=self.title,
            subtitle_str=self.subtitle,
            body_html=self._body_html(depth=depth),
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
        """
        Args:
            name: The name of the Analysis that produced this card.
            title: A human-readable title which describes the card's contents.
            subtitle: A human-readable subtitle which elaborates on the card's title if
                necessary.
            df: The raw data produced by the Analysis.
            blob: The data processed for end-user consumption, encoded as a string,
                typically JSON.
            timestamp: The time at which the Analysis was computed. This can be
                especially useful when querying the database for the most recently
                produced artifacts.
        """
        super().__init__(
            name=name,
            title=title,
            subtitle=subtitle,
            timestamp=timestamp,
        )

        self.df = df
        self.blob = blob

    @property
    def _unique_id(self) -> str:
        return self.title

    def flatten(self) -> list[AnalysisCard]:
        return [self]

    def hierarchy_str(self, level: int = 0) -> str:
        return f"{'    ' * level}{self.title}"

    def _body_html(self, depth: int) -> str:
        """
        By default, this method displays the raw data in a pandas DataFrame.
        """

        return f"<div class='content'>{self.df.to_html()}</div>"

    def _body_papermill(self) -> Any:
        """
        Return the body of the AnalysisCard in a simplified format for when html is
        undesirable (ex. when rendering the Ax website).

        By default, this method displays the raw data in a pandas DataFrame.
        """

        return self.df


class ErrorAnalysisCard(AnalysisCard):
    # TODO: Implement improved rendering which shows the traceback.
    # def _ipython_display_(self) -> None: ...
    def _body_html(self, depth: int) -> str:
        """
        By default, this method displays the raw data in a pandas DataFrame.
        """

        return f"<div class='content'>{self.blob}</div>"
