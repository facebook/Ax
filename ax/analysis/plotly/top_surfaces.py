# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Literal

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import (
    AnalysisCard,
    AnalysisCardBase,
    AnalysisCardGroup,
    ErrorAnalysisCard,
)
from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
from ax.analysis.plotly.surface.contour import (
    CONTOUR_CARDGROUP_SUBTITLE,
    CONTOUR_CARDGROUP_TITLE,
    ContourPlot,
)
from ax.analysis.plotly.surface.slice import (
    SLICE_CARDGROUP_SUBTITLE,
    SLICE_CARDGROUP_TITLE,
    SlicePlot,
)
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import assert_is_instance, override

TS_CARDGROUP_TITLE = (
    "Top Surfaces Analysis: Parameter sensitivity, slice, and contour plots"
)

TS_CARDGROUP_SUBTITLE = (
    "The top surfaces analysis displays three analyses in one. First, it shows "
    "parameter sensitivities, which shows the sensitivity of the metrics in the "
    "experiment to the most important parameters. Subsetting to only the most "
    "important parameters, it then shows slice plots and contour plots for each "
    "metric in the experiment, displaying the relationship between the metric and "
    "the most important parameters. "
)


class TopSurfacesAnalysis(Analysis):
    def __init__(
        self,
        metric_name: str | None = None,
        order: Literal["first", "second", "total"] = "second",
        top_k: int = 5,
        relativize: bool = False,
    ) -> None:
        self.metric_name = metric_name
        self.order = order
        self.top_k = top_k
        self.relativize = relativize

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        if experiment is None:
            raise UserInputError(
                "TopSurfacesAnalysis requires an Experiment to compute."
            )

        if self.metric_name is not None:
            metric_name = self.metric_name
        else:
            metric_name = select_metric(experiment=experiment)

        # Process the sensitivity analysis card to find the top K surfaces which
        # consist exclusively of tunable parameters (i.e. no fixed parameters, task
        # parameters, or OneHot parameters).
        sensitivity_analysis_card = assert_is_instance(
            SensitivityAnalysisPlot(
                metric_names=[metric_name],
                order=self.order,
                top_k=self.top_k,
            ).compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            ),
            AnalysisCard,
        )
        children: list[AnalysisCardBase] = [sensitivity_analysis_card]

        sensitivity_df = sensitivity_analysis_card.df.copy()
        filtered_df = sensitivity_df[
            sensitivity_df["parameter_name"].apply(
                lambda x: all(
                    name in experiment.search_space.tunable_parameters
                    for name in x.split(" & ")
                )
            )
        ]
        sorted_df = filtered_df.sort_values(by="sensitivity", key=abs, ascending=False)
        top_k = sorted_df.head(self.top_k)
        top_surfaces = top_k["parameter_name"].to_list()

        surface_cards_or_error = [
            _compute_surface_plot(
                surface_name=surface_name,
                metric_name=metric_name,
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
                relativize=self.relativize,
            )
            for surface_name in top_surfaces
        ]

        # Filter out any ErrorAnalysisCards (i.e. failed to compute). When these
        # occur their presence will be logged by compute_or_error_card in
        # _compute_surface_plot so it's safe to filter them here
        surface_cards = [
            card
            for card in surface_cards_or_error
            if not isinstance(card, ErrorAnalysisCard)
        ]

        slice_cards = [
            assert_is_instance(card, AnalysisCard)
            for card in surface_cards
            if card.name == "SlicePlot"
        ]

        if len(slice_cards) > 0:
            children.append(
                AnalysisCardGroup(
                    name="TopSurfaceAnalysisSlicePlots",
                    title=SLICE_CARDGROUP_TITLE,
                    subtitle=SLICE_CARDGROUP_SUBTITLE,
                    children=slice_cards,
                )
            )

        contour_cards = [
            assert_is_instance(card, AnalysisCard)
            for card in surface_cards
            if card.name == "ContourPlot"
        ]

        if len(contour_cards) > 0:
            children.append(
                AnalysisCardGroup(
                    name="TopSurfaceAnalysisContourPlots",
                    title=CONTOUR_CARDGROUP_TITLE,
                    subtitle=CONTOUR_CARDGROUP_SUBTITLE,
                    children=contour_cards,
                )
            )

        return self._create_analysis_card_group(
            title=TS_CARDGROUP_TITLE,
            subtitle=TS_CARDGROUP_SUBTITLE,
            children=children,
        )


def _compute_surface_plot(
    surface_name: str,
    metric_name: str,
    experiment: Experiment | None,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
    relativize: bool = False,
) -> AnalysisCardBase:
    """Computes either a Slice or Contour plot for a given surface.
    Args:
        surface_name: The name of the parameter to plot. Either a single parameter or
            two parameters delimited by " & ". This determines whether a Slice or
            Contour plot is computed.
        experiment: The experiment to plot.
        generation_strategy: The generation strategy to plot.
        adapter: The adapter to plot.

    Returns:
        A PlotlyAnalysisCard containing the plot.
    """

    if "&" in surface_name:
        x_parameter_name, y_parameter_name = surface_name.split(" & ")
        analysis = ContourPlot(
            x_parameter_name=x_parameter_name,
            y_parameter_name=y_parameter_name,
            metric_name=metric_name,
            relativize=relativize,
        )

    else:
        analysis = SlicePlot(
            parameter_name=surface_name, metric_name=metric_name, relativize=relativize
        )

    return analysis.compute_or_error_card(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )
