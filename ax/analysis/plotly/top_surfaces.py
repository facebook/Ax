# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Literal, Sequence

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
from ax.analysis.plotly.surface.contour import ContourPlot
from ax.analysis.plotly.surface.slice import SlicePlot
from ax.analysis.plotly.utils import select_metric
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from pyre_extensions import override


class TopSurfacesAnalysis(PlotlyAnalysis):
    def __init__(
        self,
        metric_name: str | None = None,
        order: Literal["first", "second", "total"] = "second",
        top_k: int = 5,
    ) -> None:
        self.metric_name = metric_name
        self.order = order
        self.top_k = top_k

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        if experiment is None:
            raise UserInputError(
                "TopSurfacesAnalysis requires an Experiment to compute."
            )

        if self.metric_name is not None:
            metric_name = self.metric_name
        else:
            metric_name = select_metric(experiment=experiment)

        (sensitivity_analysis_card,) = SensitivityAnalysisPlot(
            metric_names=[metric_name],
            order=self.order,
            top_k=self.top_k,
        ).compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # Process the sensitivity analysis card to find the top K surfaces which
        # consist exclusively of tunable parameters (i.e. no fixed parameters, task
        # parameters, or OneHot parameters).
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

        surface_cards = [
            _compute_surface_plot(
                surface_name=surface_name,
                metric_name=metric_name,
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            for surface_name in top_surfaces
        ]

        cards = [sensitivity_analysis_card, *surface_cards]

        # Overwrite the name of the card for grouping purposes during display_analyses.
        for card in cards:
            card.name = "TopSurfacesAnalysis"

        return cards


def _compute_surface_plot(
    surface_name: str,
    metric_name: str,
    experiment: Experiment | None,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
) -> PlotlyAnalysisCard:
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
        )

    else:
        analysis = SlicePlot(parameter_name=surface_name, metric_name=metric_name)

    (card,) = analysis.compute(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )

    return card
