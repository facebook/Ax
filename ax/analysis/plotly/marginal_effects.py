# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCardBase
from ax.analysis.plotly.color_constants import COLOR_FOR_DECREASES, COLOR_FOR_INCREASES
from ax.analysis.plotly.plotly_analysis import create_plotly_analysis_card

from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.parameter import ChoiceParameter
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.plot.helper import get_plot_data
from ax.utils.stats.statstools import marginal_effects
from plotly import graph_objects as go
from pyre_extensions import none_throws, override


MARGINAL_EFFECTS_CARDGROUP_TITLE = "Marginal Effects For Each Parameter Factor Level"

MARGINAL_EFFECTS_CARDGROUP_SUBTITLE = (
    "This analysis has a bar chart for each parameter, showing the predicted "
    "relative change in the metric for each factor level of the parameter. "
    "The bars are colored based on whether the effect is positive or negative, "
    "and the error bars represent the standard error of the effect."
)


@final
class MarginalEffectsPlot(Analysis):
    """
    Plotly bar charts showing the marginal effect of each factor level of
    selected `ChoiceParameters` on the given metric. This plot is useful
    for understanding how each parameter value marginally affects the metric.

    The DataFrame computed will contain the following columns:
        - Name: The name of the parameter.
        - Level: The level of the parameter.
        - Beta: The effect size, which is the relative (in %) change in the
            metric when the specified factor is used compared to the baseline
            which is marginalized over all factors and levels.
        - SE: The standard error.
    """

    def __init__(
        self,
        metric_name: str,
        parameters: list[str] | None = None,
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to plot.
            parameters: The names of the `ChoiceParameters` to include in the
            analysis. If not specified, all `ChoiceParameters` in the Experiment
            will be included.
        """

        self.metric_name = metric_name
        self.parameters = parameters

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardBase:
        if experiment is None:
            raise UserInputError("MarginalEffectsPlot requires an Experiment")

        # Either extract ChoiceParameters from experiment or check the ones passed in
        if self.parameters is None:
            self.parameters = [
                param_name
                for param_name, param in experiment.parameters.items()
                if isinstance(param, ChoiceParameter)
            ]
            if len(self.parameters) == 0:
                raise UserInputError(
                    "MarginalEffectsPlot is only for `ChoiceParameter`s, "
                    "but no ChoiceParameters were found in the experiment."
                )
        else:
            for param_name in none_throws(self.parameters):
                parameter = experiment.parameters.get(param_name)
                if not isinstance(parameter, ChoiceParameter):
                    raise UserInputError(
                        "MarginalEffectsPlot is only for `ChoiceParameter`s, but got."
                        f"'{param_name}' which is of type {type(parameter).__name__}."
                    )

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
        df = _prepare_data(
            metric=self.metric_name,
            adapter=relevant_adapter,
            parameters=self.parameters,
        )
        # Create a card for each parameter
        cards = []
        for param in df["Name"].unique():
            param_df = df[df["Name"] == param]
            fig = _prepare_plot(param_df, self.metric_name)
            cards.append(
                create_plotly_analysis_card(
                    name=self.__class__.__name__,
                    title=f"Marginal Effects for {param}",
                    subtitle=(
                        "This plot visualizes the predicted relative changes in "
                        f"{self.metric_name} for each factor level of {param}, "
                        "offering insights into their impact on the metric. "
                        "By comparing the effects of different levels, this plot "
                        "helps identify which factor levels have the most "
                        "significant influence, providing a detailed understanding "
                        "of the experimental results and guiding future decisions on "
                        "factor selection."
                    ),
                    df=param_df,
                    fig=fig,
                )
            )
        return self._create_analysis_card_group(
            title=MARGINAL_EFFECTS_CARDGROUP_TITLE,
            subtitle=MARGINAL_EFFECTS_CARDGROUP_SUBTITLE,
            children=cards,
        )


def compute_marginal_effects_adhoc(
    metric_name: str,
    experiment: Experiment,
    parameters: list[str] | None = None,
    generation_strategy: GenerationStrategy | None = None,
    adapter: Adapter | None = None,
) -> AnalysisCardBase:
    """
    Helper method to expose adhoc marginal effects plotting. Only for
    advanced users in a notebook setting.

    Args:
        metric_name: The name of the metric to plot.
        parameters: The names of the `ChoiceParameters` to include in the
        analysis. If not specified, all `ChoiceParameters` in the Experiment
        will be included.
        experiment: The experiment to source data from.
        generation_strategy: Optional. The generation strategy to extract
            the adapter from.
        adapter: Optional. The adapter to use for predictions.
    """

    analysis = MarginalEffectsPlot(metric_name=metric_name, parameters=parameters)

    return analysis.compute(
        experiment=experiment,
        generation_strategy=generation_strategy,
        adapter=adapter,
    )


def _prepare_data(
    metric: str,
    adapter: Adapter,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare the in sample data for the marginal effects plot, then call
    `marginal_effects()` to create a table of the effects containing each
    factor level for the specified metric. The df returned will have the columns:
        - Name: The name of the parameter.
        - Level: The level of the parameter.
        - Beta: The effect size, which is the relative change in the
            metric when the specified factor is used compared to the baseline
            which is marginalized over all factors and levels.
        - SE: The standard error.
    """

    # TODO (T230319678, enofrey): replace legacy `get_plot_data` method
    plot_data, _, _ = get_plot_data(
        model=adapter, generator_runs_dict={}, metric_names={metric}
    )

    arm_dfs = []
    for arm in plot_data.in_sample.values():
        arm_df = pd.DataFrame(arm.parameters, index=[arm.name])
        arm_df["mean"] = arm.y_hat[metric]
        arm_df["sem"] = arm.se_hat[metric]
        arm_dfs.append(arm_df)
    effect_table = marginal_effects(
        df=pd.concat(arm_dfs, axis=0), covariates=parameters
    )
    if parameters is not None:
        effect_table = effect_table[effect_table["Name"].isin(parameters)]

    return effect_table


def _prepare_plot(df: pd.DataFrame, metric_name: str) -> go.Figure:
    """Create a bar plot of the marginal effects for a single parameter."""
    param = df["Name"].iloc[0]
    colors = [
        COLOR_FOR_INCREASES if beta > 0 else COLOR_FOR_DECREASES for beta in df["Beta"]
    ]

    bar_plot = go.Bar(
        x=df["Level"],
        y=df["Beta"]
        / 100,  # Divide by 100 (multiplied 100x later in percentage formatting)
        error_y={
            "type": "data",
            "array": df["SE"] / 100,
        },
        marker={"color": colors},
    )

    fig = go.Figure(data=[bar_plot])
    fig.update_layout(
        title=f"Marginal Effects for {param} on {metric_name}",
        xaxis_title="Factor Level",
        yaxis_title="Effect relative to experiment average",
        yaxis_tickformat=".2%",
        yaxis_hoverformat=".2%",
        showlegend=False,
    )
    return fig
