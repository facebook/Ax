# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import chain
from logging import Logger
from typing import Sequence

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.plotly.arm_effects.utils import (
    get_predictions_by_arm,
    prepare_arm_effects_plot,
)

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysis, PlotlyAnalysisCard
from ax.analysis.plotly.utils import get_nudge_value, is_predictive
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.outcome_constraint import OutcomeConstraint
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.registry import Generators
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws, override

logger: Logger = get_logger(__name__)


class InSampleEffectsPlot(PlotlyAnalysis):
    """
    Plotly Insample Effects plot for a single metric on a single trial, with one point
    per unique arm across all trials. The plot may either use modeled effects, or
    raw / observed data.

    This plot is useful for understanding how arms compare to eachother for a given
    metric.

    TODO: Allow trial index to be optional so we can plot all trials for non batch
    experiments.

    The DataFrame computed will contain one row per arm and the following columns:
        - source: In-sample or model key that generated the candidate
        - arm_name: The name of the arm
        - mean: The observed or predicted mean of the metric specified
        - sem: The observed or predicted sem of the metric specified
        - error_margin: The 95% CI of the metric specified for the arm
        - size_column: The size of the circle in the plot, which represents
            the probability that the arm is feasible (does not violate any
            constraints).
        - parameters: A string representation of the parameters for the arm
            to be viewed in the tooltip.
        - constraints_violated: A string representation of the probability
            each constraint is violated for the arm, to be viewed in the tooltip.
    """

    def __init__(
        self, metric_name: str, trial_index: int, use_modeled_effects: bool
    ) -> None:
        """
        Args:
            metric_name: The name of the metric to plot.
            trial_index: The of the trial to plot arms for.
            use_modeled_effects: Whether to use modeled effects or show
                observed effects.
        """

        self.metric_name = metric_name
        self.trial_index = trial_index
        self.use_modeled_effects = use_modeled_effects

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[PlotlyAnalysisCard]:
        if experiment is None:
            raise UserInputError("InSampleEffectsPlot requires an Experiment.")

        model = _get_model(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
            use_modeled_effects=self.use_modeled_effects,
            trial_index=self.trial_index,
            metric_name=self.metric_name,
        )

        outcome_constraints = (
            []
            if experiment.optimization_config is None
            else Derelativize()
            .transform_optimization_config(
                # TODO[T203521207]: move cloning into transform_optimization_config
                optimization_config=none_throws(experiment.optimization_config).clone(),
                modelbridge=model,
            )
            .outcome_constraints
        )
        df = _prepare_data(
            experiment=experiment,
            model=model,
            outcome_constraints=outcome_constraints,
            metric_name=self.metric_name,
            trial_index=self.trial_index,
        )
        fig = prepare_arm_effects_plot(
            df=df, metric_name=self.metric_name, outcome_constraints=outcome_constraints
        )
        nudge = get_nudge_value(
            metric_name=self.metric_name,
            experiment=experiment,
            use_modeled_effects=self.use_modeled_effects,
        )

        subtitle = (
            "The in-sample effects plot visualizes the "
            f"{self._plot_type_string.lower()} effects from previously-run "
            "arms on a specific metric, providing insights into their "
            "performance. This plot allows one to compare and contrast the "
            "effectiveness of different arms, highlighting which configurations "
            "have yielded the most favorable outcomes. Note, this plot applies "
            "Empirical Bayes shrinkage by default, so metric effects may not "
            "perfectly match raw observations."
        )

        return [
            self._create_plotly_analysis_card(
                title=(
                    f"{self._plot_type_string} Effects for {self.metric_name} "
                    f"on trial {self.trial_index}"
                ),
                subtitle=subtitle,
                level=AnalysisCardLevel.MID + nudge,
                df=df,
                fig=fig,
                category=AnalysisCardCategory.INSIGHT,
            )
        ]

    @property
    def name(self) -> str:
        return f"{self._plot_type_string}EffectsPlot"

    @property
    def _plot_type_string(self) -> str:
        return "Modeled" if self.use_modeled_effects else "Observed"


def _get_model(
    experiment: Experiment,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
    use_modeled_effects: bool,
    trial_index: int,
    metric_name: str,
) -> Adapter:
    """Get a model for predictions.

    Args:
        experiment: Used to get the data for the model.
        generation_strategy: Used to get the model if we want to use modeled effects
            and the current model is predictive.
        adapter: A custom adapter that can be passed and used in place of the current
            model on the generation strategy.
        use_modeled_effects: Whether to use modeled effects.
        trial_index: The trial index to get data for in training the model.
        metric_name: The name of the metric we're plotting, which we validate has
            data on the trial.

    Returns:
        If use_modeled_effects is False, returns a Thompson model, which just predicts
        from the data.
        If use_modeled_effects is True, returns the current model on the generation
        strategy if it is predictive.  Otherwise, returns an empirical Bayes model.
    """
    # TODO: merge this method into get_adapter helper in the utils file
    trial_data = experiment.lookup_data(trial_indices=[trial_index])
    if trial_data.filter(metric_names=[metric_name]).df.empty:
        raise DataRequiredError(
            f"Cannot plot effects for '{metric_name}' on trial {trial_index} "
            "because it has no data.  Either the data is not available yet, "
            "or we encountered an error fetching it."
        )
    # If we're not using modeled effects, we just want to predict observed data
    if not use_modeled_effects:
        return Generators.THOMPSON(
            data=trial_data,
            search_space=experiment.search_space,
            experiment=experiment,
        )

    try:
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
    except UserInputError:
        logger.info(
            "Using Empirical Bayes to model effects because we were unable to find "
            " a suitable adapter."
        )
        return Generators.EMPIRICAL_BAYES_THOMPSON(
            experiment=experiment, data=trial_data
        )

    if not is_predictive(adapter=relevant_adapter):
        logger.debug(
            "Using Empirical Bayes to model effects because we were unable to find "
            + " a suitable adapter on the current Generation Strategy. Current "
            + f" Generation Strategy is: {generation_strategy} and model is: {adapter}"
        )
        return Generators.EMPIRICAL_BAYES_THOMPSON(
            experiment=experiment, data=trial_data
        )

    return relevant_adapter


def _prepare_data(
    experiment: Experiment,
    model: Adapter,
    outcome_constraints: list[OutcomeConstraint],
    metric_name: str,
    trial_index: int,
) -> pd.DataFrame:
    """Prepare data for plotting.  Data should include columns for:
    - source: In-sample or model key that generated the candidate
    - arm_name: Name of the arm
    - mean: Predicted metric value
    - error_margin: 1.96 * predicted sem for plotting 95% CI
    - **PARAMETER_NAME: The value of each parameter for the arm.  Will be used
        for the tooltip.
    There will be one row for each arm in the model's training data and one for
    each arm in the generator runs of the candidate trial.  If an arm is in both
    the training data and the candidate trial, it will only appear once for the
    candidate trial.

    Args:
        experiment: Experiment to plot
        model: Adapter being used for prediction
        outcome_constraints: Derelatives outcome constraints used for
            assessing feasibility
        metric_name: Name of metric to plot
        trial_index: Optional trial index to plot.  If not specified, will
            plot the most recent non-abandoned trial with all observations.
    """
    try:
        trial = experiment.trials[trial_index]
    except KeyError:
        raise UserInputError(
            f"Cannot plot effects for {trial_index} because "
            f"it's missing from {experiment}."
        )

    status_quo_prediction = (
        []
        if experiment.status_quo is None
        else [
            get_predictions_by_arm(
                model=model,
                metric_name=metric_name,
                outcome_constraints=outcome_constraints,
                gr=GeneratorRun(
                    arms=[experiment.status_quo],
                    model_key="Status Quo",
                ),
            )
        ]
    )
    trial_predictions = [
        get_predictions_by_arm(
            model=model,
            metric_name=metric_name,
            outcome_constraints=outcome_constraints,
            gr=gr,
            abandoned_arms={a.name for a in trial.abandoned_arms},
        )
        for gr in trial.generator_runs
    ]

    df = pd.DataFrame.from_records(
        list(
            chain(
                *[
                    *trial_predictions,
                    *status_quo_prediction,
                ]
            )
        )
    )
    df.drop_duplicates(subset="arm_name", keep="last", inplace=True)
    return df
