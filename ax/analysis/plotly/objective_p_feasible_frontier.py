# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from logging import Logger
from typing import final

from ax.adapter.base import Adapter
from ax.adapter.torch import TorchAdapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.plotly_analysis import (
    create_plotly_analysis_card,
    PlotlyAnalysisCard,
)
from ax.analysis.plotly.scatter import _prepare_figure as _prepare_figure_scatter
from ax.analysis.plotly.utils import get_trial_statuses_with_fallback
from ax.analysis.utils import (
    extract_relevant_adapter,
    prepare_arm_data,
    validate_experiment,
)
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.multi_acquisition import MultiAcquisition
from ax.utils.common.logger import get_logger
from botorch.acquisition.analytic import LogProbabilityOfFeasibility, PosteriorMean
from pyre_extensions import assert_is_instance, none_throws, override

logger: Logger = get_logger(__name__)
OBJ_PFEAS_CARDGROUP_SUBTITLE = (
    "This plot shows <b>newly generated arms</b> with optimal trade-offs between"
    " <b>Ax model-estimated effect on the objective (x-axis)</b> and <b>Ax-model"
    " estimated probability of satisfying the constraints (y-axis)</b>. This plot"
    " is useful for understanding: 1) how tight the constraints are (sometimes the"
    " constraints can be configured too conservatively, making it difficult to find"
    " an arm that improves the objective(s) while satisfying the constraints),"
    " 2) how much headroom there is with the current optimization configuration "
    " (objective(s) and constraints). <b>If arms that are likely feasible (y-axis),"
    " do not improve your objective enough, revisiting your optimization config and"
    " relaxing the constraints may be helpful.</b> This analysis can be computed"
    " adhoc in a notebook environment, and will change with modifications to the"
    " optimization config, so you can understand the potential impact of optimization"
    " config modifications prior to running another iteration. Get in touch with the"
    " Ax developers for pointers on including these arms in a trial or running this"
    " via a notebook."
)


@final
class ObjectivePFeasibleFrontierPlot(Analysis):
    """
    Plotly Scatter plot for the objective vs p(feasible). Each arm is represented by
    a single point with 95% confidence intervals if the data is available. Effects are
    the predicted effects using a model.
    """

    def __init__(
        self,
        relativize: bool = False,
        additional_arms: Sequence[Arm] | None = None,
        label: str | None = None,
        num_points_to_generate: int = 10,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
    ) -> None:
        """
        Args:
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            label: A label to use in the plot in place of the metric name.
            trial_index: If present, only use arms from the trial with the given index.
            trial_statuses: If present, only use arms from trials with the given
                statuses. By default, exclude STALE, ABANDONED, and FAILED trials.
            num_points_to_generate: The number of points to generate on the frontier.
                Ideally this should be sufficiently large to provide a frontier with
                reasonably good coverage.
        """
        self.relativize = relativize
        # store original additional_arms so can add newly generated frontier arms to
        # additional_arms, but have `compute` remain idempotent.
        self._additional_arms: list[Arm] = (
            [] if additional_arms is None else [*additional_arms]
        )
        self.additional_arms: list[Arm] = self._additional_arms
        self.label = label
        self.num_points_to_generate = num_points_to_generate
        self.trial_statuses: list[TrialStatus] | None = (
            get_trial_statuses_with_fallback(
                trial_statuses=trial_statuses, trial_index=trial_index
            )
        )
        self.trial_index = trial_index

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        ObjectivePFeasibleFrontierPlot requires an Experiment with trials and data, and
        is only valid for single objective constrained problems. Additionally, the
        supplied adapter must be a TorchAdapter using a BoTorchGenerator.
        """
        if (
            experiment_invalid_reason := validate_experiment(
                experiment=experiment,
                require_trials=True,
                require_data=True,
            )
        ) is not None:
            return experiment_invalid_reason

        experiment = none_throws(experiment)

        if experiment.optimization_config is None:
            return "Optimization_config must be set to compute frontier."

        if isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig):
            return "Multi-objective optimization is not supported."

        if len(experiment.optimization_config.outcome_constraints) == 0:
            return (
                "Plotting the objective-p(feasible) frontier requires at least one "
                "outcome constraint."
            )

        if any(
            isinstance(oc, ScalarizedOutcomeConstraint)
            for oc in experiment.optimization_config.outcome_constraints
        ):
            return "Scalarized outcome constraints are not supported yet."

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if not isinstance(relevant_adapter, TorchAdapter) or not isinstance(
            relevant_adapter.generator, BoTorchGenerator
        ):
            return (
                "The Objective vs P(feasible) plot cannot be computed using the"
                f" current Adapter ({relevant_adapter}) and generator"
                f" ({relevant_adapter.generator}). Only TorchAdapters using"
                " BoTorchGenerators are supported."
            )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        experiment = none_throws(experiment)
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )
        # Generate arms on the objective p_feasible frontier
        # Specify to optimize multiple acquisition functions (via
        # MultiAcquisition). This will optimize the PosteriorMean and the
        # LogProbabilityOfFeasibility using MOO.
        generator = assert_is_instance(relevant_adapter.generator, BoTorchGenerator)
        orig_acquisition_class = generator.acquisition_class

        orig_acquisition_options = generator.acquisition_options
        orig_botorch_acqf_classes_with_options = (
            generator._botorch_acqf_classes_with_options
        )
        generator.acquisition_class = MultiAcquisition
        generator.acquisition_options = {}
        generator._botorch_acqf_classes_with_options = [
            (PosteriorMean, {}),
            (LogProbabilityOfFeasibility, {}),
        ]
        frontier_gr = relevant_adapter.gen(n=self.num_points_to_generate)
        # in case this generator is used again, restore the original settings
        generator.acquisition_class = orig_acquisition_class
        generator.acquisition_options = orig_acquisition_options
        generator._botorch_acqf_classes_with_options = (
            orig_botorch_acqf_classes_with_options
        )

        arms = [
            Arm(name=f"frontier_{i}", parameters=arm.parameters)
            for i, arm in enumerate(frontier_gr.arms)
        ]
        # concatenate self._additional_arms and arms, so that `compute`
        # remains idempotent
        self.additional_arms = self._additional_arms + arms

        optimization_config = none_throws(experiment.optimization_config)

        df = prepare_arm_data(
            experiment=experiment,
            metric_names=[*optimization_config.metrics.keys()],
            adapter=relevant_adapter,
            use_model_predictions=True,
            relativize=self.relativize,
            additional_arms=self.additional_arms,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
        )

        objective_name = optimization_config.objective.metric.name

        fig = _prepare_figure_scatter(
            df=df,
            x_metric_name=objective_name,
            y_metric_name="p_feasible",
            x_metric_label=self.label if self.label is not None else objective_name,
            y_metric_label="% Chance of Satisfying the Constraints",
            is_relative=self.relativize,
            show_pareto_frontier=False,
            x_lower_is_better=optimization_config.objective.minimize,
            y_lower_is_better=False,
        )

        return create_plotly_analysis_card(
            name="ObjectivePFeasibleFrontierPlot",
            title=(
                f"Modeled {'Relativized ' if self.relativize else ''} Effect on the"
                " Objective vs % Chance of Satisfying the Constraints"
            ),
            subtitle=OBJ_PFEAS_CARDGROUP_SUBTITLE,
            df=df,
            fig=fig,
        )
